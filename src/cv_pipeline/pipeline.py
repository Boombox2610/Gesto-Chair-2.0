import os
import sys
import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List

from .config import config
from .capture import WebcamCapture
from .detector import Detector
from .depth import DepthEstimator
from .spatial import SpatialMapper
from .face_recognizer import FaceRecognizer
from .ocr import OCRProcessor, EasyOCRBackend
from .payload import PayloadBuilder
from .vocabulary import HAZARD_CLASSES, AGENT_CLASSES, OCR_TRIGGER_CLASSES, PRIORITY_CLASSES


class Pipeline:
    """Main orchestration loop for the CV Pipeline."""
    
    def __init__(self):
        self.device = config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
            self.device = "cpu"
            
        print(f"[INFO] Initializing pipeline on {self.device}...", file=sys.stderr)
        
        # Initialize modules
        self.detector = Detector(config.yoloe_model, config.yolo_conf_thresh, config.tracker, config.scene_mode, config.offline, is_legacy=False)
        self.depth_estimator = DepthEstimator(config.midas_model, self.device)
        self.spatial_mapper = SpatialMapper(config.frame_width, config.spatial_zones)
        self.face_recognizer = FaceRecognizer(config.face_db_path, config.face_tolerance)
        self.ocr_processor = OCRProcessor(
            EasyOCRBackend(gpu=(self.device == "cuda")), 
            config.ocr_conf_thresh
        )
        self.payload_builder = PayloadBuilder(config.payload_sink, config.payload_file)
        
        # Thread pool for CPU-bound or parallelizable tasks (Face, OCR)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Dynamic interleaving state
        self.current_depth_interval = config.depth_fallback_interval
        # OCR is now purely event-driven, so interval is irrelevant. Keeping variable for fallback compatibility if needed.
        self.current_ocr_interval = config.ocr_every_n_frames
        self.degraded_frames = 0
        self.healthy_frames = 0
        self.is_degraded = False
        
        # Cached results for skipping
        self.last_ocr_results = []
        self.last_face_results = []
        self.active_face_future = None
        self.active_ocr_future = None
        
    def _manage_degradation(self, fps_monitor):
        """Monitor FPS and adjust interleaving if falling behind."""
        if fps_monitor.is_degraded(config.fps_target):
            self.degraded_frames += 1
            self.healthy_frames = 0
            if self.degraded_frames >= 5 and not self.is_degraded:
                self.is_degraded = True
                self.current_depth_interval = config.depth_fallback_interval + 2
                self.current_ocr_interval = 20
                print(f"[WARN] Pipeline degraded. FPS={fps_monitor.current_fps:.1f}. Adjusting intervals.", file=sys.stderr)
        else:
            self.healthy_frames += 1
            self.degraded_frames = 0
            if self.healthy_frames >= 10 and self.is_degraded:
                self.is_degraded = False
                self.current_depth_interval = config.depth_fallback_interval
                self.current_ocr_interval = config.ocr_every_n_frames
                print(f"[INFO] Pipeline recovered. FPS={fps_monitor.current_fps:.1f}. Restoring intervals.", file=sys.stderr)
                
    def _draw_debug_window(self, frame_bgr, detections, zones, depth_scores, face_results, ocr_results, fps, status):
        """Draw annotations on a copy of the frame for the OpenCV debug window."""
        annotated = frame_bgr.copy()
        
        # Draw status and FPS
        color = (0, 255, 0) if status == "ok" else (0, 165, 255)
        cv2.putText(annotated, f"FPS: {fps:.1f} | {status.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Map face names by object_id
        face_map = {f.object_id: f.name for f in face_results}
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            name = face_map.get(det.object_id, det.class_name)
            
            label = f"[{det.object_id}] {name} ({zones[i]}) D:{depth_scores[i]:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow("Gesto-Drive CV Pipeline", annotated)
        # Also write the latest frame for Streamlit atomically, max 4 times a second
        if config.show_dashboard:
            import time
            current_time = time.time()
            if not hasattr(self, 'last_dashboard_write') or (current_time - self.last_dashboard_write) >= 0.25:
                temp_file = "latest_frame_tmp.jpg"
                cv2.imwrite(temp_file, annotated)
                try:
                    os.replace(temp_file, "latest_frame.jpg")
                except OSError:
                    pass # Safe fallback on Windows if file lock is held
                self.last_dashboard_write = current_time

            
    def _should_run_depth(self, detections, frame_id):
        """Semantic conditional gate for MiDaS depth."""
        frame_area = config.frame_width * config.frame_height
        for det in detections:
            w = det.bbox_xyxy[2] - det.bbox_xyxy[0]
            h = det.bbox_xyxy[3] - det.bbox_xyxy[1]
            area_ratio = (w * h) / frame_area
            if det.class_name.lower() in HAZARD_CLASSES and area_ratio > config.depth_hazard_area_thresh:
                return True
            if det.class_name.lower() in AGENT_CLASSES and area_ratio > config.depth_agent_area_thresh:
                return True
        return frame_id % self.current_depth_interval == 0

    def run(self):
        """Start the pipeline loop."""
        
        print("[INFO] Starting pipeline loop...", file=sys.stderr)
        
        profile_mode = os.environ.get("PIPELINE_PROFILE") == "1"
        import time

        with WebcamCapture(config.camera_index, config.frame_width, config.frame_height) as cap:
            print(f"[INFO] Pipeline running on input source: {config.camera_index}", file=sys.stderr)
            while True:
                t_start = time.time()
                
                success, frame, frame_id, timestamp = cap.read()
                t_read = time.time()
                
                if not success:
                    print("[ERROR] Webcam read failed.", file=sys.stderr)
                    break
                    
                # Frame Request Handshake (Pull Architecture)
                has_image_this_frame = False
                request_flag = "frame_request.flag"
                if os.path.exists(request_flag):
                    cv2.imwrite("latest_frame.jpg", frame)
                    has_image_this_frame = True
                    try:
                        os.remove(request_flag)
                    except OSError:
                        pass
                    
                errors = []
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. Object Detection (YOLO + ByteTrack)
                try:
                    detections = self.detector.detect_and_track(frame)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    errors.append("CUDA OOM on YOLO — skipping frame")
                    continue
                t_yolo = time.time()
                    
                detected_classes = [det.class_name for det in detections]
                
                # 2. Depth Estimation (MiDaS) - Conditional Trigger
                ran_depth_this_frame = False
                if self._should_run_depth(detections, frame_id):
                    ran_depth_this_frame = True
                    try:
                        self.depth_estimator.infer(frame_rgb)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        errors.append("CUDA OOM on MiDaS — fell back to cached depth")
                t_depth = time.time()
                        
                # 3. Spatial Zones & ROI Depth
                zones = []
                depth_scores = []
                person_bboxes = []
                
                for det in detections:
                    zones.append(self.spatial_mapper.get_zone(det.bbox_xyxy))
                    depth_scores.append(self.depth_estimator.get_roi_depth(det.bbox_xyxy))
                    if det.class_name.lower() in AGENT_CLASSES or det.class_name.lower() == "person":
                        person_bboxes.append((det.object_id, det.bbox_xyxy))
                t_spatial = time.time()
                        
                # 4. Face Recognition (Threaded, Gated, Skipped)
                if self.active_face_future is None:
                    if person_bboxes and frame_id % config.face_every_n_frames == 0:
                        self.active_face_future = self.executor.submit(self.face_recognizer.recognize, frame.copy(), person_bboxes)
                        
                # 5. OCR (Threaded, Skipped, ROI-triggered)
                if self.active_ocr_future is None:
                    # Purely event-driven OCR
                    if any(cls.lower() in OCR_TRIGGER_CLASSES for cls in detected_classes):
                        self.active_ocr_future = self.executor.submit(self.ocr_processor.run, frame_rgb.copy())
                        
                # Collect thread results (Non-blocking)
                if self.active_face_future is not None:
                    if self.active_face_future.done():
                        try:
                            self.last_face_results = self.active_face_future.result()
                        except Exception as e:
                            errors.append(f"Face recognition error: {e}")
                        self.active_face_future = None
                        
                if self.active_ocr_future is not None:
                    if self.active_ocr_future.done():
                        try:
                            self.last_ocr_results = self.active_ocr_future.result()
                        except Exception as e:
                            errors.append(f"OCR error: {e}")
                        self.active_ocr_future = None
                t_threads = time.time()
                        
                # 6. Status and FPS
                cap.fps_monitor.tick()
                self._manage_degradation(cap.fps_monitor)
                status = "degraded" if self.is_degraded else "ok"
                if errors:
                    status = "error"
                    
                payload = self.payload_builder.build(
                    frame_id, timestamp, status, cap.fps_monitor.current_fps,
                    detections, zones, depth_scores, 
                    self.last_face_results, self.last_ocr_results, errors,
                    has_image=has_image_this_frame
                )
                self.payload_builder.emit(payload)
                
                # 8. Visualization
                if config.show_cv_window:
                    self._draw_debug_window(
                        frame, detections, zones, depth_scores, 
                        self.last_face_results, self.last_ocr_results, 
                        cap.fps_monitor.current_fps, status
                    )
                    
                if frame_id > 5 and (cv2.waitKey(1) & 0xFF == ord('q')):
                    print("[INFO] 'q' pressed. Shutting down...", file=sys.stderr)
                    # Emit final shutdown payload
                    payload["status"] = "shutdown"
                    self.payload_builder.emit(payload)
                    break
                    
                if profile_mode:
                    t_end = time.time()
                    ms_read = (t_read - t_start) * 1000
                    ms_yolo = (t_yolo - t_read) * 1000
                    ms_depth = (t_depth - t_yolo) * 1000
                    ms_spat = (t_spatial - t_depth) * 1000
                    ms_threads = (t_threads - t_spatial) * 1000
                    ms_out = (t_end - t_threads) * 1000
                    ms_total = (t_end - t_start) * 1000
                    d_run = f"{ms_depth:.1f}" if ran_depth_this_frame else "--"
                    
                    print(f"[PROFILE] frame={frame_id} read={ms_read:.1f}ms yolo={ms_yolo:.1f}ms "
                          f"depth={d_run}ms spat={ms_spat:.1f}ms threads={ms_threads:.1f}ms "
                          f"out={ms_out:.1f}ms total={ms_total:.1f}ms", file=sys.stderr)
                    
        self.executor.shutdown(wait=False)
        cv2.destroyAllWindows()
