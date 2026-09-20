from collections import namedtuple
from typing import List, Any
import numpy as np
from ultralytics import YOLOE, YOLO
import os
import pickle
from .vocabulary import get_vocabulary

DetectionResult = namedtuple("DetectionResult", ["object_id", "class_name", "confidence", "bbox_xyxy"])

class Detector:
    """Wrapper for YOLOv8n object detection and ByteTrack tracking."""
    
    def __init__(self, model_path: str, conf_thresh: float, tracker_config: str, scene_mode: str = "outdoor", offline: bool = False, is_legacy: bool = False):
        if is_legacy:
            self.model = YOLO(model_path)
            self.is_legacy = True
        else:
            self.model = YOLOE(model_path)
            self.is_legacy = False
        
        self.conf_thresh = conf_thresh
        self.tracker_config = tracker_config
        self.offline = offline
        self.scene_mode = scene_mode
        self.active_classes = []
        
        if not self.is_legacy:
            self.set_scene_mode(scene_mode)
        
        # To handle frames where tracker hasn't assigned an ID yet
        self._temp_id_counter = -1
        
    def set_scene_mode(self, mode: str):
        """Update active vocabulary dynamically at runtime."""
        if self.is_legacy:
            return
            
        self.scene_mode = mode
        self.active_classes = get_vocabulary(mode)
        
        if self.offline:
            # Phase 4 offline embeddings loading
            embeddings_path = os.path.join("data", "yoloe_vocab_embeddings.pkl")
            if os.path.exists(embeddings_path):
                with open(embeddings_path, "rb") as f:
                    embeddings_cache = pickle.load(f)
                    
                if mode in embeddings_cache:
                    # In ultralytics YOLOE, you can typically pass precomputed embeds 
                    # but for now we just fallback to set_classes which might fail without internet
                    # if text encoder is not cached.
                    self.model.set_classes(self.active_classes)
            else:
                self.model.set_classes(self.active_classes)
        else:
            self.model.set_classes(self.active_classes)
            
    def get_active_classes(self):
        if self.is_legacy:
            return list(self.model.names.values())
        return self.active_classes
        
    def detect_and_track(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Run inference and tracking on a frame.
        Returns a list of DetectionResult.
        """
        # Run tracking. persist=True tells the tracker to remember across frames.
        results = self.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_config, 
            conf=self.conf_thresh, 
            verbose=False
        )
        
        detections = []
        if not results or not len(results):
            return detections
            
        result = results[0]
        
        # If no boxes detected
        if result.boxes is None or len(result.boxes) == 0:
            return detections
            
        # Extract data
        boxes = result.boxes
        
        for i in range(len(boxes)):
            box = boxes[i]
            
            # Get class name
            cls_id = int(box.cls[0].item())
            class_name = self.model.names[cls_id].capitalize()
            
            # Get confidence
            conf = float(box.conf[0].item())
            
            # Get bbox
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Get track ID (may be None in first few frames)
            if box.id is not None:
                track_id = int(box.id[0].item())
            else:
                track_id = self._temp_id_counter
                self._temp_id_counter -= 1
                
            detections.append(DetectionResult(
                object_id=track_id,
                class_name=class_name,
                confidence=conf,
                bbox_xyxy=(x1, y1, x2, y2)
            ))
            
        return detections
