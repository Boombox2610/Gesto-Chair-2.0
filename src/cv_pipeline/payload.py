import json
import logging
from logging.handlers import RotatingFileHandler
import os
from typing import List, Dict, Any, Optional
import sys
from .vocabulary import PRIORITY_CLASSES

class PayloadBuilder:
    """Builds and emits the JSON schema v1.0 payload."""
    
    def __init__(self, sink_mode: str, file_path: str = "payloads.jsonl"):
        """
        sink_mode: 'stdout', 'file', or 'both'
        file_path: Path to the rolling JSONL file
        """
        self.sink_mode = sink_mode
        self.file_path = file_path
        
        # Setup file logger if needed
        self.file_logger = None
        if self.sink_mode in ["file", "both"]:
            self.file_logger = logging.getLogger("payload_logger")
            self.file_logger.setLevel(logging.INFO)
            # Max 50 MB, keep 3 backups
            handler = RotatingFileHandler(
                self.file_path, maxBytes=50*1024*1024, backupCount=3
            )
            # We only want the JSON message in the file, no standard log formatting
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.file_logger.addHandler(handler)
            self.file_logger.propagate = False
            
    def build(
        self, 
        frame_id: int, 
        timestamp: float, 
        status: str,
        fps: float,
        detections: List[Any], 
        zones: List[str],
        depth_scores: List[float],
        face_results: List[Any],
        ocr_results: List[Any],
        errors: List[str],
        has_image: bool = False
    ) -> Dict[str, Any]:
        """Constructs the JSON dict according to schema v1.0."""
        
        scene_understanding = []
        for i, det in enumerate(detections):
            label = det.class_name.lower()
            priority = PRIORITY_CLASSES.get(label, "contextual")
            item = {
                "object_id": det.object_id,
                "class": det.class_name,
                "confidence": round(det.confidence, 3),
                "position": zones[i],
                "depth_score": round(depth_scores[i], 3),
                "distance_confidence": "relative",
                "priority": priority
            }
            if hasattr(det, "bbox_xyxy") and det.bbox_xyxy is not None:
                item["bbox"] = [round(float(c), 1) for c in det.bbox_xyxy]
            scene_understanding.append(item)
            
        recognized_faces = []
        for face in face_results:
            recognized_faces.append({
                "object_id": face.object_id,
                "name": face.name,
                "match_confidence": face.match_confidence
            })
            
        environmental_text = []
        for ocr in ocr_results:
            environmental_text.append({
                "text": ocr.text,
                "confidence": round(ocr.confidence, 3)
            })
            
        payload = {
            "schema_version": "1.0",
            "frame_id": frame_id,
            "timestamp": round(timestamp, 3),
            "status": status,
            "fps": round(fps, 1),
            "scene_understanding": scene_understanding,
            "recognized_faces": recognized_faces,
            "environmental_text": environmental_text,
            "errors": errors,
            "has_image": has_image
        }
        
        return payload
        
    def emit(self, payload: Dict[str, Any]):
        """Emit payload to the configured sinks."""
        payload_str = json.dumps(payload)
        
        if self.sink_mode in ["stdout", "both"]:
            # Print and flush to ensure downstream consumers get it immediately
            print(payload_str)
            sys.stdout.flush()
            
        if self.sink_mode in ["file", "both"] and self.file_logger:
            self.file_logger.info(payload_str)
