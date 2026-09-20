import pytest
import numpy as np
from cv_pipeline.detector import Detector, DetectionResult

# Mock the YOLO model to avoid loading weights in tests
class MockYOLOBox:
    def __init__(self, cls_id, conf, xyxy, id_val=None):
        self.cls = [np.array(cls_id)]
        self.conf = [np.array(conf)]
        self.xyxy = [np.array(xyxy)]
        self.id = [np.array(id_val)] if id_val is not None else None

class MockYOLOResult:
    def __init__(self, boxes):
        self.boxes = boxes

class MockYOLOModel:
    def __init__(self):
        self.names = {0: "person", 1: "chair"}
        
    def set_classes(self, classes):
        self.names = {i: c for i, c in enumerate(classes)}
        
    def track(self, frame, persist, tracker, conf, verbose):
        # Return a mock result
        b1 = MockYOLOBox(0, 0.9, [0, 0, 10, 10], id_val=1)
        b2 = MockYOLOBox(1, 0.8, [20, 20, 30, 30], id_val=None) # No ID yet
        return [MockYOLOResult([b1, b2])]

def test_detector(mocker):
    # Mock ultralytics YOLOE and YOLO
    mocker.patch("cv_pipeline.detector.YOLOE", return_value=MockYOLOModel())
    mocker.patch("cv_pipeline.detector.YOLO", return_value=MockYOLOModel())
    
    detector = Detector("dummy.pt", conf_thresh=0.4, tracker_config="dummy.yaml", scene_mode="outdoor")
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = detector.detect_and_track(dummy_frame)
    
    assert len(detections) == 2
    
    # Check first detection (has ID)
    d1 = detections[0]
    assert d1.object_id == 1
    assert d1.class_name.lower() == "pothole"
    assert d1.confidence == 0.9
    assert d1.bbox_xyxy == (0, 0, 10, 10)
    
    # Check second detection (no ID, should get temp ID)
    d2 = detections[1]
    assert d2.object_id == -1 # First temp ID
    assert d2.class_name.lower() == "road crater"
