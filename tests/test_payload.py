import pytest
from cv_pipeline.payload import PayloadBuilder
from cv_pipeline.detector import DetectionResult
from cv_pipeline.face_recognizer import FaceResult
from cv_pipeline.ocr import OCRResult

def test_payload_builder_schema():
    builder = PayloadBuilder(sink_mode="stdout")
    
    detections = [DetectionResult(object_id=1, class_name="Person", confidence=0.9, bbox_xyxy=(0,0,10,10))]
    zones = ["Center"]
    depth_scores = [0.8]
    
    face_results = [FaceResult(object_id=1, name="Alice", match_confidence="high")]
    ocr_results = [OCRResult(text="EXIT", confidence=0.95)]
    errors = ["Test error"]
    
    payload = builder.build(
        frame_id=10,
        timestamp=1000.5,
        status="degraded",
        fps=12.5,
        detections=detections,
        zones=zones,
        depth_scores=depth_scores,
        face_results=face_results,
        ocr_results=ocr_results,
        errors=errors
    )
    
    # Assert top-level keys
    assert payload["schema_version"] == "1.0"
    assert payload["frame_id"] == 10
    assert payload["timestamp"] == 1000.5
    assert payload["status"] == "degraded"
    assert payload["fps"] == 12.5
    
    # Assert scene understanding
    assert len(payload["scene_understanding"]) == 1
    su = payload["scene_understanding"][0]
    assert su["object_id"] == 1
    assert su["class"] == "Person"
    assert su["confidence"] == 0.9
    assert su["position"] == "Center"
    assert su["depth_score"] == 0.8
    assert su["distance_confidence"] == "relative"
    assert su["priority"] == "standard"
    
    # Assert face recognition
    assert len(payload["recognized_faces"]) == 1
    face = payload["recognized_faces"][0]
    assert face["object_id"] == 1
    assert face["name"] == "Alice"
    assert face["match_confidence"] == "high"
    
    # Assert OCR
    assert len(payload["environmental_text"]) == 1
    ocr = payload["environmental_text"][0]
    assert ocr["text"] == "EXIT"
    assert ocr["confidence"] == 0.95
    
    # Assert errors
    assert len(payload["errors"]) == 1
    assert payload["errors"][0] == "Test error"
