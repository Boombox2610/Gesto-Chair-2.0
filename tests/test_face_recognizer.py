import pytest
import numpy as np
import os
from cv_pipeline.face_recognizer import FaceRecognizer, FaceDB

def test_face_db_enroll_no_face(tmp_path, mocker):
    db_path = str(tmp_path / "known_faces.pkl")
    db = FaceDB(db_path)
    
    # Mock face_recognition to return empty encodings
    mocker.patch("cv_pipeline.face_recognizer.face_recognition.load_image_file", return_value=np.zeros((10,10,3)))
    mocker.patch("cv_pipeline.face_recognizer.face_recognition.face_encodings", return_value=[])
    
    with pytest.raises(ValueError, match="No face found"):
        # Image path must exist for the initial check to pass
        test_img = str(tmp_path / "test.jpg")
        with open(test_img, "w") as f:
            f.write("dummy")
            
        db.enroll("Alice", test_img)

def test_face_recognizer_gate(tmp_path):
    # Test that recognizer returns empty if no person bboxes are provided
    db_path = str(tmp_path / "known_faces.pkl")
    recognizer = FaceRecognizer(db_path)
    
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # No bboxes -> returns empty list
    results = recognizer.recognize(dummy_frame, person_bboxes=[])
    assert len(results) == 0
