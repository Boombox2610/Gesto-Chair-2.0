import os
import pickle
import shutil
import cv2
import numpy as np
import face_recognition
from typing import List, Tuple, Any
from collections import namedtuple

FaceResult = namedtuple("FaceResult", ["object_id", "name", "match_confidence"])

class FaceDB:
    """Manages the face encoding database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def load(self) -> dict:
        """Load DB from disk."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                return pickle.load(f)
        return {"encodings": [], "names": []}
        
    def save(self, db: dict):
        """Atomic save to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        temp_path = f"{self.db_path}.tmp"
        with open(temp_path, "wb") as f:
            pickle.dump(db, f)
        # Atomic rename
        shutil.move(temp_path, self.db_path)
        
    def enroll(self, name: str, image_path: str):
        """Enroll a new face from an image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        
        if not encodings:
            raise ValueError(f"No face found in {image_path}")
            
        db = self.load()
        db["encodings"].append(encodings[0])
        db["names"].append(name)
        self.save(db)
        
    def delete(self, name: str) -> bool:
        """Delete a face from the DB. Returns True if found and deleted."""
        db = self.load()
        
        indices_to_delete = [i for i, n in enumerate(db["names"]) if n == name]
        if not indices_to_delete:
            return False
            
        # Delete in reverse order to not mess up indices
        for i in reversed(indices_to_delete):
            del db["names"][i]
            del db["encodings"][i]
            
        self.save(db)
        return True


class FaceRecognizer:
    """Wraps face_recognition (dlib) for pipeline inference."""
    
    def __init__(self, db_path: str, tolerance: float = 0.5):
        self.db_path = db_path
        self.db = FaceDB(db_path).load()
        self.tolerance = tolerance
        self.last_db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0.0
        
    def recognize(
        self, 
        frame_bgr: np.ndarray, 
        person_bboxes: List[Tuple[int, Tuple[float, float, float, float]]]
    ) -> List[FaceResult]:
        """
        Run face recognition on cropped person ROIs.
        person_bboxes is a list of (object_id, (x1, y1, x2, y2))
        """
        # Dynamic reload if database was updated by enrollment controller
        if os.path.exists(self.db_path):
            current_mtime = os.path.getmtime(self.db_path)
            if current_mtime > self.last_db_mtime:
                self.db = FaceDB(self.db_path).load()
                self.last_db_mtime = current_mtime

        results = []
        if not self.db["names"]:
            return results # Nothing to match against
            
        h, w = frame_bgr.shape[:2]
        
        for object_id, bbox in person_bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure bounds are valid
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if y2 <= y1 or x2 <= x1:
                continue
                
            crop_bgr = frame_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                continue
                
            # Resize crop to speed up dlib. Max dimension 256.
            max_dim = max(crop_bgr.shape[0], crop_bgr.shape[1])
            if max_dim > 256:
                scale = 256.0 / max_dim
                small_crop = cv2.resize(crop_bgr, (0, 0), fx=scale, fy=scale)
            else:
                small_crop = crop_bgr
                
            small_rgb = cv2.cvtColor(small_crop, cv2.COLOR_BGR2RGB)
            
            # Find face locations in the crop (0 upsamples for max speed)
            face_locations = face_recognition.face_locations(small_rgb, number_of_times_to_upsample=0)
            if not face_locations:
                continue
                
            # Compute encodings
            face_encodings = face_recognition.face_encodings(small_rgb, face_locations)
            
            # Match against DB
            # We only care about the first found face in the crop to associate with the person bounding box
            encoding = face_encodings[0] 
            
            # Get distances to all known faces
            distances = face_recognition.face_distance(self.db["encodings"], encoding)
            if len(distances) == 0:
                continue
                
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            
            if best_distance <= self.tolerance:
                name = self.db["names"][best_match_index]
                match_confidence = "high" if best_distance < 0.4 else "low"
                
                results.append(FaceResult(
                    object_id=object_id,
                    name=name,
                    match_confidence=match_confidence
                ))
                
        return results
