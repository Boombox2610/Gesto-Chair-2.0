import time
import logging
from typing import Dict, Any, List

from .nlp_config import config
from .tts import tts_manager
from .enrollment import enrollment_controller

logger = logging.getLogger("nlp_engine.announcer")

class AnnouncementManager:
    def __init__(self):
        # State tracking for cooldowns and depth
        self.hazard_state = {}  # (class_name, position) -> {"last_time": float, "last_depth": float}
        self.hazard_class_state = {}  # class_name -> {"last_time": float, "last_depth": float}
        self.person_state = {}  # name -> {"last_time": float, "last_depth": float}
        self.ocr_state = {}     # text -> {"last_seen_time": float, "announced": bool}
        self.unknown_person_state = {} # (object_id, position) -> {"last_time": float}

    def process_payload(self, payload: Dict[str, Any]):
        """
        Evaluate the CV payload against trigger rules and announce if needed.
        """
        # Do not announce while user is holding F5 to speak or system is busy with VLM
        from .stt import stt_engine
        from .vlm import vlm_engine
        if stt_engine.is_recording or getattr(vlm_engine, "is_busy", False):
            return

        current_time = time.time()
        
        # Always process hazards, as critical hazards should interrupt
        self._process_hazards(payload, current_time)
        
        # Skip other proactive announcements if the system is speaking or in enrollment dialog
        if tts_manager.is_speaking or enrollment_controller.is_active():
            return
            
        self._process_persons(payload, current_time)
        self._process_ocr(payload, current_time)

    def _process_hazards(self, payload: Dict[str, Any], current_time: float):
        # 1. Deduplicate hazards within the frame by class: pick the closest (max depth_score) per class
        candidates_by_class: Dict[str, Dict[str, Any]] = {}
        for item in payload.get("scene_understanding", []):
            priority = item.get("priority")
            if priority not in ["critical", "standard"]:
                continue
                
            class_name = item.get("class", "hazard")
            if class_name.lower() in ["person", "pedestrian", "walking person", "standing person", "sitting person", "child"]:
                continue
                
            depth_score = item.get("depth_score", 0.0)
            if depth_score < config.announce_hazard_depth_thresh:
                continue

            if class_name not in candidates_by_class or depth_score > candidates_by_class[class_name].get("depth_score", 0.0):
                candidates_by_class[class_name] = item

        for class_name, item in candidates_by_class.items():
            priority = item.get("priority")
            depth_score = item.get("depth_score", 0.0)
            position = item.get("position", "ahead")
            key = (class_name, position)

            state = self.hazard_state.get(key)
            class_state = self.hazard_class_state.get(class_name)
            class_cooldown = getattr(config, "hazard_class_cooldown", 8)

            should_announce = False

            if not state:
                # If this specific position wasn't seen, check if the class was recently announced in any position
                if not class_state:
                    should_announce = True
                else:
                    time_since_class = current_time - class_state["last_time"]
                    depth_increase_class = depth_score - class_state["last_depth"]
                    # If announced recently (within class cooldown), only re-announce if significantly closer
                    if time_since_class > class_cooldown or depth_increase_class >= 0.20:
                        should_announce = True
            else:
                time_since_last = current_time - state["last_time"]
                depth_increase = depth_score - state["last_depth"]

                if time_since_last > config.announce_hazard_cooldown:
                    should_announce = True
                elif depth_increase >= config.announce_depth_increase_thresh:
                    should_announce = True

            if should_announce:
                # If the system is busy answering a user's question, ONLY allow critical hazards
                if tts_manager.is_speaking and priority != "critical":
                    continue

                self.hazard_state[key] = {"last_time": current_time, "last_depth": depth_score}
                self.hazard_class_state[class_name] = {"last_time": current_time, "last_depth": depth_score}
                text = self._format_hazard_text(class_name, position, config.language)
                tts_manager.speak(text, is_critical=(priority == "critical"))

    def _process_persons(self, payload: Dict[str, Any], current_time: float):
        recognized_obj_ids = set()
        
        for face in payload.get("recognized_faces", []):
            name = face.get("name")
            if not name:
                continue
                
            depth_score = face.get("depth_score", 0.0)
            position = face.get("position", "ahead")
            
            obj_id = face.get("object_id")
            if obj_id is not None:
                recognized_obj_ids.add(obj_id)
                
            if depth_score < config.announce_person_depth_thresh:
                continue
                
            state = self.person_state.get(name)
            should_announce = False
            
            if not state:
                should_announce = True
            else:
                time_since_last = current_time - state["last_time"]
                depth_increase = depth_score - state["last_depth"]
                
                if time_since_last > config.announce_person_cooldown:
                    should_announce = True
                elif depth_increase >= 0.15:
                    should_announce = True
                    
            if should_announce:
                self.person_state[name] = {"last_time": current_time, "last_depth": depth_score}
                text = self._format_person_text(name, position, config.language)
                tts_manager.speak(text, is_critical=False)
                
        # Handle unknown persons
        if getattr(config, "enrollment_enabled", True):
            enrollment_controller.process_payload(payload)
            return

        person_detections = [item for item in payload.get("scene_understanding", []) 
                             if item.get("class", "").lower() in ["person", "pedestrian", "walking person", "standing person", "sitting person", "child"]]
        
        for person in person_detections:
            obj_id = person.get("object_id")
            if obj_id in recognized_obj_ids:
                continue
                
            depth_score = person.get("depth_score", 0.0)
            if depth_score < config.announce_person_depth_thresh:
                continue
                
            position = person.get("position", "ahead")
            key = (obj_id, position)
            state = self.unknown_person_state.get(key)
            
            should_announce = False
            if not state:
                should_announce = True
            else:
                if current_time - state["last_time"] > config.announce_person_cooldown:
                    should_announce = True
                    
            if should_announce:
                self.unknown_person_state[key] = {"last_time": current_time}
                text = self._format_unknown_person_text(position, config.language)
                tts_manager.speak(text, is_critical=False)

    def _process_ocr(self, payload: Dict[str, Any], current_time: float):
        raw_texts = payload.get("environmental_text", [])
        current_texts = [t["text"] for t in raw_texts if isinstance(t, dict) and "text" in t]
        
        # Cleanup OCR state for texts not seen in 5+ seconds
        to_delete = []
        for text, state in self.ocr_state.items():
            if text not in current_texts and (current_time - state["last_seen_time"]) > 5.0:
                to_delete.append(text)
        for t in to_delete:
            del self.ocr_state[t]
            
        for text in current_texts:
            state = self.ocr_state.get(text)
            if not state:
                # First time seeing this text
                self.ocr_state[text] = {"last_seen_time": current_time, "announced": True}
                speech = self._format_ocr_text(text, config.language)
                tts_manager.speak(speech, is_critical=False)
            else:
                # Update last seen time, but don't re-announce
                state["last_seen_time"] = current_time

    def _format_hazard_text(self, class_name: str, position: str, lang: str) -> str:
        if lang == "hi":
            return f"सावधान। {class_name} आगे है, {position} तरफ।"
        elif lang == "gu":
            return f"સાવધાન। {class_name} આગળ છે, {position} બાજુ।"
        else:
            return f"Caution. {class_name.capitalize()} ahead, {position}."

    def _format_person_text(self, name: str, position: str, lang: str) -> str:
        if lang == "hi":
            return f"{name} पास में हैं, आपके {position} तरफ।"
        elif lang == "gu":
            return f"{name} નજીકમાં છે, તમારી {position} બાજુ।"
        else:
            return f"{name} is nearby, on your {position}."

    def _format_unknown_person_text(self, position: str, lang: str) -> str:
        if lang == "hi":
            return f"अज्ञात व्यक्ति आपके {position} तरफ है।"
        elif lang == "gu":
            return f"અજાણ્યો વ્યક્તિ તમારી {position} બાજુ છે।"
        else:
            return f"Unknown person detected on your {position}."

    def _format_ocr_text(self, text: str, lang: str) -> str:
        if lang == "hi":
            return f"बोर्ड पर लिखा है: {text}।"
        elif lang == "gu":
            return f"બોર્ડ પર લખ્યું છે: {text}。"
        else:
            return f"Sign reads: {text}."

# Global instance
announcer = AnnouncementManager()
