import os
import time
import re
import logging
from typing import Dict, Any, Optional, Tuple
import cv2

try:
    import face_recognition
except ImportError:
    face_recognition = None

from src.cv_pipeline.face_recognizer import FaceDB
from .nlp_config import config
from .tts import tts_manager
from .vlm import vlm_engine

logger = logging.getLogger("nlp_engine.enrollment")


class EnrollmentController:
    """
    Manages interactive voice-driven face enrollment when unknown persons are detected.
    """

    STATE_IDLE = "IDLE"
    STATE_AWAIT_CONFIRMATION = "AWAIT_CONFIRMATION"
    STATE_AWAIT_NAME = "AWAIT_NAME"
    STATE_ENROLLING = "ENROLLING"

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.face_db_path
        self.faces_dir = getattr(config, "faces_dir", "data/sample_faces")
        os.makedirs(self.faces_dir, exist_ok=True)

        self.state = self.STATE_IDLE
        self.state_start_time = 0.0
        self.state_timeout = 12.0  # Seconds to wait for voice input before timing out

        self.current_candidate: Optional[Dict[str, Any]] = None
        self.cooldowns: Dict[int, float] = {}  # object_id -> timestamp
        self.last_prompt_time = 0.0

        # Affirmatives and Negatives for yes/no responses
        self.affirmative_keywords = [
            "yes", "yeah", "yep", "sure", "add", "okay", "ok", 
            "haan", "ha", "sahi", "zaroor", "jaroor", "thik", "bilkul"
        ]
        self.negative_keywords = [
            "no", "nope", "skip", "cancel", "don't", "dont", "leave",
            "nahi", "na", "nathi", "mat", "rehne do"
        ]

    def is_active(self) -> bool:
        """Returns True if the user is currently in an enrollment dialog."""
        return self.state in [
            self.STATE_AWAIT_CONFIRMATION,
            self.STATE_AWAIT_NAME,
            self.STATE_ENROLLING
        ]

    def process_payload(self, payload: Dict[str, Any]):
        """
        Called on every CV payload frame. Handles timeouts and detects unknown people.
        """
        if not getattr(config, "enrollment_enabled", True):
            return

        now = time.time()

        # Handle timeout in active states
        if self.state != self.STATE_IDLE:
            if now - self.state_start_time > self.state_timeout:
                logger.info(f"Enrollment dialog timed out in state {self.state}.")
                self._cancel()
            return

        # Check for unknown person in IDLE state
        from .vlm import vlm_engine
        if tts_manager.is_speaking or getattr(vlm_engine, "is_busy", False):
            return  # Do not interrupt active speech or visual query reasoning

        # Global cooldown between any enrollment prompts to prevent nag loops
        global_cooldown = getattr(config, "enrollment_global_cooldown", 45.0)
        if now - self.last_prompt_time < global_cooldown:
            return

        items = payload.get("scene_understanding", [])
        faces = payload.get("recognized_faces", [])
        known_object_ids = {f.get("object_id") for f in faces if f.get("object_id") is not None}

        thresh = getattr(config, "enrollment_depth_thresh", 0.5)

        for item in items:
            cls = item.get("class", "").lower()
            if cls in ["person", "pedestrian", "child"]:
                obj_id = item.get("object_id")
                depth = item.get("depth_score", 0.0)

                # Must be close enough and not already recognized
                if depth >= thresh and obj_id not in known_object_ids:
                    # Check per-ID cooldown
                    if obj_id in self.cooldowns and now < self.cooldowns[obj_id]:
                        continue

                    # Found an eligible unknown candidate!
                    self._trigger_candidate(item)
                    break

    def _trigger_candidate(self, candidate: Dict[str, Any]):
        """Initiate the enrollment prompt for an unknown person candidate."""
        obj_id = candidate.get("object_id")
        pos = candidate.get("position", "ahead")
        now = time.time()

        # Set 60s cooldown immediately so we don't spam if ignored
        cooldown_sec = getattr(config, "enrollment_cooldown", 60)
        self.cooldowns[obj_id] = now + cooldown_sec
        self.last_prompt_time = now

        self.current_candidate = candidate
        self.state = self.STATE_AWAIT_CONFIRMATION
        self.state_start_time = now

        lang = config.language
        pos_display = {
            "Center": "Ahead",
            "Center-Left": "Center-Left",
            "Center-Right": "Center-Right",
            "Left": "Left",
            "Right": "Right"
        }.get(pos, pos)

        if lang == "hi":
            prompt = f"{pos_display} तरफ एक अज्ञात व्यक्ति है। क्या आप उन्हें जोड़ना चाहते हैं?"
        elif lang == "gu":
            prompt = f"{pos_display} બાજુ એક અજાણી વ્યક્તિ છે. શું તમે તેમને ઉમેરવા માંગો છો?"
        else:
            prompt = f"Unknown person detected on your {pos_display}. Would you like to add them?"

        logger.info(f"Triggering face enrollment prompt for object_id {obj_id}: '{prompt}'")
        tts_manager.speak(prompt, language=lang)

    def handle_voice_input(self, text: str, language: str = None) -> bool:
        """
        Process transcribed voice input during an active enrollment dialog.
        Returns True if the voice was consumed by the enrollment flow.
        """
        if not self.is_active():
            return False

        if language is None:
            language = config.language

        text_clean = text.lower().strip()
        logger.info(f"Enrollment input in state {self.state}: '{text_clean}'")

        if self.state == self.STATE_AWAIT_CONFIRMATION:
            return self._handle_confirmation(text_clean, language)

        elif self.state == self.STATE_AWAIT_NAME:
            return self._handle_name(text_clean, language)

        return False

    def _handle_confirmation(self, text: str, lang: str) -> bool:
        """Handle Yes/No response for whether to enroll the person."""
        # If user asks a scene question instead of answering yes/no, cancel enrollment and let Reasoner answer
        question_words = [
            "what", "how", "who", "where", "why", "describe", "scene", "happening",
            "around", "hazard", "danger", "safe", "person", "people", "fps", "sign",
            "read", "kya", "kaun", "kaisa", "batao", "surrounding"
        ]
        if any(w in text for w in question_words):
            logger.info(f"User asked a question ('{text}') during enrollment prompt. Cancelling enrollment and routing to Scene Reasoner.")
            self.state = self.STATE_IDLE
            self.current_candidate = None
            return False

        # Check negative first
        for kw in self.negative_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                self._cancel(lang)
                return True

        # Check affirmative
        for kw in self.affirmative_keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text):
                self.state = self.STATE_AWAIT_NAME
                self.state_start_time = time.time()

                if lang == "hi":
                    prompt = "उनका नाम क्या है?"
                elif lang == "gu":
                    prompt = "તેમનું નામ શું છે?"
                else:
                    prompt = "What is their name?"

                logger.info(f"Confirmation received. Asking for name: '{prompt}'")
                tts_manager.speak(prompt, language=lang)
                return True

        # Ambiguous response
        rep = "Please say yes to add them, or no to cancel."
        if lang == "hi": rep = "कृपया जोड़ने के लिए हाँ, या रद्द करने के लिए ना कहें।"
        elif lang == "gu": rep = "કૃપા કરીને ઉમેરવા માટે હા, અથવા રદ કરવા માટે ના કહો."
        tts_manager.speak(rep, language=lang)
        self.state_start_time = time.time()  # refresh timeout
        return True

    def _handle_name(self, text: str, lang: str) -> bool:
        """Extract name from speech and enroll the person."""
        text_clean = text.strip()

        # Reject monologues, questions, or excessively long sentences (> 4 words or > 30 chars)
        words = text_clean.split()
        is_question = any(q in text_clean.lower() for q in ["?", "what", "who", "where", "why", "how", "phone", "video"])
        if len(words) > 4 or len(text_clean) > 30 or is_question:
            logger.warning(f"Rejected invalid enrollment name input: '{text_clean}'")
            if lang == "hi":
                prompt = "कृपया केवल व्यक्ति का पहला और अंतिम नाम बोलें।"
            elif lang == "gu":
                prompt = "કૃપા કરીને માત્ર વ્યક્તિનું નામ બોલો."
            else:
                prompt = "Please say just the first and last name."
            tts_manager.speak(prompt, language=lang)
            self.state_start_time = time.time()
            return True

        # Clean common prefixes (e.g., "his name is rahul", "unka naam amit hai")
        cleaned_name = text_clean
        prefixes = [
            r"^his name is\s+", r"^her name is\s+", r"^their name is\s+",
            r"^the name is\s+", r"^name is\s+", r"^my friend\s+", r"^it is\s+",
            r"^call him\s+", r"^call her\s+", r"^i am\s+",
            r"^unka naam\s+", r"^naam\s+", r"^iska naam\s+", r"^aenu naam\s+"
        ]
        for p in prefixes:
            cleaned_name = re.sub(p, "", cleaned_name, flags=re.IGNORECASE)

        suffixes = [r"\s+hai$", r"\s+che$"]
        for s in suffixes:
            cleaned_name = re.sub(s, "", cleaned_name, flags=re.IGNORECASE)

        # Remove filler repetitions (mh, uh, um, ah) and punctuation
        cleaned_name = re.sub(r'\b(mh|uh|um|ah|oh)\b', '', cleaned_name, flags=re.IGNORECASE)
        cleaned_name = re.sub(r'[^a-zA-Z\s\.\-]', '', cleaned_name).strip()
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name).title()

        if len(cleaned_name) < 2:
            prompt = "I didn't catch the name. Please say their name."
            if lang == "hi": prompt = "मुझे नाम समझ नहीं आया। कृपया दोबारा नाम बोलें।"
            elif lang == "gu": prompt = "મને નામ સમજાયું નહીં. કૃપા કરીને ફરીથી નામ બોલો."
            tts_manager.speak(prompt, language=lang)
            self.state_start_time = time.time()
            return True

        self.state = self.STATE_ENROLLING
        ack = f"Adding {cleaned_name}..."
        if lang == "hi": ack = f"{cleaned_name} को जोड़ा जा रहा है..."
        elif lang == "gu": ack = f"{cleaned_name} ને ઉમેરવામાં આવી રહ્યા છે..."
        tts_manager.speak(ack, language=lang)

        # Execute enrollment in background/sync
        success, msg = self._execute_enrollment(cleaned_name, lang)
        tts_manager.speak(msg, language=lang)

        self.state = self.STATE_IDLE
        self.current_candidate = None
        return True

    def _execute_enrollment(self, name: str, lang: str) -> Tuple[bool, str]:
        """Capture frame, crop face, extract encoding, and save to FaceDB."""
        if not face_recognition:
            return False, "Face recognition module is not installed."

        # Request fresh frame via pull architecture
        frame_path = vlm_engine._request_frame(timeout=2.0)
        if not frame_path or not os.path.exists(frame_path):
            if lang == "hi": return False, "कैमरे से फ़्रेम प्राप्त नहीं हो सका।"
            return False, "Could not capture frame from the camera."

        try:
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                return False, "Failed to read captured frame."

            h, w = frame_bgr.shape[:2]
            crop_bgr = None

            # 1. Try cropping from bbox if candidate has one
            if self.current_candidate and "bbox" in self.current_candidate:
                bbox = self.current_candidate["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                # Expand box by 15% margin for head inclusion
                bw, bh = x2 - x1, y2 - y1
                x1 = max(0, int(x1 - 0.15 * bw))
                y1 = max(0, int(y1 - 0.15 * bh))
                x2 = min(w, int(x2 + 0.15 * bw))
                y2 = min(h, int(y2 + 0.15 * bh))

                if y2 > y1 and x2 > x1:
                    crop_bgr = frame_bgr[y1:y2, x1:x2]

            # Convert to RGB for face_recognition
            target_rgb = cv2.cvtColor(crop_bgr if crop_bgr is not None else frame_bgr, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(target_rgb)

            # If no face found in crop, fall back to searching full frame
            if not locations and crop_bgr is not None:
                target_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                locations = face_recognition.face_locations(target_rgb)

            if not locations:
                if lang == "hi":
                    return False, "चेहरा साफ़ नहीं दिखा। कृपया उनके सामने होकर पुनः प्रयास करें।"
                elif lang == "gu":
                    return False, "ચહેરો સ્પષ્ટ દેખાતો નથી. કૃપા કરીને ફરી પ્રયાસ કરો."
                return False, "Could not see a face clearly. Please try again when facing them."

            # Compute encoding
            encodings = face_recognition.face_encodings(target_rgb, locations)
            if not encodings:
                return False, "Could not compute face encoding."

            # Save sample crop image
            top, right, bottom, left = locations[0]
            face_crop_rgb = target_rgb[top:bottom, left:right]
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)

            safe_name = re.sub(r'[^a-zA-Z0-9_]', '', name.lower())
            sample_filename = f"{safe_name}_{int(time.time())}.jpg"
            sample_path = os.path.join(self.faces_dir, sample_filename)
            cv2.imwrite(sample_path, face_crop_bgr)

            # Enroll into FaceDB
            db = FaceDB(self.db_path)
            current_db = db.load()
            current_db["encodings"].append(encodings[0])
            current_db["names"].append(name)
            db.save(current_db)

            logger.info(f"Successfully enrolled face '{name}' saved to {sample_path} and DB {self.db_path}")

            if lang == "hi":
                return True, f"{name} को जोड़ दिया गया है।"
            elif lang == "gu":
                return True, f"{name} ને ઉમેરી દેવામાં આવ્યા છે."
            return True, f"{name} has been added to your recognized faces."

        except Exception as e:
            logger.error(f"Error during enrollment execution: {e}", exc_info=True)
            return False, f"Failed to enroll {name}."

    def _cancel(self, lang: str = "en"):
        """Cancel current enrollment dialog."""
        self.state = self.STATE_IDLE
        self.current_candidate = None
        if lang == "hi":
            tts_manager.speak("रद्द कर दिया गया।", language=lang)
        elif lang == "gu":
            tts_manager.speak("રદ કરવામાં આવ્યું.", language=lang)
        else:
            tts_manager.speak("Enrollment cancelled.", language=lang)


# Global instance
enrollment_controller = EnrollmentController()
