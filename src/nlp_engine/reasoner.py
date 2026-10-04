import logging
import re
from typing import Dict, Any, Optional

from .nlp_config import config
from .tts import tts_manager
from .vlm import vlm_engine

logger = logging.getLogger("nlp_engine.reasoner")

class SceneReasoner:
    def __init__(self, ingester, vlm=None):
        self.ingester = ingester
        self.vlm = vlm or vlm_engine
        self.is_busy = False
        
        # Keyword dictionaries for intent matching
        self.keywords = {
            "SCENE_SUMMARY": ["around", "describe", "scene", "surroundings", "kya hai", "aas paas", "what is in front", "what's in front"],
            "HAZARD_CHECK": ["hazard", "danger", "khatara", "khatraa", "khatra"],
            "PEOPLE_CHECK": ["anyone", "person", "people", "who is", "kaun", "koi hai", "koi"],
            "OCR_READ": ["sign", "text", "read", "board", "likha", "padho", "written"],
            "LANG_SWITCH": ["switch to hindi", "hindi mein", "gujarati", "english", "speak in hindi", "speak in english", "speak in gujarati"],
            "STATUS": ["fps", "status", "speed", "performance"],
            "VLM_QUERY": ["color", "colour", "wearing", "carrying", "holding", "safe to cross", "cross the road", "detail", "look like"]
        }

    def answer(self, question: str, language: str = None) -> str:
        """Process a spoken question and return a spoken response."""
        self.is_busy = True
        try:
            if language is None:
                language = config.language
                
            question = question.lower()
            intent = self._classify_intent(question)
            logger.info(f"Question: '{question}' -> Intent: {intent}")
            
            if intent == "LANG_SWITCH":
                return self._handle_language_switch(question)
                
            latest_payload = self.ingester.history[-1] if self.ingester.history else None
            
            # If intent is VLM_QUERY or UNKNOWN and VLM is enabled, route to VLM
            if (intent in ["VLM_QUERY", "UNKNOWN"]) and config.vlm_enabled and self.vlm:
                return self._handle_vlm_query(latest_payload, language, question)
                
            # For structured template intents, we need the latest payload
            if not latest_payload:
                return self._no_data_response(language)
                
            handler = {
                "SCENE_SUMMARY": self._handle_scene_summary,
                "HAZARD_CHECK": self._handle_hazard_check,
                "PEOPLE_CHECK": self._handle_people_check,
                "OCR_READ": self._handle_ocr_read,
                "STATUS": self._handle_status,
            }.get(intent, self._handle_unknown)
            
            return handler(latest_payload, language, question)
        finally:
            self.is_busy = False

    def _classify_intent(self, question: str) -> str:
        """Keyword-based intent classification."""
        for intent, kws in self.keywords.items():
            for kw in kws:
                if kw in question:
                    return intent
        return "UNKNOWN"

    def _no_data_response(self, lang: str) -> str:
        if lang == "hi":
            return "अभी मेरे पास कोई दृश्य जानकारी नहीं है।"
        elif lang == "gu":
            return "અત્યારે મારી પાસે કોઈ દ્રશ્ય માહિતી નથી."
        else:
            return "I don't have any visual data right now."

    def _handle_vlm_query(self, payload: Optional[Dict[str, Any]], lang: str, question: str) -> str:
        """Escalate open-ended visual question to the on-demand Qwen3-VL-2B model."""
        # Immediate spoken acknowledgement
        ack_text = {
            "hi": "एक सेकंड, देख के बताता हूँ...",
            "gu": "એક મિનિટ, જોઈને કહું છું...",
            "en": "Let me take a look..."
        }.get(lang, "Let me take a look...")
        
        logger.info(f"Immediate VLM acknowledgement: '{ack_text}'")
        try:
            tts_manager.speak(ack_text, language=lang)
        except Exception as e:
            logger.warning(f"Could not speak acknowledgement: {e}")

        # Run VLM inference
        return self.vlm.answer(
            question=question,
            payload=payload,
            language=lang,
            max_new_tokens=getattr(config, "vlm_max_tokens", 100),
            temperature=getattr(config, "vlm_temperature", 0.2),
        )

    def _handle_unknown(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        if lang == "hi":
            return "क्षमा करें, मुझे समझ नहीं आया। आप आस-पास, खतरों, या लोगों के बारे में पूछ सकते हैं।"
        elif lang == "gu":
            return "માફ કરશો, મને સમજાયું નહીં। તમે આસપાસ, જોખમો અથવા લોકો વિશે પૂછી શકો છો."
        else:
            return "Sorry, I didn't understand. You can ask about your surroundings, hazards, people, or signs."

    def _handle_scene_summary(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        items = payload.get("scene_understanding", [])
        faces = payload.get("recognized_faces", [])
        texts = payload.get("environmental_text", [])
        
        if not items and not faces and not texts:
            if lang == "hi": return "रास्ता बिल्कुल साफ़ लग रहा है।"
            if lang == "gu": return "રસ્તો બિલકુલ સાફ લાગે છે."
            return "The path ahead appears clear."

        parts = []
        
        # Group by proximity
        close = [i for i in items if i.get("depth_score", 0) >= 0.5]
        far = [i for i in items if i.get("depth_score", 0) < 0.5]
        
        if close:
            prefix = "पास में" if lang == "hi" else "નજીકમાં" if lang == "gu" else "Nearby:"
            parts.append(f"{prefix} {self._describe_items(close, lang)}.")
        if far:
            prefix = "थोड़ी दूर" if lang == "hi" else "થોડે દૂર" if lang == "gu" else "Further away:"
            parts.append(f"{prefix} {self._describe_items(far, lang)}.")
            
        if faces:
            names = [f.get("name", "Unknown") for f in faces]
            if lang == "hi":
                parts.append(f"मुझे {', '.join(names)} दिख रहे हैं।")
            elif lang == "gu":
                parts.append(f"મને {', '.join(names)} દેખાય છે.")
            else:
                parts.append(f"I can see {', '.join(names)}.")
                
        if texts:
            text_strs = [t["text"] for t in texts if isinstance(t, dict) and "text" in t]
            if text_strs:
                if lang == "hi":
                    parts.append(f"बोर्ड पर लिखा है: {', '.join(text_strs)}।")
                elif lang == "gu":
                    parts.append(f"બોર્ડ પર લખ્યું છે: {', '.join(text_strs)}.")
                else:
                    parts.append(f"Sign reads: {', '.join(text_strs)}.")
        
        return " ".join(parts)

    def _handle_hazard_check(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        items = payload.get("scene_understanding", [])
        hazards = [i for i in items if i.get("priority") == "critical"]
        
        if not hazards:
            if lang == "hi": return "मुझे कोई खतरा नहीं दिख रहा।"
            if lang == "gu": return "મને કોઈ જોખમ દેખાતું નથી."
            return "I don't see any hazards."
            
        desc = self._describe_items(hazards, lang)
        if lang == "hi": return f"हाँ, {desc}।"
        if lang == "gu": return f"હા, {desc}."
        return f"Yes. {desc}."

    def _handle_people_check(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        items = payload.get("scene_understanding", [])
        faces = payload.get("recognized_faces", [])
        
        people_items = [i for i in items if i.get("class", "").lower() in ["person", "pedestrian", "child"]]
        
        if not people_items and not faces:
            if lang == "hi": return "आस-पास कोई नहीं है।"
            if lang == "gu": return "આસપાસ કોઈ નથી."
            return "I don't see anyone nearby."
            
        parts = []
        if faces:
            names = [f.get("name") for f in faces if f.get("name")]
            if names:
                if lang == "hi": parts.append(f"{', '.join(names)} पास में हैं।")
                elif lang == "gu": parts.append(f"{', '.join(names)} નજીકમાં છે.")
                else: parts.append(f"{', '.join(names)} is nearby.")
                
        # Count unknown people
        known_ids = {f.get("object_id") for f in faces if f.get("object_id") is not None}
        unknown_people = [p for p in people_items if p.get("object_id") not in known_ids]
        
        if unknown_people:
            count = len(unknown_people)
            if lang == "hi": parts.append(f"{count} अज्ञात व्यक्ति भी हैं।")
            elif lang == "gu": parts.append(f"{count} અજાણ્યા લોકો પણ છે.")
            else: parts.append(f"There {'is' if count == 1 else 'are'} {count} unknown person{'s' if count > 1 else ''}.")
            
        return " ".join(parts)

    def _handle_ocr_read(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        texts = payload.get("environmental_text", [])
        text_strs = [t["text"] for t in texts if isinstance(t, dict) and "text" in t]
        
        if not text_strs:
            if lang == "hi": return "मुझे कोई लिखावट नहीं दिख रही।"
            if lang == "gu": return "મને કોઈ લખાણ દેખાતું નથી."
            return "I don't see any text to read."
            
        if lang == "hi": return f"लिखा है: {', '.join(text_strs)}।"
        if lang == "gu": return f"લખ્યું છે: {', '.join(text_strs)}."
        return f"I can read: {', '.join(text_strs)}."

    def _handle_language_switch(self, question: str) -> str:
        if "hindi" in question:
            config.language = "hi"
            return "मैं अब हिंदी में बात करूँगी।"
        elif "gujarati" in question:
            config.language = "gu"
            return "હું હવે ગુજરાતીમાં વાત કરીશ."
        elif "english" in question:
            config.language = "en"
            return "I will speak in English now."
        return "I'm not sure which language you want."

    def _handle_status(self, payload: Dict[str, Any], lang: str, question: str) -> str:
        fps = payload.get("fps", 0)
        if lang == "hi": return f"सिस्टम {fps} फ्रेम प्रति सेकंड पर चल रहा है।"
        if lang == "gu": return f"સિસ્ટમ {fps} ફ્રેમ પ્રતિ સેકન્ડ પર ચાલી રહી છે."
        return f"Pipeline is running at {fps:.1f} frames per second."

    def _describe_items(self, items: list, lang: str) -> str:
        """Helper to combine multiple items into a natural sentence."""
        if not items:
            return ""
            
        descriptions = []
        for item in items:
            cls = item.get("class", "object").capitalize()
            pos = item.get("position", "ahead")
            depth = item.get("depth_score", 0.0)
            
            # Map depth to words
            if depth >= 0.8:
                dist = "very close"
                if lang == "hi": dist = "बिल्कुल पास"
                if lang == "gu": dist = "ખૂબ નજીક"
            elif depth >= 0.5:
                dist = "nearby"
                if lang == "hi": dist = "पास में"
                if lang == "gu": dist = "નજીકમાં"
            else:
                dist = "ahead"
                if lang == "hi": dist = "आगे"
                if lang == "gu": dist = "આગળ"
                
            if lang == "hi":
                descriptions.append(f"{pos} तरफ {dist} {cls} है")
            elif lang == "gu":
                descriptions.append(f"{pos} બાજુ {dist} {cls} છે")
            else:
                descriptions.append(f"a {cls} {dist} on your {pos}")
                
        if len(descriptions) == 1:
            return descriptions[0]
        elif len(descriptions) == 2:
            return f"{descriptions[0]} and {descriptions[1]}"
        else:
            return ", ".join(descriptions[:-1]) + f", and {descriptions[-1]}"
