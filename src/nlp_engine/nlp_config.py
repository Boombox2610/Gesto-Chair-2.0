import os
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class NLPConfig:
    # Language & Dual TTS
    language: str = "en"
    tts_primary_engine: str = "edge-tts"
    tts_emergency_engine: str = "pyttsx3"
    tts_rate: str = "+10%"
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    
    # Edge-TTS Voice Mapping
    voices: Dict[str, str] = field(default_factory=lambda: {
        "en": "en-IN-NeerjaExpressiveNeural",
        "hi": "hi-IN-SwaraNeural",
        "gu": "gu-IN-DhwaniNeural"
    })
    
    # Emergency pyttsx3 parameters
    pyttsx3_rate: int = 150
    
    # Announcements & Smart Cooldown
    announce_hazard_depth_thresh: float = 0.5
    announce_person_depth_thresh: float = 0.6
    announce_hazard_cooldown: int = 15
    announce_person_cooldown: int = 30
    announce_depth_increase_thresh: float = 0.1
    announce_ocr_cooldown: int = 30
    
    # STT
    stt_model: str = "small"
    stt_device: str = "cpu"       # CPU to preserve GPU for CV + VLM
    stt_compute_type: str = "int8"
    stt_silence_timeout: float = 2.0
    
    # Hotkey
    push_to_talk_key: str = "f5"
    
    # Reasoner & VLM
    vlm_enabled: bool = True
    vlm_model: str = "Qwen/Qwen3-VL-2B-Instruct"
    vlm_model_path: Optional[str] = None
    vlm_max_tokens: int = 100
    vlm_temperature: float = 0.2
    
    # Face Enrollment
    enrollment_enabled: bool = True
    enrollment_cooldown: int = 60
    enrollment_global_cooldown: int = 45
    enrollment_depth_thresh: float = 0.5
    face_db_path: str = "data/known_faces.pkl"
    faces_dir: str = "data/sample_faces"
    
    # Hazard chatter suppression
    hazard_class_cooldown: int = 8
    
    def load_from_yaml(self, yaml_path: Optional[str] = None):
        """Load configuration options from config.yaml if available."""
        if yaml_path is None:
            # Default lookup at project root
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            yaml_path = os.path.join(root_dir, "config.yaml")

        if not os.path.exists(yaml_path):
            return

        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            # Map nlp section
            nlp_data = data.get("nlp", {})
            for k, v in nlp_data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        except Exception as e:
            # Fail gracefully and retain defaults
            print(f"Warning: Failed to load NLP config from yaml: {e}")

# Global instance for easy import
config = NLPConfig()
config.load_from_yaml()
