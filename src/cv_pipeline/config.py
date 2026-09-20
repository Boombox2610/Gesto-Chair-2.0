import os
# Prevent OpenBLAS/OMP from oversubscribing CPU in background threads
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from dataclasses import dataclass, field
from typing import Literal, Union, Optional

@dataclass
class PipelineConfig:
    # Camera / Video settings
    camera_index: Union[int, str] = 0
    frame_width: int = 640
    frame_height: int = 480
    
    # Compute device
    device: str = "cuda"  # Auto-falls back to "cpu" if unavailable
    
    # YOLO object detection
    yoloe_model: str = "yoloe-11s-seg.pt"  # New open-vocabulary model
    yolo_model: str = "yolov8n.pt"         # Legacy model (fallback)
    yolo_conf_thresh: float = 0.4
    tracker: str = "bytetrack.yaml"
    scene_mode: Literal["outdoor", "indoor"] = "outdoor"
    
    # MiDaS depth estimation
    midas_model: str = "MiDaS_small"
    depth_fallback_interval: int = 6
    depth_hazard_area_thresh: float = 0.04
    depth_agent_area_thresh: float = 0.08
    
    # OCR settings
    ocr_every_n_frames: int = 10
    ocr_conf_thresh: float = 0.7
    
    # Face recognition settings
    face_every_n_frames: int = 5
    face_tolerance: float = 0.5
    face_db_path: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "known_faces.pkl")
    
    # Performance & Status
    fps_target: float = 10.0
    
    # Spatial layout
    spatial_zones: int = 5
    
    # Output
    payload_sink: Literal["stdout", "file", "both"] = "both"
    payload_file: str = "payloads.jsonl"
    
    # UI
    show_cv_window: bool = True
    show_dashboard: bool = True
    
    # Offline mode
    offline: bool = False

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

            # Map pipeline section
            p_data = data.get("pipeline", {})
            for k, v in p_data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

            # Map models section
            m_data = data.get("models", {})
            for k, v in m_data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

        except Exception as e:
            # Fail gracefully and retain defaults
            pass

# Global instance for easy import
config = PipelineConfig()
config.load_from_yaml()

