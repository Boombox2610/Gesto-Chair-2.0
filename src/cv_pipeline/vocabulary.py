import os
from typing import List, Set, Dict

# --- DEFAULT FALLBACK LABELS ---
_DEFAULT_OUTDOOR = [
    "pothole", "road crater", "asphalt bump", "plastic rumble strip", "unpainted speed bump",
    "closed manhole", "open manhole", "improperly seated manhole",
    "longitudinal crack", "transverse crack", "alligator cracking", "uneven paving",
    "water puddle", "oil spill", "road debris", "garbage pile", "fallen tree branch",
    "zebra crossing", "crosswalk", "solid lane line", "dashed lane line", "stop line",
    "road directional arrow", "stop sign", "yield sign", "speed limit sign",
    "pedestrian crossing sign", "traffic light", "red traffic light", "green traffic light",
    "yellow traffic light", "pedestrian walk signal", "pedestrian don't walk signal",
    "construction barricade", "traffic cone", "bollard", "guardrail", "road divider",
    "median strip", "curb edge", "tactile paving", "braille blocks", "staircase edge",
    "curb ramp", "sidewalk", "pedestrian", "person", "wheelchair", "baby stroller",
    "cyclist", "bicycle", "motorcycle", "scooter", "car", "suv", "auto rickshaw",
    "bus", "truck", "stray dog", "street cattle"
]

_DEFAULT_INDOOR = [
    "staircase going down", "staircase going up", "single step down", "single step up",
    "closed glass door", "open door", "narrow doorway", "hallway corner",
    "sliding glass partition", "indoor wall", "pillar", "column",
    "wet floor", "wet floor sign", "loose rug", "carpet edge",
    "exposed electrical cable", "floor wire", "dropped object", "scattered toys",
    "dropped clothes", "liquid spill on floor",
    "office chair", "dining chair", "armchair", "wheelchair", "desk", "dining table",
    "coffee table", "sofa", "bed", "cabinet", "bookshelf", "trash bin", "indoor plant",
    "elevator door", "elevator call button", "emergency exit sign", "restroom sign",
    "room number plaque", "directional indoor sign",
    "walking person", "sitting person", "standing person", "person", "child",
    "pet dog", "pet cat", "cleaning cart", "luggage trolley"
]

_DEFAULT_HAZARDS = {
    "pothole", "road crater", "asphalt bump", "plastic rumble strip", "unpainted speed bump",
    "open manhole", "improperly seated manhole", "water puddle", "oil spill", "fallen tree branch",
    "curb edge", "staircase edge", "staircase going down", "single step down", "closed glass door",
    "wet floor", "exposed electrical cable", "floor wire", "liquid spill on floor"
}

_DEFAULT_AGENTS = {
    "pedestrian", "person", "walking person", "sitting person", "standing person", "child",
    "cyclist", "bicycle", "motorcycle", "scooter", "car", "suv", "auto rickshaw",
    "bus", "truck", "stray dog", "street cattle", "pet dog", "pet cat",
    "cleaning cart", "luggage trolley", "wheelchair", "baby stroller"
}

_DEFAULT_OCR_TRIGGERS = {
    "stop sign", "yield sign", "speed limit sign", "pedestrian crossing sign",
    "traffic light", "red traffic light", "green traffic light", "yellow traffic light",
    "pedestrian walk signal", "pedestrian don't walk signal", "wet floor sign",
    "emergency exit sign", "restroom sign", "room number plaque", "directional indoor sign"
}

# Module variables
OUTDOOR_LABELS: List[str] = list(_DEFAULT_OUTDOOR)
INDOOR_LABELS: List[str] = list(_DEFAULT_INDOOR)
HAZARD_CLASSES: Set[str] = set(_DEFAULT_HAZARDS)
AGENT_CLASSES: Set[str] = set(_DEFAULT_AGENTS)
OCR_TRIGGER_CLASSES: Set[str] = set(_DEFAULT_OCR_TRIGGERS)
PRIORITY_CLASSES: Dict[str, str] = {}

def load_vocabulary_from_yaml(yaml_path: str = None):
    """Dynamically load and configure vocabulary sets from config.yaml."""
    global OUTDOOR_LABELS, INDOOR_LABELS, HAZARD_CLASSES, AGENT_CLASSES, OCR_TRIGGER_CLASSES, PRIORITY_CLASSES
    
    if yaml_path is None:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        yaml_path = os.path.join(root_dir, "config.yaml")

    if os.path.exists(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            vocab_data = data.get("vocabulary", {})
            if "outdoor" in vocab_data and isinstance(vocab_data["outdoor"], list):
                OUTDOOR_LABELS = [str(x).strip().lower() for x in vocab_data["outdoor"]]
            if "indoor" in vocab_data and isinstance(vocab_data["indoor"], list):
                INDOOR_LABELS = [str(x).strip().lower() for x in vocab_data["indoor"]]
            if "hazards" in vocab_data and isinstance(vocab_data["hazards"], list):
                HAZARD_CLASSES = {str(x).strip().lower() for x in vocab_data["hazards"]}
            if "agents" in vocab_data and isinstance(vocab_data["agents"], list):
                AGENT_CLASSES = {str(x).strip().lower() for x in vocab_data["agents"]}
            if "ocr_triggers" in vocab_data and isinstance(vocab_data["ocr_triggers"], list):
                OCR_TRIGGER_CLASSES = {str(x).strip().lower() for x in vocab_data["ocr_triggers"]}
        except Exception:
            pass

    # Build priority map
    PRIORITY_CLASSES.clear()
    for label in OUTDOOR_LABELS + INDOOR_LABELS:
        PRIORITY_CLASSES[label] = "contextual"
        
    for label in AGENT_CLASSES:
        PRIORITY_CLASSES[label] = "standard"
        
    for label in HAZARD_CLASSES:
        PRIORITY_CLASSES[label] = "critical"

# Initialize priority mappings
load_vocabulary_from_yaml()

def get_vocabulary(mode: str) -> List[str]:
    """Return the active vocabulary list based on the scene mode."""
    if mode.lower() == "indoor":
        return INDOOR_LABELS
    return OUTDOOR_LABELS

