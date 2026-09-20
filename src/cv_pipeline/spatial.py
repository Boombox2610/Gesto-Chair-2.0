from typing import Tuple

class SpatialMapper:
    """Maps bounding boxes to discrete spatial zones."""
    
    def __init__(self, frame_width: int, n_zones: int = 5):
        self.frame_width = frame_width
        self.n_zones = n_zones
        self.zone_width = frame_width / n_zones
        
        # Define zone labels based on number of zones
        if n_zones == 5:
            self.labels = ["Left", "Center-Left", "Center", "Center-Right", "Right"]
        elif n_zones == 3:
            self.labels = ["Left", "Center", "Right"]
        else:
            self.labels = [f"Zone_{i}" for i in range(n_zones)]
            
    def get_zone(self, bbox_xyxy: Tuple[float, float, float, float]) -> str:
        """
        Determine the spatial zone of a bounding box.
        Uses the center x-coordinate of the box.
        """
        x1, _, x2, _ = bbox_xyxy
        center_x = (x1 + x2) / 2.0
        
        # Calculate zone index (0 to n_zones - 1)
        zone_index = int(center_x / self.zone_width)
        
        # Clamp just in case center_x is exactly at the right edge
        zone_index = max(0, min(self.n_zones - 1, zone_index))
        
        return self.labels[zone_index]
