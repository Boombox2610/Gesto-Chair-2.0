import cv2
import numpy as np
import torch
from typing import Tuple

class DepthEstimator:
    """Wrapper for MiDaS monocular depth estimation."""
    
    def __init__(self, model_name: str = "MiDaS_small", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # We use torch.hub to load MiDaS
        # (It checks local cache first if previously downloaded)
        hub_kwargs = {"trust_repo": True}
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, skip_validation=True, **hub_kwargs)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True, **hub_kwargs)
        except TypeError:
            self.model = torch.hub.load("intel-isl/MiDaS", model_name, **hub_kwargs)
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", **hub_kwargs)

        self.model.to(self.device)
        self.model.eval()

        if model_name == "MiDaS_small":
            self.transform = midas_transforms.small_transform
        else:
            self.transform = midas_transforms.default_transform
            
        # Cache for interleaving
        self.cached_depth_map = None
        
    def infer(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Run inference and return normalized depth map (0-255 uint8).
        Updates the cache.
        """
        input_batch = self.transform(frame_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize back to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255 uint8 for easy caching and visualization
        # Note: MiDaS values are inverse depth (higher = closer)
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        self.cached_depth_map = depth_norm
        return depth_norm
        
    def get_roi_depth(self, bbox_xyxy: Tuple[float, float, float, float]) -> float:
        """
        Get the median depth score (0.0 to 1.0) for a bounding box ROI.
        Uses the cached depth map. If no cache, returns 0.0.
        """
        if self.cached_depth_map is None:
            return 0.0
            
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        
        # Ensure bounds are within image
        h, w = self.cached_depth_map.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if y2 <= y1 or x2 <= x1:
            return 0.0
            
        roi = self.cached_depth_map[y1:y2, x1:x2]
        median_val = float(np.median(roi))
        
        # Normalize 0-255 back to 0.0-1.0
        return median_val / 255.0
