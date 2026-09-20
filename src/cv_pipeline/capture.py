import time
import cv2
from collections import deque
from typing import Tuple, Optional, Union
import numpy as np

class FPSMonitor:
    """Monitors FPS using a rolling window."""
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
    
    def tick(self):
        """Record a frame completion time."""
        self.frame_times.append(time.time())
        
    @property
    def current_fps(self) -> float:
        """Calculate FPS over the rolling window."""
        if len(self.frame_times) < 2:
            return 0.0
        elapsed = self.frame_times[-1] - self.frame_times[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.frame_times) - 1) / elapsed
        
    def is_degraded(self, target_fps: float) -> bool:
        """Check if current FPS is below target."""
        if len(self.frame_times) < self.window_size:
            return False # Need a full window to judge
        return self.current_fps < target_fps

class WebcamCapture:
    """Wraps cv2.VideoCapture with FPS monitoring, video file support, and error handling."""
    def __init__(self, camera_index: Union[int, str], width: int, height: int, loop_video: bool = True):
        # Convert numeric string to int
        if isinstance(camera_index, str) and camera_index.isdigit():
            self.camera_index = int(camera_index)
        else:
            self.camera_index = camera_index
            
        self.width = width
        self.height = height
        self.loop_video = loop_video
        self.is_file = isinstance(self.camera_index, str)
        self.cap = None
        self.frame_id = -1 # Starts at -1 so first frame is 0
        self.fps_monitor = FPSMonitor()
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera or video file at: {self.camera_index}")
        
        # Only set capture resolution for live camera feeds
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def read(self) -> Tuple[bool, Optional[np.ndarray], int, float]:
        """
        Read a frame from webcam or video file.
        Returns: (success, frame, frame_id, timestamp)
        """
        if not self.cap:
            return False, None, self.frame_id, time.time()
            
        success, frame = self.cap.read()
        timestamp = time.time()
        
        if not success and self.is_file and self.loop_video:
            # Auto-rewind video on EOF
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.cap.read()
            
        if success:
            self.frame_id += 1
            
        return success, frame, self.frame_id, timestamp
