from abc import ABC, abstractmethod
from typing import List
from collections import namedtuple
import numpy as np
import easyocr

OCRResult = namedtuple("OCRResult", ["text", "confidence"])

class OCRBackend(ABC):
    """Abstract base class for OCR engines."""
    
    @abstractmethod
    def run(self, frame_rgb: np.ndarray) -> List[OCRResult]:
        pass

class EasyOCRBackend(OCRBackend):
    """EasyOCR implementation."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = True):
        # Setting gpu=True will use CUDA if available
        self.reader = easyocr.Reader(languages, gpu=gpu)
        
    def run(self, frame_rgb: np.ndarray) -> List[OCRResult]:
        # readtext returns [(bbox, text, prob), ...]
        raw_results = self.reader.readtext(frame_rgb)
        
        results = []
        for (_, text, prob) in raw_results:
            # Uppercase for consistency in environmental text
            results.append(OCRResult(text=text.upper(), confidence=float(prob)))
            
        return results

class PaddleOCRBackend(OCRBackend):
    """Stub for PaddleOCR backend (not installed by default)."""
    def run(self, frame_rgb: np.ndarray) -> List[OCRResult]:
        raise NotImplementedError("PaddleOCR backend not implemented")

class TesseractBackend(OCRBackend):
    """Stub for Tesseract backend (not installed by default)."""
    def run(self, frame_rgb: np.ndarray) -> List[OCRResult]:
        raise NotImplementedError("Tesseract backend not implemented")

class OCRProcessor:
    """Manages OCR execution and confidence filtering."""
    
    def __init__(self, backend: OCRBackend, conf_thresh: float = 0.7):
        self.backend = backend
        self.conf_thresh = conf_thresh
        
    def run(self, frame_rgb: np.ndarray, detected_classes: List[str] = None, force: bool = False) -> List[OCRResult]:
        """Run OCR and filter by confidence."""
        
        # We assume should_trigger was already checked before calling this in ThreadPoolExecutor
        # but just in case:
        if not force and detected_classes is not None:
            # We don't have frame_count here easily if using this check, 
            # so the orchestration loop should handle the should_trigger logic.
            pass
            
        raw_results = self.backend.run(frame_rgb)
        
        # Filter by confidence
        filtered = [res for res in raw_results if res.confidence >= self.conf_thresh]
        
        return filtered
