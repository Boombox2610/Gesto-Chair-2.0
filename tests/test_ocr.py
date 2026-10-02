import pytest
import numpy as np
from cv_pipeline.ocr import OCRProcessor, OCRBackend, OCRResult

class MockOCRBackend(OCRBackend):
    def run(self, frame_rgb: np.ndarray):
        return [
            OCRResult(text="EXIT", confidence=0.9),
            OCRResult(text="BLURRY", confidence=0.4)
        ]

def test_ocr_processor_confidence_filter():
    backend = MockOCRBackend()
    processor = OCRProcessor(backend, conf_thresh=0.7)
    
    dummy_frame = np.zeros((10, 10, 3))
    
    # It should filter out the 0.4 confidence result
    results = processor.run(dummy_frame, force=True)
    
    assert len(results) == 1
    assert results[0].text == "EXIT"
    
def test_ocr_processor_trigger():
    from cv_pipeline.vocabulary import OCR_TRIGGER_CLASSES
    
    # Test OCR Trigger membership
    assert "stop sign" in OCR_TRIGGER_CLASSES
    assert "traffic light" in OCR_TRIGGER_CLASSES
    assert "wet floor sign" in OCR_TRIGGER_CLASSES
    assert "person" not in OCR_TRIGGER_CLASSES
