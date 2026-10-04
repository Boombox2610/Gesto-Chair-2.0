import numpy as np
import logging

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

from .nlp_config import config

logger = logging.getLogger("nlp_engine.stt")

class STTEngine:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        self.model = None
        self.sample_rate = 16000
        self.is_recording = False
        self._audio_buffer = []
        self._stream = None

        if WhisperModel:
            logger.info(f"Loading Whisper model '{model_size}' on {device} ({compute_type})...")
            try:
                self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
                logger.info("Whisper model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
        else:
            logger.error("faster_whisper is not installed. STT disabled.")

    def start_recording(self):
        """Called when hotkey is pressed."""
        if not sd:
            logger.error("sounddevice is not installed. Cannot record.")
            return

        self._audio_buffer = []
        self.is_recording = True
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32', 
                callback=self._audio_callback
            )
            self._stream.start()
            logger.info("Started recording...")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False

    def stop_recording_and_transcribe(self) -> str:
        """Called when hotkey is released. Returns transcribed text."""
        if not self.is_recording:
            return ""

        self.is_recording = False
        
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
                
        logger.info("Stopped recording. Transcribing...")

        if not self._audio_buffer:
            logger.warning("Audio buffer is empty.")
            return ""

        if not self.model:
            logger.warning("Whisper model is not loaded. Cannot transcribe.")
            return ""

        try:
            audio = np.concatenate(self._audio_buffer)
            # Flatten to 1D array
            audio = audio.flatten()
            
            # Use appropriate language hint if it's not english
            language = config.language if config.language in ["hi", "gu"] else "en"
            
            segments, info = self.model.transcribe(audio, beam_size=5, language=language)
            text = " ".join(seg.text for seg in segments).strip()
            logger.info(f"Transcribed text: {text}")
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Audio callback status: {status}")
        if self.is_recording:
            self._audio_buffer.append(indata.copy())

# Global instance
stt_engine = STTEngine(
    model_size=config.stt_model, 
    device=config.stt_device, 
    compute_type=config.stt_compute_type
)
