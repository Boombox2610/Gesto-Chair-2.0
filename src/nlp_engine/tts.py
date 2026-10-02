import os
import time
import asyncio
import traceback
import tempfile
import logging
import threading

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import pygame
except ImportError:
    pygame = None

from .nlp_config import config

logger = logging.getLogger("nlp_engine.tts")

class TTSManager:
    """
    Intelligently routes speech to Edge-TTS or pyttsx3 based on priority and network health.
    """
    def __init__(self):
        self.pyttsx3_engine = None
        self._speaking_lock = threading.Lock()
        self.is_speaking = False
        
        self._init_pyttsx3()
        
        if pygame:
            pygame.mixer.init()
        else:
            logger.warning("pygame not installed. edge-tts audio playback may fail.")

    def _init_pyttsx3(self):
        if not pyttsx3:
            logger.error("pyttsx3 is not installed. Offline fallback disabled.")
            return
            
        try:
            self.pyttsx3_engine = pyttsx3.init()
            self.pyttsx3_engine.setProperty('rate', config.pyttsx3_rate)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3: {e}")
            self.pyttsx3_engine = None

    def speak(self, text: str, language: str = None, is_critical: bool = False):
        if language is None:
            language = config.language
            
        with self._speaking_lock:
            self.is_speaking = True
            try:
                if is_critical and self.pyttsx3_engine:
                    logger.info(f"[TTS - pyttsx3 Critical] Speaking: '{text}'")
                    self._speak_pyttsx3(text)
                    return

                if not edge_tts:
                    logger.warning("edge-tts not installed. Falling back to pyttsx3.")
                    self._speak_pyttsx3(text)
                    return
                    
                try:
                    voice = config.voices.get(language, config.voices.get("en"))
                    logger.info(f"[TTS - edge-tts] Speaking: '{text}' in {language}")
                    
                    # Create a temporary file for the generated audio
                    fd, temp_path = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    
                    asyncio.run(self._speak_edge(text, voice, temp_path))
                    
                except Exception as e:
                    logger.warning(f"Edge-TTS failed ({e}). Falling back to pyttsx3.")
                    self._speak_pyttsx3(text)
            finally:
                self.is_speaking = False

    def _speak_pyttsx3(self, text: str):
        if not self.pyttsx3_engine:
            logger.error("pyttsx3 is not initialized. Cannot speak.")
            return
            
        try:
            self.pyttsx3_engine.say(text)
            self.pyttsx3_engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 playback failed: {e}")

    async def _speak_edge(self, text: str, voice: str, output_file: str):
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=config.tts_rate,
            volume=config.tts_volume,
            pitch=config.tts_pitch
        )
        
        await communicate.save(output_file)
        
        if not pygame:
            logger.error("Cannot play audio: pygame is missing")
            return
            
        try:
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            logger.error(f"pygame playback failed: {e}")
        finally:
            pygame.mixer.music.unload()
            if os.path.exists(output_file):
                try:
                    os.remove(output_file)
                except:
                    pass

# Global instance
tts_manager = TTSManager()
