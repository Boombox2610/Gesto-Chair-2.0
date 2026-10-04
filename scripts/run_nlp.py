import os
import sys

# Ensure correct working directory and add project root to python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if os.getcwd() != project_root:
    os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import logging
import threading
import time

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

from src.nlp_engine.ingester import PayloadIngester
from src.nlp_engine.nlp_config import config
from src.nlp_engine.stt import stt_engine
from src.nlp_engine.reasoner import SceneReasoner
from src.nlp_engine.tts import tts_manager
from src.nlp_engine.enrollment import enrollment_controller


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_nlp")

def main():
    parser = argparse.ArgumentParser(description="Run the NLP Engine")
    parser.add_argument("--mode", type=str, choices=["pipe", "file-tail"], default="pipe", 
                        help="Ingestion mode. 'pipe' reads from stdin, 'file-tail' reads payloads.jsonl")
    parser.add_argument("--file", type=str, default="payloads.jsonl", 
                        help="Path to the payload JSONL file (for file-tail mode)")
    parser.add_argument("--lang", type=str, choices=["en", "hi", "gu"], default=None,
                        help="Override default language")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Disable VLM fallback, use template matching only")
    parser.add_argument("--no-enrollment", action="store_true",
                        help="Disable interactive voice face enrollment")
                        
    args = parser.parse_args()
    
    if args.lang:
        config.language = args.lang
        logger.info(f"Language overridden to: {args.lang}")
        
    if args.no_vlm:
        config.vlm_enabled = False
        logger.info("VLM fallback disabled via CLI flag.")
        
    if args.no_enrollment:
        config.enrollment_enabled = False
        logger.info("Face enrollment disabled via CLI flag.")
        
    ingester = PayloadIngester(mode=args.mode, file_path=args.file)
    reasoner = SceneReasoner(ingester)
    
    # Hotkey thread function
    def run_hotkey_listener():
        if not keyboard:
            logger.warning("pynput not installed. Voice hotkey disabled.")
            return
            
        def on_press(key):
            try:
                # Check for F5
                if key == keyboard.Key.f5:
                    if not stt_engine.is_recording:
                        logger.info("Hotkey F5 pressed. Recording...")
                        stt_engine.start_recording()
                        # Play a simple beep
                        print("\a", end="", flush=True) 
            except AttributeError:
                pass
                
        def on_release(key):
            if key == keyboard.Key.f5:
                if stt_engine.is_recording:
                    logger.info("Hotkey F5 released. Processing...")
                    text = stt_engine.stop_recording_and_transcribe()
                    
                    if text:
                        # Priority: if interactive face enrollment is active, process there
                        if enrollment_controller.is_active():
                            logger.info(f"Routing voice to enrollment controller: '{text}'")
                            handled = enrollment_controller.handle_voice_input(text, config.language)
                            if handled:
                                return

                        answer = reasoner.answer(text)
                        if answer:
                            logger.info(f"Reasoner answer: {answer}")
                            tts_manager.speak(answer)
                    else:
                        logger.warning("No speech transcribed.")
                        
        logger.info("Starting hotkey listener on background thread (Press F5 to talk)...")
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
            
    # Start the background listener
    hotkey_thread = threading.Thread(target=run_hotkey_listener, daemon=True)
    hotkey_thread.start()
    
    try:
        ingester.run()
    except KeyboardInterrupt:
        logger.info("NLP Engine stopped by user.")
    except Exception as e:
        logger.error(f"NLP Engine error: {e}")

if __name__ == "__main__":
    main()
