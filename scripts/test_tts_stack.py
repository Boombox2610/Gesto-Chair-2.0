import time
import argparse
import os

def test_pyttsx3(text):
    print("\n" + "="*50)
    print("--- Testing pyttsx3 (Offline Fallback) ---")
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        
        print(f"Speaking: '{text}'")
        start_time = time.time()
        engine.say(text)
        engine.runAndWait()
        print(f"Done in {time.time() - start_time:.2f} seconds.")
    except ImportError:
        print("pyttsx3 not installed. Run: pip install pyttsx3")
    except Exception as e:
        print(f"Failed: {e}")

def test_edge_tts(text, voice, rate="+10%", volume="+0%", pitch="+0Hz", output_file="edge_output.mp3"):
    print("\n" + "="*50)
    print(f"--- Testing edge-tts (Online, Zero-VRAM) | Voice: {voice} ---")
    print(f"Parameters: rate={rate}, volume={volume}, pitch={pitch}")
    try:
        import edge_tts
        import asyncio
        import pygame
        
        async def generate_and_play():
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )
            print(f"Generating audio for: '{text}'...")
            start_time = time.time()
            await communicate.save(output_file)
            print(f"Audio generated in {time.time() - start_time:.2f} seconds.")
            
            print("Playing audio...")
            pygame.mixer.init()
            pygame.mixer.music.load(output_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            pygame.mixer.quit()
            if os.path.exists(output_file):
                os.remove(output_file)
            
        asyncio.run(generate_and_play())
    except ImportError:
        print("Dependencies missing. Run: pip install edge-tts pygame")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed: {e}")

def test_indic_parler_suite(samples, description):
    print("\n" + "="*50)
    print("--- Testing ai4bharat/indic-parler-tts (Suite) ---")
    try:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        import soundfile as sf
        import sounddevice as sd
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print("Loading model from HuggingFace (Watch your VRAM!)...")
        load_start = time.time()
        model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        print(f"Model loaded in {time.time() - load_start:.2f} seconds.")
        
        if torch.cuda.is_available():
            vram_allocated = torch.cuda.memory_allocated() / (1024**2)
            vram_reserved = torch.cuda.memory_reserved() / (1024**2)
            print(f"VRAM Allocated: {vram_allocated:.2f} MB")
            print(f"VRAM Reserved: {vram_reserved:.2f} MB")

        description_input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)

        for text, lang in samples:
            print(f"\n[{lang}] Speaking: '{text}'")
            prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
            
            gen_start = time.time()
            generation = model.generate(
                input_ids=description_input_ids,
                prompt_input_ids=prompt_input_ids
            )
            print(f"Audio generated in {time.time() - gen_start:.2f} seconds.")
            
            audio_arr = generation.cpu().numpy().squeeze()
            sample_rate = model.config.sampling_rate
            
            out_filename = f"indic_test_{lang.lower()}.wav"
            sf.write(out_filename, audio_arr, sample_rate)
            print(f"Saved audio to {out_filename}")

            try:
                print("Playing audio...")
                sd.play(audio_arr, sample_rate)
                sd.wait()
            except Exception as audio_err:
                print(f"Audio playback warning (file saved successfully): {audio_err}")

        # Free memory
        del model
        del tokenizer
        if "generation" in locals():
            del generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\nAll samples processed. Model unloaded and VRAM freed.")
        
    except ImportError:
        print("Dependencies missing. Run: pip install transformers soundfile sounddevice git+https://github.com/huggingface/parler-tts.git")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["pyttsx3", "edge", "indic", "all"], default="all")
    parser.add_argument("--voice", type=str, default=None, help="Custom Edge-TTS voice name (e.g. en-IN-NeerjaExpressiveNeural)")
    parser.add_argument("--rate", type=str, default="+10%", help="Speed rate (e.g. +10%%, -5%%)")
    parser.add_argument("--volume", type=str, default="+0%", help="Volume (e.g. +0%%, -10%%)")
    parser.add_argument("--pitch", type=str, default="+0Hz", help="Pitch (e.g. +0Hz, +2Hz)")
    args = parser.parse_args()
    
    english_text = "Caution. Pothole ahead, Center-Left."
    hindi_text = "सावधान। गड्ढा आगे है, बीच-बाएं तरफ।"
    gujarati_text = "સાવધાન। ખાડો આગળ છે, વચ્ચે-ડાબી બાજુ।"
    
    description = "A female speaker, clear voice, moderate speaking rate, neutral emotion."
    
    if args.engine in ["pyttsx3", "all"]:
        test_pyttsx3(english_text)
        
    if args.engine in ["edge", "all"]:
        if args.voice:
            # Custom voice requested by user
            test_edge_tts(english_text, voice=args.voice, rate=args.rate, volume=args.volume, pitch=args.pitch)
        else:
            # Test default Indian female voices
            test_edge_tts(english_text, voice="en-IN-NeerjaExpressiveNeural", rate=args.rate, volume=args.volume, pitch=args.pitch)
            test_edge_tts(hindi_text, voice="hi-IN-SwaraNeural", rate=args.rate, volume=args.volume, pitch=args.pitch)
            test_edge_tts(gujarati_text, voice="gu-IN-DhwaniNeural", rate=args.rate, volume=args.volume, pitch=args.pitch)
            
            # Optional male voices comparison:
            print("\n💡 Tip: To test Male voices or others, run:")
            print("  python scripts/test_tts_stack.py --engine edge --voice en-IN-PrabhatNeural")
            print("  python scripts/test_tts_stack.py --engine edge --voice hi-IN-MadhurNeural")
            print("  python scripts/test_tts_stack.py --engine edge --voice gu-IN-NiranjanNeural")
        
    if args.engine in ["indic", "all"]:
        samples = [
            (english_text, "English"),
            (hindi_text, "Hindi"),
            (gujarati_text, "Gujarati")
        ]
        test_indic_parler_suite(samples, description)

