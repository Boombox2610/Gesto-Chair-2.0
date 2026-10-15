import os
import time
import logging
import gc
from typing import Dict, Any, Optional

import torch
from PIL import Image

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None
    process_vision_info = None

from .nlp_config import config

logger = logging.getLogger("nlp_engine.vlm")


class VLMEngine:
    """
    On-Demand Vision-Language Model Engine (Qwen3-VL-2B).
    Loads model weights into VRAM only when queried, runs inference, and frees memory immediately.
    """

    DEFAULT_SNAPSHOT_PATH = r"C:\Users\HP\.cache\huggingface\hub\models--Qwen--Qwen3-VL-2B-Instruct\snapshots\89644892e4d85e24eaac8bacfd4f463576704203"

    def __init__(self, model_id: Optional[str] = None):
        # Prefer the local cached snapshot if available
        if model_id is None:
            if os.path.isdir(self.DEFAULT_SNAPSHOT_PATH):
                self.model_id = self.DEFAULT_SNAPSHOT_PATH
            else:
                self.model_id = getattr(config, "vlm_model", "Qwen/Qwen3-VL-2B-Instruct")
        else:
            self.model_id = model_id

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16
            if (self.device == "cuda" and torch.cuda.is_bf16_supported())
            else (torch.float16 if self.device == "cuda" else torch.float32)
        )

        self.flag_path = "frame_request.flag"
        self.frame_path = "latest_frame.jpg"
        self.is_busy = False

        logger.info(f"VLMEngine initialized with model: {self.model_id} on {self.device} ({self.dtype})")

    def _request_frame(self, timeout: float = 2.0) -> Optional[str]:
        """
        Request the CV pipeline to capture and save the latest frame via Pull Architecture.
        """
        try:
            if os.path.exists(self.flag_path):
                try:
                    os.remove(self.flag_path)
                except OSError:
                    pass

            # Signal CV pipeline
            with open(self.flag_path, "w", encoding="utf-8") as f:
                f.write("request")

            deadline = time.time() + timeout
            while time.time() < deadline:
                # When pipeline writes latest_frame.jpg, it removes frame_request.flag
                if not os.path.exists(self.flag_path) and os.path.exists(self.frame_path):
                    if os.path.getsize(self.frame_path) > 0:
                        return self.frame_path
                time.sleep(0.05)

            # Check if frame exists even if flag was lingering
            if os.path.exists(self.frame_path) and os.path.getsize(self.frame_path) > 0:
                logger.info("Frame request flag timed out, but found existing latest_frame.jpg.")
                return self.frame_path

        except Exception as e:
            logger.warning(f"Error requesting frame from CV pipeline: {e}")

        logger.warning("Frame acquisition timed out. Falling back to text-only mode.")
        return None

    def _build_context_string(self, payload: Optional[Dict[str, Any]]) -> str:
        """
        Convert structured CV payload into a brief text summary for model conditioning.
        """
        if not payload:
            return ""

        parts = []
        items = payload.get("scene_understanding", [])
        faces = payload.get("recognized_faces", [])
        texts = payload.get("environmental_text", [])

        for item in items[:6]:
            cls = item.get("class", "object")
            pos = item.get("position", "ahead")
            depth = item.get("depth_score", 0.0)
            dist = "very close" if depth >= 0.8 else "nearby" if depth >= 0.5 else "ahead"
            parts.append(f"{cls} ({dist}, {pos})")

        if faces:
            names = [f.get("name", "Unknown") for f in faces if f.get("name")]
            if names:
                parts.append(f"Faces seen: {', '.join(names)}")

        if texts:
            sign_texts = [t["text"] for t in texts if isinstance(t, dict) and "text" in t]
            if sign_texts:
                parts.append(f"Signs detected: {', '.join(sign_texts)}")

        return "; ".join(parts)

    def answer(
        self,
        question: str,
        payload: Optional[Dict[str, Any]] = None,
        language: str = "en",
        max_pixels: int = 480 * 640,
        max_new_tokens: int = 100,
        temperature: float = 0.2,
    ) -> str:
        """
        Run on-demand inference with Qwen3-VL-2B and return the spoken answer.
        Immediately frees VRAM upon completion.
        """
        if not AutoModelForImageTextToText or not AutoProcessor:
            return "Vision language model dependencies are not installed."

        self.is_busy = True
        frame_file = self._request_frame()
        context_str = self._build_context_string(payload)

        # Craft prompt tailored for natural audio delivery to a visually impaired user
        lang_instruction = {
            "hi": "उत्तर शुद्ध और स्वाभाविक हिंदी में दें।",
            "gu": "જવાબ શુદ્ધ અને સરળ ગુજરાતીમાં આપો.",
            "en": "Answer in clear English."
        }.get(language, "Answer in clear English.")

        system_instruction = (
            "You are an assistive navigation eye for a visually impaired person. "
            "Describe the scene and answer the user's question directly, accurately, and concisely in 1 to 2 spoken sentences. "
            f"{lang_instruction}"
        )

        user_prompt = f"Question: {question}"
        if context_str:
            user_prompt = f"Detected context: {context_str}\n{user_prompt}"

        image = None
        if frame_file and os.path.exists(frame_file):
            try:
                image = Image.open(frame_file).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open frame {frame_file}: {e}")
                image = None

        if image:
            messages = [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                            "max_pixels": max_pixels,
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": system_instruction,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],
                },
            ]

        model = None
        processor = None

        try:
            logger.info("Loading Qwen3-VL-2B into VRAM on demand...")
            t_load_start = time.time()
            processor = AutoProcessor.from_pretrained(self.model_id)
            model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                dtype=self.dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            load_duration = time.time() - t_load_start
            logger.info(f"Model loaded in {load_duration:.2f}s.")

            # Prepare inputs
            text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            logger.info("Running VLM inference...")
            t_inf_start = time.time()
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    repetition_penalty=1.1,
                )
            inf_duration = time.time() - t_inf_start

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            logger.info(f"VLM responded in {inf_duration:.2f}s: '{output_text}'")
            return output_text

        except Exception as e:
            logger.error(f"Error during VLM execution: {e}", exc_info=True)
            if language == "hi":
                return "क्षमा करें, मुझे इस दृश्य का विश्लेषण करने में समस्या आ रही है।"
            elif language == "gu":
                return "માફ કરશો, આ દ્રશ્ય સમજવામાં સમસ્યા આવી રહી છે."
            return "Sorry, I had trouble analyzing the scene."

        finally:
            self.is_busy = False
            # Enforce immediate memory release to return to steady-state VRAM
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-VL-2B unloaded. VRAM freed.")


# Global instance
vlm_engine = VLMEngine()
