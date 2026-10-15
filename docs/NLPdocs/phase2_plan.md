# Phase 2 — Voice Input & Scene Q&A: Implementation Plan

> **Status:** ✅ COMPLETE — Phase 2A (Template Q&A) & Phase 2B (Qwen3-VL-2B Integration) fully implemented and tested  
> **Created:** 2026-10-10  
> **Parent doc:** [nlp_plan.md](./nlp_plan.md)  
> **Depends on:** Phase 1 (Dual TTS & Core Announcements) — ✅ COMPLETE  
> **Scope:** Voice Q&A (template-based) + VLM fallback (Qwen3-VL-2B) — ✅ COMPLETE

---

## 1. Phase 2 Objective

Give the user the ability to **ask questions about their surroundings using their voice** and receive **spoken answers**. The system operates in two tiers:

- **Tier 1 (Template Reasoner):** Structured JSON-based responses with zero LLM latency — handles the vast majority of questions ("What's around me?", "Any hazards?", etc.).
- **Tier 2 (VLM Fallback):** When the question cannot be answered from JSON alone ("What color is that?", "Is it safe to cross?"), Qwen3-VL-2B is invoked on demand with the latest camera frame.

By the end of Phase 2, the user can press `F5`, ask any question, and receive either an instant template answer or a VLM-reasoned answer within ~3-5 seconds.

---

## 2. What Phase 2 Delivers

| Capability | Example |
|---|---|
| Push-to-talk voice input | User holds `Space` → speaks → releases |
| Scene summary | *"There are 3 objects around you. A person is on your left, very close. A cabinet is on your right. An armchair is ahead."* |
| Hazard query | *"Any hazards?"* → *"Yes. Wet floor ahead, Center-Right."* |
| People query | *"Is anyone nearby?"* → *"Yes. An unknown person is on your left, very close."* |
| OCR query | *"What does the sign say?"* → *"The sign reads: 603-67."* |
| Language switch | *"Switch to Hindi"* → All subsequent announcements and answers switch to Hindi |
| Status query | *"What's the FPS?"* → *"Pipeline is running at 29 frames per second."* |

---

## 3. New Modules

### 3.1 STT Engine — `stt.py`

**Responsibility:** Record microphone audio and transcribe it to text.

**Model:** `faster-whisper` with `small` model, INT8 quantized.

| Property | Value |
|---|---|
| **Device** | **CPU** (keeps GPU free for CV + VLM headroom) |
| VRAM | **0 MB** (CPU inference) |
| RAM | ~600 MB system RAM |
| Languages | Auto-detect (English, Hindi, Gujarati native) |
| Latency | ~2–3s for short utterances on CPU (acceptable for Q&A) |
| Dependency | `faster-whisper`, `sounddevice`, `numpy` |
| Loading | **At startup** — model stays in RAM permanently |

#### Recording Strategy

```
User holds Space bar
  → Microphone begins recording (16kHz, mono, float32)
  → User releases Space bar
  → Recording stops
  → Audio buffer sent to faster-whisper
  → Transcribed text returned
```

**Why push-to-talk, not wake word?**
- Zero false positives — critical for an accessibility system
- No always-on microphone (privacy)
- No additional wake-word model (saves VRAM)
- Simple, reliable, deterministic

#### Implementation Skeleton

```python
class STTEngine:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.sample_rate = 16000
        self.is_recording = False
        self._audio_buffer = []

    def start_recording(self):
        """Called when hotkey is pressed."""
        self._audio_buffer = []
        self.is_recording = True
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1, 
                                       dtype='float32', callback=self._audio_callback)
        self._stream.start()

    def stop_recording_and_transcribe(self) -> str:
        """Called when hotkey is released. Returns transcribed text."""
        self.is_recording = False
        self._stream.stop()
        self._stream.close()
        
        audio = np.concatenate(self._audio_buffer)
        segments, info = self.model.transcribe(audio, beam_size=5)
        text = " ".join(seg.text for seg in segments).strip()
        return text
    
    def _audio_callback(self, indata, frames, time_info, status):
        if self.is_recording:
            self._audio_buffer.append(indata.copy())
```

#### Resolved Decisions

- **Device:** CPU. Keeps 0 MB of VRAM occupied, preserving headroom for the VLM (Qwen3-VL-2B needs ~2.5 GB on demand).
- **Loading:** At startup. The `small` model loads in ~3s and stays in RAM. First query has no cold-start delay.
- **Compute type:** `int8` for CPU (fastest CPU inference path in faster-whisper).

---

### 3.2 Scene Reasoner — `reasoner.py`

**Responsibility:** Take a user's transcribed question and the current CV payload, then construct a structured spoken answer *without* any LLM.

#### Intent Classification (Keyword Matching)

We do NOT need a language model for intent detection. Simple keyword matching covers the real-world question space for a visually impaired user:

| Intent | Trigger Keywords (EN/HI/GU) | Handler | Tier |
|---|---|---|---|
| `SCENE_SUMMARY` | "what's around", "describe", "scene", "surroundings", "kya hai", "aas paas" | `_handle_scene_summary()` | Template |
| `HAZARD_CHECK` | "hazard", "danger", "safe", "khatara", "khatraa" | `_handle_hazard_check()` | Template |
| `PEOPLE_CHECK` | "anyone", "person", "people", "who", "kaun", "koi hai" | `_handle_people_check()` | Template |
| `OCR_READ` | "sign", "text", "read", "board", "likha", "padho" | `_handle_ocr_read()` | Template |
| `LANG_SWITCH` | "switch to hindi", "hindi mein", "gujarati", "english" | `_handle_language_switch()` | Template |
| `STATUS` | "fps", "status", "speed", "performance" | `_handle_status()` | Template |
| `VLM_QUERY` | *(any question not matched above)* | `_handle_vlm_query()` | **VLM** |

> **Escalation rule:** If keyword matching scores `UNKNOWN`, the reasoner automatically escalates to the VLM tier instead of returning an error. The VLM handles open-ended visual questions the JSON cannot answer.

#### Response Construction

Responses are built from **templates** using the last N payloads from the ingester's sliding window. No generative model, no latency uncertainty.

```python
class SceneReasoner:
    def __init__(self, ingester: PayloadIngester):
        self.ingester = ingester  # Access to .history (sliding window)

    def answer(self, question: str, language: str = "en") -> str:
        intent = self._classify_intent(question)
        latest = self.ingester.history[-1] if self.ingester.history else None
        
        if latest is None:
            return self._no_data_response(language)
            
        handler = {
            "SCENE_SUMMARY": self._handle_scene_summary,
            "HAZARD_CHECK": self._handle_hazard_check,
            "PEOPLE_CHECK": self._handle_people_check,
            "OCR_READ": self._handle_ocr_read,
            "LANG_SWITCH": self._handle_language_switch,
            "STATUS": self._handle_status,
        }.get(intent, self._handle_unknown)
        
        return handler(latest, language, question)
```

#### Example: `_handle_scene_summary()`

```python
def _handle_scene_summary(self, payload, lang, question):
    items = payload.get("scene_understanding", [])
    faces = payload.get("recognized_faces", [])
    texts = payload.get("environmental_text", [])
    
    if not items and not faces and not texts:
        return "The path ahead appears clear." if lang == "en" else ...

    parts = []
    
    # Group by proximity
    close = [i for i in items if i.get("depth_score", 0) >= 0.5]
    far = [i for i in items if i.get("depth_score", 0) < 0.5]
    
    if close:
        parts.append(f"Nearby: {self._describe_items(close, lang)}")
    if far:
        parts.append(f"Further away: {self._describe_items(far, lang)}")
    if faces:
        names = [f.get("name", "Unknown") for f in faces]
        parts.append(f"I can see {', '.join(names)}.")
    if texts:
        text_strs = [t["text"] for t in texts if isinstance(t, dict)]
        if text_strs:
            parts.append(f"Sign reads: {', '.join(text_strs)}.")
    
    return " ".join(parts)
```

#### Depth-to-Distance Mapping

For natural language, we translate raw depth scores to human-understandable proximity labels:

| Depth Score | Label (EN) | Label (HI) |
|---|---|---|
| >= 0.8 | "very close" | "bahut paas" |
| 0.5-0.8 | "nearby" | "paas mein" |
| 0.3-0.5 | "moderate distance" | "thodi door" |
| < 0.3 | "far away" | "door" |

---

### 3.3 Hotkey Listener — Integration into `run_nlp.py`

**Responsibility:** Listen for the push-to-talk key in a background thread while the ingester loop continues processing payloads.

#### Threading Architecture

```
Main Thread                    Background Thread
    |                               |
    |  PayloadIngester.run()        |  keyboard listener (pynput)
    |  (blocking loop reading       |  
    |   JSONL and calling           |  Space pressed -> stt.start_recording()
    |   announcer.process_payload)  |  Space released -> text = stt.stop_and_transcribe()
    |                               |                 -> answer = reasoner.answer(text)
    |                               |                 -> tts_manager.speak(answer)
    |                               |
```

**Key consideration:** TTS playback is synchronous (blocks until utterance finishes). The announcer's proactive alerts must be **suppressed** while the reasoner is speaking an answer, so they don't overlap. We need a simple lock or "speaking" flag.

#### Library Choice: `pynput`

- Works on Windows without admin
- Non-blocking keyboard listener
- Runs in its own daemon thread
- Lightweight, no additional dependencies beyond pip

---

## 4. Modifications to Existing Modules

### 4.1 `ingester.py` — Expose History

Already stores `self.history` (sliding window of 30 payloads). No changes needed, but the `SceneReasoner` needs a reference to it at construction time.

### 4.2 `tts.py` — Add Speaking Lock

```python
class TTSManager:
    def __init__(self):
        ...
        self._speaking_lock = threading.Lock()
        self.is_speaking = False  # Public flag for announcer suppression
    
    def speak(self, text, ...):
        with self._speaking_lock:
            self.is_speaking = True
            try:
                # existing speak logic
            finally:
                self.is_speaking = False
```

### 4.3 `announcer.py` — Respect Speaking Lock

When the user is interacting (asking a question / listening to an answer), proactive announcements should be queued or suppressed:

```python
def process_payload(self, payload):
    if tts_manager.is_speaking:
        return  # Don't interrupt the user's Q&A session
    # ... existing logic
```

**Exception:** `priority == "critical"` hazards should STILL interrupt (safety first). We can route those through `pyttsx3` which uses a separate audio channel.

### 4.4 `run_nlp.py` — Wire Everything Together

The entrypoint needs to:
1. Initialize the `STTEngine` (load Whisper model)
2. Initialize the `SceneReasoner` (with reference to ingester)
3. Start the `pynput` keyboard listener in a daemon thread
4. Start the `PayloadIngester` in the main thread (blocking)

### 4.5 `nlp_config.py` — Add Phase 2 Config Fields

```python
# STT
stt_model: str = "small"
stt_device: str = "cpu"       # DECIDED: CPU to preserve GPU for CV + VLM
stt_compute_type: str = "int8"
stt_silence_timeout: float = 2.0

# Hotkey
push_to_talk_key: str = "f5"  # DECIDED: F5 avoids terminal Space conflicts

# Reasoner
vlm_enabled: bool = True       # Enable VLM fallback for unmatched questions
vlm_model: str = "Qwen/Qwen3-VL-2B"
vlm_text_fallback_model: str = "Qwen/Qwen3.5-4B"
```

---

## 5. New Dependencies

| Package | Purpose | Install |
|---|---|---|
| `faster-whisper` | STT (Whisper model inference) | `pip install faster-whisper` |
| `sounddevice` | Microphone audio capture | `pip install sounddevice` |
| `pynput` | Keyboard hotkey listener | `pip install pynput` |
| `numpy` | Audio buffer handling | Already installed |

---

## 6. VRAM Budget (Phase 2)

### Steady-State (No VLM active)

| Component | VRAM | Device |
|---|---|---|
| YOLOE-11s-seg | ~700 MB | GPU |
| MiDaS Small | ~400 MB | GPU |
| EasyOCR | ~350 MB | GPU |
| CUDA overhead | ~500 MB | GPU |
| **CV Subtotal** | **~1.95 GB** | |
| faster-whisper small (INT8) | **0 MB** | CPU (RAM only) |
| Edge-TTS / pyttsx3 | 0 MB | Network / CPU |
| **Steady-State Total** | **~1.95 GB** | **32% of 6 GB** |

### Peak (VLM active — during a visual question)

| Component | VRAM | Notes |
|---|---|---|
| CV Pipeline (steady-state) | ~1.95 GB | Continues running |
| Qwen3-VL-2B (FP16) | ~2.50 GB | Loaded on demand, unloaded after response |
| **Peak Total** | **~4.45 GB** | **74% of 6 GB — within safe headroom** |

> **VLM loading strategy:** Qwen3-VL-2B is **not** kept resident in VRAM. It is loaded when a VLM question is triggered, generates the response, then unloaded. This takes ~3-5s but keeps the CV pipeline running at full FPS during idle time.

---

## 7. User Interaction Flow (End-to-End)

```
+-------------------------------------------------------------------+
|                    Normal Operation (Phase 1)                      |
|                                                                    |
|  CV payloads arrive -> Announcer fires proactive alerts            |
|  "Caution. Wet floor ahead, Center-Right."                         |
|  "Unknown person detected on your Left."                           |
|                                                                    |
|  ----------- User presses SPACE -----------                        |
|                                                                    |
|  -> Proactive announcements PAUSE (except critical hazards)        |
|  -> Microphone starts recording                                    |
|  -> Audio indicator tone plays (short beep / click)                |
|                                                                    |
|  User speaks: "What's around me?"                                  |
|                                                                    |
|  ----------- User releases SPACE -----------                       |
|                                                                    |
|  -> Recording stops                                                |
|  -> faster-whisper transcribes: "What's around me?"                |
|  -> SceneReasoner classifies intent: SCENE_SUMMARY                 |
|  -> SceneReasoner reads latest payload from ingester.history       |
|  -> Constructs response:                                           |
|    "There are 4 objects around you. Nearby: a person on your       |
|     left, very close. A dining chair ahead. Further away:          |
|     a cabinet on your right. Sign reads: 603-67."                  |
|  -> TTS speaks the response                                        |
|  -> Proactive announcements RESUME                                 |
|                                                                    |
+-------------------------------------------------------------------+
```

---

## 8. Implementation Steps (Ordered)

### Step 1: `stt.py` -- STT Engine
- [ ] Install `faster-whisper`, `sounddevice`
- [ ] Implement `STTEngine` class: `start_recording()`, `stop_recording_and_transcribe()`
- [ ] Handle edge cases: empty recording, very short recording, Whisper errors
- [ ] Unit test: record a short clip, transcribe, assert non-empty result

### Step 2: `reasoner.py` -- Scene Reasoner
- [ ] Implement intent classifier (keyword matching)
- [ ] Implement response handlers for each intent:
  - `_handle_scene_summary()`
  - `_handle_hazard_check()`
  - `_handle_people_check()`
  - `_handle_ocr_read()`
  - `_handle_language_switch()`
  - `_handle_status()`
  - `_handle_unknown()`
- [ ] Add depth-to-distance label mapping
- [ ] Multilingual response templates (EN, HI, GU)
- [ ] Unit test: mock payload -> assert correct response string

### Step 3: Modify `tts.py` -- Add Speaking Lock
- [ ] Add `threading.Lock` and `is_speaking` flag
- [ ] Wrap all speak methods with the lock

### Step 4: Modify `announcer.py` -- Respect Lock
- [ ] Skip non-critical announcements when `tts_manager.is_speaking`
- [ ] Critical hazards still go through `pyttsx3` (separate channel)

### Step 5: Hotkey Listener + Wiring in `run_nlp.py`
- [ ] Install `pynput`
- [ ] Add `--push-to-talk-key` CLI argument (default: `space`)
- [ ] Spawn pynput listener as daemon thread
- [ ] On key press: `stt.start_recording()` + play beep
- [ ] On key release: `text = stt.stop_and_transcribe()` -> `reasoner.answer(text)` -> `tts.speak(answer)`
- [ ] Integrate with the existing ingester main loop

### Step 6: Config Updates
- [ ] Add STT and hotkey fields to `nlp_config.py`
- [ ] Fields already exist in `config.yaml` -- ensure they are loaded

### Step 7: Testing
- [ ] Manual integration test: run pipeline on video + NLP engine -> press Space -> ask questions
- [ ] Test all 6 intents
- [ ] Test language switching mid-session
- [ ] Test that critical hazard announcements still interrupt during Q&A
- [ ] Test graceful handling when no CV data is available

---

## 9. File Changes Summary

| File | Action | What Changes |
|---|---|---|
| `src/nlp_engine/stt.py` | **CREATE** | STT engine with faster-whisper + sounddevice (CPU) |
| `src/nlp_engine/reasoner.py` | **CREATE** | Scene Reasoner: keyword intent classifier + template handlers + VLM escalation |
| `src/nlp_engine/vlm.py` | **CREATE** | VLM Engine: Qwen3-VL-2B loader/unloader + inference + text-only fallback |
| `src/nlp_engine/tts.py` | **MODIFY** | Add `_speaking_lock` and `is_speaking` flag |
| `src/nlp_engine/announcer.py` | **MODIFY** | Check `tts_manager.is_speaking` before non-critical announcements |
| `src/nlp_engine/nlp_config.py` | **MODIFY** | Add STT + hotkey + VLM config fields |
| `scripts/run_nlp.py` | **MODIFY** | Add F5 pynput listener thread, wire STT -> Reasoner -> (Template or VLM) -> TTS |
| `requirements.txt` | **MODIFY** | Add `faster-whisper`, `sounddevice`, `pynput`, `transformers`, `accelerate` |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Microphone not available on test machine | STT fails | Graceful fallback: log warning, disable push-to-talk, proactive announcements still work |
| `pynput` doesn't capture key in some terminal contexts | Hotkey doesn't fire | Provide alternative: `--no-hotkey` mode with stdin text input for testing |
| Whisper transcription accuracy for Hindi/Gujarati | Wrong intent | Keep intent keywords broad; add multiple transliterations (e.g., "kya hai", "kya hain") |
| Audio feedback (beep) on key press | Needs an audio file | Generate a simple sine-wave tone programmatically or bundle a tiny WAV |
| TTS and STT fighting over audio device | Garbled audio | Use speaking lock; STT only records while TTS is silent |

---

## 11. Resolved Design Decisions

| # | Decision | Resolution |
|---|---|---|
| 1 | **Whisper device** | **CPU** — preserves GPU for CV + VLM |
| 2 | **Whisper loading** | **At startup** — no cold-start delay on first question |
| 3 | **Push-to-talk key** | **F5** — avoids Space conflicts in terminal |
| 4 | **Audio beep** | **Yes** — programmatic sine-wave tone (no external file needed) |
| 5 | **Payload aggregation** | **Latest single payload** — keeps implementation simple; history is available if needed |
| 6 | **Language switching** | Changes **both** TTS voice AND Whisper's language hint for best multilingual accuracy |
| 7 | **Critical hazard during Q&A** | **Interrupt immediately via pyttsx3** (separate audio channel) — safety is paramount |
| 8 | **Unmatched questions** | **Escalate to VLM** — no "I don't understand" dead ends |

---

## 12. VLM Engine — `vlm.py`

**Responsibility:** Answer open-ended visual questions that the template reasoner cannot handle from JSON alone.

### 12.1 When Is VLM Triggered?

The `SceneReasoner` escalates to VLM when:
- The user's question does not match any keyword intent (`VLM_QUERY` intent).
- Example questions that require VLM:
  - *"What color is the car?"*
  - *"Is it safe to cross?"*
  - *"What's written on that door?"*
  - *"Is that person carrying anything?"*

### 12.2 Model Specs

| Property | Primary | Text-Only Fallback |
|---|---|---|
| **Model** | `Qwen/Qwen3-VL-2B` | `Qwen/Qwen3.5-4B` |
| **Input** | Image (`latest_frame.jpg`) + JSON context + question | JSON context + question only |
| **VRAM** | ~2.50 GB (FP16) | ~3.0 GB (FP16) |
| **Latency** | ~3–5s | ~2–4s |
| **When used** | Frame available (`has_image: true`) | Frame not available / image request failed |

### 12.3 Frame Acquisition

Before invoking the VLM, the engine must request a fresh frame via the Pull Architecture:

```python
def _request_frame(self) -> Optional[str]:
    """Signal CV pipeline to save latest_frame.jpg. Returns path if successful."""
    flag_path = "ipc/frame_request.flag"
    frame_path = "ipc/latest_frame.jpg"
    
    # Write flag → CV pipeline sees it within next frame interval
    open(flag_path, 'w').close()
    
    # Wait up to 2s for the frame to appear
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if os.path.exists(frame_path):
            return frame_path
        time.sleep(0.05)
    
    logger.warning("Frame request timed out. Falling back to text-only VLM.")
    return None
```

### 12.4 VLM Inference Flow

```
User asks: "What color is the car?"
   |
   v
SceneReasoner: intent = VLM_QUERY
   |
   v
VLMEngine._request_frame()
   → writes ipc/frame_request.flag
   → waits for ipc/latest_frame.jpg (up to 2s)
   |
   v
if frame_available:
    VLMEngine.load_model()  # Load Qwen3-VL-2B into VRAM (~3-5s, one-time cost)
    Build prompt:
      system: "You are a visual assistant. Answer briefly in 1-2 sentences."
      context: <scene_understanding JSON summary>
      image: latest_frame.jpg
      question: "What color is the car?"
    Run inference → answer text
    VLMEngine.unload_model()  # Free VRAM immediately
else:
    VLMEngine.load_text_model()  # Load Qwen3.5-4B (text-only)
    Build prompt with JSON context only
    Run inference → answer text
    VLMEngine.unload_text_model()
   |
   v
TTS speaks: "The car appears to be silver, parked on your right."
```

### 12.5 Implementation Skeleton

```python
class VLMEngine:
    def __init__(self, vlm_model: str, text_model: str):
        self.vlm_model_name = vlm_model
        self.text_model_name = text_model
        self._model = None
        self._processor = None

    def answer(self, question: str, payload: dict, language: str = "en") -> str:
        """Request frame, load VLM, run inference, unload. Returns answer string."""
        frame_path = self._request_frame()
        
        if frame_path:
            return self._run_vlm(question, payload, frame_path, language)
        else:
            return self._run_text_llm(question, payload, language)

    def _run_vlm(self, question, payload, frame_path, language):
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            import torch
            
            logger.info("Loading Qwen3-VL-2B into VRAM...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.vlm_model_name,
                torch_dtype=torch.float16,
                device_map="cuda"
            )
            processor = AutoProcessor.from_pretrained(self.vlm_model_name)
            
            # Build context string from payload
            context = self._build_context_string(payload)
            prompt = f"Scene context: {context}\n\nQuestion: {question}"
            
            # Build messages with image
            from PIL import Image
            image = Image.open(frame_path)
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_input], images=[image], return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=100)
            
            answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
            return answer.strip()
            
        finally:
            # Always unload to free VRAM
            del model, processor
            import gc; gc.collect()
            torch.cuda.empty_cache()
            logger.info("Qwen3-VL-2B unloaded from VRAM.")

    def _build_context_string(self, payload: dict) -> str:
        """Convert JSON payload to a concise text summary for LLM context."""
        items = payload.get("scene_understanding", [])
        faces = payload.get("recognized_faces", [])
        texts = payload.get("environmental_text", [])
        
        parts = []
        for item in items:
            cls = item.get("class", "object")
            pos = item.get("position", "ahead")
            depth = item.get("depth_score", 0)
            dist = "very close" if depth >= 0.8 else "nearby" if depth >= 0.5 else "far"
            parts.append(f"{cls} ({pos}, {dist})")
        
        if faces:
            parts.append(f"Recognized: {', '.join(f.get('name','?') for f in faces)}")
        if texts:
            text_strs = [t['text'] for t in texts if isinstance(t, dict)]
            if text_strs:
                parts.append(f"Signs: {', '.join(text_strs)}")
        
        return "; ".join(parts) if parts else "No objects detected."
```

### 12.6 User-Facing VLM Experience

```
User presses F5: "Is it safe to cross the road?"
   
  → TTS: "Let me take a look..."  (immediate acknowledgement, <0.5s)
  → Frame requested from CV pipeline
  → VLM loads and processes (~3-5s)
  → TTS: "There is a pedestrian crossing ahead with a red traffic light.
           It is not safe to cross right now."
```

The immediate *"Let me take a look..."* acknowledgement is critical — it tells the user the system heard them and is working, so they don't press F5 again.

### 12.7 VLM Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| VLM load time blocks TTS for 3-5s | User confusion | Immediately speak "Let me take a look..." before loading |
| OOM if CV pipeline spikes during VLM load | CUDA OOM crash | Check `torch.cuda.memory_reserved()` before loading; fall back to text-only if insufficient |
| VLM hallucinates | Wrong answer | Prefix responses with "I think..." or "It looks like..." to signal uncertainty |
| Frame stale (captured 1s ago) | Wrong frame for question | Acceptable — user asked about the scene at that moment |
| `has_image: false` in payload | VLM has no image | Automatically fall back to text-only Qwen3.5-4B |

---

## 13. Updated Implementation Steps

### Phase 2A — Template Q&A (Steps 1–7, no VLM)
*(Same as Section 8 above — implement and test these first)*

### Phase 2B — VLM Integration (Steps 8–11)

#### Step 8: `vlm.py` — VLM Engine
- [x] Install `transformers==4.57.6`, `accelerate`, `qwen-vl-utils`, `Pillow`
- [x] Implement `VLMEngine`: `answer()`, `_request_frame()`, `_build_context_string()`, dynamic load/unload
- [x] Test VLM load/unload cycle: verified VRAM allocated and freed (0 MB after unload)
- [x] Verified sample image forward inference with Qwen3-VL-2B snapshot

#### Step 9: Wire VLM into `reasoner.py`
- [x] Add `VLM_QUERY` intent as the final fallback in `_classify_intent()`
- [x] Add `_handle_vlm_query()` that calls `vlm_engine.answer(question, payload, lang)`
- [x] Speak immediate acknowledgement (*"Let me take a look..."*) before triggering VLM

#### Step 10: Config for VLM
- [x] Add `vlm_enabled`, `vlm_model`, `vlm_max_tokens`, `vlm_temperature` to `nlp_config.py`
- [x] Add corresponding entries to `config.yaml` under `nlp:`
- [x] Added `--no-vlm` CLI flag to `scripts/run_nlp.py`

#### Step 11: VLM Integration Testing
- [x] Unit test suite created in `tests/test_nlp.py` (all 7 tests passing)
- [x] Full test suite (15 tests) passing with 100% success rate
- [x] Updated `docs/setup_guide.md` with Phase 2B test instructions
