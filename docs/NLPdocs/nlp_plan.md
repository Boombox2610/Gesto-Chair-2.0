# NLP Layer — Architecture & Implementation Plan

> **Status:** DRAFT — Refining with user before implementation.
> **Last Updated:** 2026-09-29

---

## 1. High-Level Purpose

The NLP layer is the **semantic reasoning brain** of Gesto-Drive. It consumes the structured JSON payloads emitted by the CV pipeline and transforms raw detections into:

1. **Proactive spoken announcements** — hazards, recognized people, OCR text, unknown persons
2. **Interactive voice Q&A** — user asks about the scene; system answers using payload data (and optionally a VLM)
3. **Live face enrollment** — interactive voice dialog to add unknown persons to the face database

The NLP layer runs as a **separate process** from the CV pipeline. The two systems communicate via stdout pipe or shared JSONL file. This keeps them fully decoupled — the CV pipeline can run alone, and the NLP layer can be developed, tested, and restarted independently.

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CV Pipeline (Process 1)                      │
│  Webcam → YOLOE → MiDaS → FaceRec → OCR → PayloadBuilder          │
│                                  │                                  │
│                          stdout (JSONL)                              │
│                          payloads.jsonl                              │
│                          frame_request.flag (NLP → CV)               │
│                          latest_frame.jpg (CV → NLP, on demand)      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Payload Ingester  │
                    │  (stdin / file tail) │
                    └──────────┬──────────┘
                               │
          ┌────────────────────▼────────────────────┐
          │           NLP Engine (Process 2)         │
          │                                          │
          │  ┌──────────────┐  ┌──────────────────┐  │
          │  │ Announcement │  │  Scene Reasoner  │  │
          │  │   Manager    │  │  (JSON / VLM)    │  │
          │  └──────┬───────┘  └────────┬─────────┘  │
          │         │                   │             │
          │  ┌──────▼───────────────────▼──────────┐  │
          │  │        Event Router / Dispatcher     │  │
          │  └──────┬───────────────────┬──────────┘  │
          │         │                   │             │
          │  ┌──────────────┐  ┌───────▼──────────┐  │
          │  │   Dual TTS   │  │  STT Engine       │  │
          │  │Edge / pyttsx3│  │  Whisper Small    │  │
          │  └──────────────┘  └──────────────────┘  │
          │                                          │
          │  ┌──────────────────────────────────────┐ │
          │  │  Face Enrollment Controller           │ │
          │  │  (auto-crop → enroll → confirm)       │ │
          │  └──────────────────────────────────────┘ │
          └──────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### 3.1 Payload Ingester (`ingester.py`)

**Responsibility:** Continuously read CV payloads and feed them to the NLP engine.

| Mode | How it works |
|---|---|
| **Pipe mode** (default) | CV pipeline's stdout is piped: `python run_pipeline.py \| python run_nlp.py` |
| **File-tail mode** | Reads `payloads.jsonl`, tailing new lines (for when CV is already running) |

- Parses each JSON line into a Python dict.
- Feeds the latest payload to the Announcement Manager on every frame.
- Maintains a small sliding window of the last N payloads (default: 30) for scene context queries.

---

### 3.2 Announcement Manager (`announcer.py`)

**Responsibility:** Decide *what* to say and *when*, based on payload events.

#### Trigger Rules (Smart Cooldown)

| Event | Trigger condition | Cooldown logic |
|---|---|---|
| **Known person nearby** | `recognized_faces` entry with `depth_score ≥ 0.6` | Announce on first detection. Re-announce only if `depth_score` increases by ≥ 0.15 (person moving closer). Suppress same name for 30 seconds otherwise. |
| **Critical hazard** | `scene_understanding` entry with `priority == "critical"` and `depth_score ≥ 0.5` | Announce immediately. Re-announce if `depth_score` increases by ≥ 0.1. Suppress same class+position for 15 seconds. |
| **OCR text** | `environmental_text` non-empty | Announce each unique text string once per session (until it disappears for 5+ seconds, then re-arm). |
| **Unknown person** | `scene_understanding` has a Person-class detection with NO matching `recognized_faces` entry, and `depth_score ≥ 0.5` | Trigger interactive enrollment dialog (see §3.6). Suppress re-trigger for same `object_id` for 60 seconds. |

#### Announcement Templates (Multilingual)

Announcements are generated from templates, not free-form LLM text. This keeps latency near-zero and guarantees predictable output.

```python
# English
"Caution. {class} ahead, {position}."                    # "Caution. Pothole ahead, Center-Left."
"{name} is nearby, on your {position}."                   # "Harsh is nearby, on your Right."
"Sign reads: {text}."                                     # "Sign reads: STOP."
"Unknown person detected on your {position}."             # enrollment trigger

# Hindi
"सावधान। {class} आगे है, {position} तरफ।"
"{name} पास में हैं, आपके {position} तरफ।"
"बोर्ड पर लिखा है: {text}।"

# Gujarati
"સાવધાન। {class} આગળ છે, {position} બાજુ।"
"{name} નજીકમાં છે, તમારી {position} બાજુ।"
```

The active language is set via `config.yaml` or a voice command ("switch to Hindi").

---

### 3.3 Dual TTS Engine (`tts.py`)

**Responsibility:** Convert announcement text strings to spoken audio using an intelligent dual-routing strategy.

#### Dual-Routing Policy

| Event Class | Engine Used | Rationale |
|---|---|---|
| **Critical Hazards & Emergencies** (`priority == "critical"`) | **`pyttsx3`** (Offline) | Safety-critical. Cannot risk network latency, buffering, or dropped Wi-Fi/hotspot connections. Runs 100% offline with zero external dependencies. |
| **Standard Events, Faces, OCR & Scene Q&A** (`priority in ["standard", "contextual"]`) | **`edge-tts`** (Online) | High quality, natural cadence, native Indian accents for English, Hindi, and Gujarati. Zero VRAM footprint, ~1.5s latency. |
| **Network Loss Fallback** | **`pyttsx3`** (Offline) | If `edge-tts` encounters any network timeout/HTTP error, automatically fail over to `pyttsx3`. |

#### Engine Specifications

| Property | Primary (`edge-tts`) | Emergency (`pyttsx3`) |
|---|---|---|
| **VRAM Footprint** | **0 MB** (Over network / CPU) | **0 MB** (Local system synthesizer) |
| **Latency** | ~1.5s per sentence | Local, zero network delay |
| **Languages Supported** | English (`en-IN-NeerjaExpressiveNeural`), Hindi (`hi-IN-SwaraNeural`), Gujarati (`gu-IN-DhwaniNeural`) | System SAPI5 voices (English default) |
| **Audio Playback** | `pygame.mixer` | Built-in SAPI5 event loop / `pygame.mixer` |
| **Internet Required?** | Yes (Hotspot / Wi-Fi) | **No (100% Offline)** |

#### TTS Architecture

```python
class TTSManager:
    """
    Intelligently routes speech to Edge-TTS or pyttsx3 based on priority and network health.
    """
    def speak(self, text: str, language: str = "en", is_critical: bool = False):
        if is_critical:
            # Emergency route: bypass network entirely
            self.pyttsx3_backend.speak(text)
            return

        try:
            # Primary route: natural neural voice
            self.edge_backend.speak(text, language)
        except Exception as e:
            # Automatic graceful failover on network interruption
            logger.warning(f"Edge-TTS failed ({e}). Falling back to pyttsx3.")
            self.pyttsx3_backend.speak(text)
```

---

### 3.4 STT Engine (`stt.py`)

**Responsibility:** Convert user's spoken question to text.

| Property | Value |
|---|---|
| **Model** | `faster-whisper` with `small` model (INT8) |
| **VRAM** | ~0.35 GB (INT8 on CUDA) or CPU-only |
| **Languages** | Auto-detect (supports Hindi, Gujarati, English natively) |
| **Activation** | Push-to-talk hotkey (e.g., `Space` bar) or wake word |

#### Flow
1. User presses hotkey → microphone records until silence detected (VAD).
2. Audio chunk sent to `faster-whisper` → transcribed text.
3. Transcribed text routed to Scene Reasoner or Face Enrollment Controller.

---

### 3.5 Scene Reasoner (`reasoner.py`)

**Responsibility:** Answer user's natural language questions about the scene.

#### Phase 1 — JSON-Only Reasoning (No LLM)

Use the structured payload directly to answer common queries with template matching:

| User asks | System does |
|---|---|
| "What's around me?" / "Describe the scene" | Enumerate all `scene_understanding` entries by position and depth. |
| "Is there anyone nearby?" | Filter for Person-class detections. Cross-ref with `recognized_faces`. |
| "Any hazards?" | Filter for `priority == "critical"` entries. |
| "What does the sign say?" | Read `environmental_text`. |

Response is constructed from templates → fed to TTS.

```python
# Example constructed response:
"There are 3 objects around you. A pothole is ahead on your center-left, very close.
 Harsh is nearby on your right. A stop sign is on your left, it reads STOP."
```

#### Phase 2 — VLM-Powered Reasoning (If Phase 1 is insufficient)

If the user asks a question that template matching can't answer (e.g., "What color is the car?", "Is it safe to cross?"):

| Property | Value |
|---|---|
| **Model** | Qwen3-VL-2B (vision-language) — pass `latest_frame.jpg` + JSON payload |
| **Fallback** | Qwen3.5-4B (text-only) — pass JSON payload only |
| **VRAM** | ~2–3 GB (Qwen3-VL-2B, FP16) |
| **Latency** | ~2–5 seconds per response |

> **Important:** VLM and TTS cannot be resident in VRAM simultaneously alongside CV models. The system must use **model swapping** — unload TTS, load VLM, generate response text, unload VLM, load TTS, speak response.

---

### 3.6 Face Enrollment Controller (`enrollment.py`)

**Responsibility:** Interactive voice-driven face enrollment when an unknown person is detected.

#### Flow

```
1. CV payload arrives: Person detected (object_id=42), no matching recognized_face
2. Announcement Manager: "Unknown person detected on your Center-Right."
3. TTS speaks: "Unknown person detected. Would you like to add them?"
4. STT listens for response:
   ├── User says "no" / "skip" / silence → suppress object_id=42 for 60 seconds
   └── User says "yes" / "haan" / "ha"
       5. TTS speaks: "What is their name?"
       6. STT captures name → e.g., "Rahul"
       7. System auto-crops the person's bbox from latest_frame.jpg
       8. Calls FaceDB.enroll("Rahul", cropped_face_image)
       9. TTS speaks: "Rahul has been added."
```

#### Technical Details
- The person's bbox is available in `scene_understanding[].bbox` (we may need to add bbox coordinates to the payload for NLP consumption — currently only position zone is emitted).
- Auto-crop uses `latest_frame.jpg` + bbox coordinates from the detection.
- Enrollment reuses the existing `FaceDB.enroll()` and `face_recognition.face_encodings()` logic.

> **CV Schema Change Required:** Add optional `bbox` field to `scene_understanding[]` objects in the payload for NLP-side cropping. This is a backward-compatible additive change (schema v1.1).

---

## 4. Process Communication

```
┌─────────────┐     stdout pipe      ┌─────────────┐
│  CV Pipeline │ ──────────────────► │  NLP Engine  │
│  (Process 1) │                     │  (Process 2) │
└──────┬───────┘                     └──────┬───────┘
       │                                    │
       │  payloads.jsonl (shared file)      │
       │  frame_request.flag (signal file)  │
       │  latest_frame.jpg (on demand)      │
       │◄──────────────────────────────────►│
       │                                    │
       │  data/known_faces.pkl (shared DB)  │
       │◄──────────────────────────────────►│
```

### Launch Commands

```powershell
# Option 1: Piped mode (recommended for real-time)
python scripts/run_pipeline.py | python scripts/run_nlp.py

# Option 2: Independent processes (file-tail mode)
# Terminal 1:
python scripts/run_pipeline.py
# Terminal 2:
python scripts/run_nlp.py --mode file-tail
```

---

## 4.5 Frame Request Handshake (Pull Architecture)

To eliminate the overhead of continuously writing/sending images from the CV Pipeline to the NLP Engine, we use a **Pull Architecture**:

1. **Default State:** The CV Pipeline only emits lightweight JSON payloads. It does *not* write image frames to disk or pipe.
2. **Trigger:** When the NLP Engine needs an image (e.g., user asks "What's in front of me?", or triggers Face Enrollment), it writes an empty signal file: `ipc/frame_request.flag`.
3. **Fulfillment:** At the start of its processing loop, the CV Pipeline checks for `frame_request.flag`. 
    - If found, it saves the current frame to `ipc/latest_frame.jpg`, adds a flag to the payload (`"has_image": true`), and deletes `frame_request.flag`.
4. **Consumption:** The NLP Engine reads `latest_frame.jpg` and uses it for VQA or cropping.

This ensures frames are only transferred on demand, saving bandwidth and disk I/O.

---

## 5. VRAM Budget (Combined CV + NLP)

With the Dual-TTS architecture, TTS has a **0 MB VRAM footprint** on the GPU!

| Component | VRAM (FP16 / INT8) | Host / Device |
|---|---|---|
| YOLOE-11s-seg | ~700 MB | GPU (CUDA) |
| MiDaS Small | ~400 MB | GPU (CUDA) |
| EasyOCR English | ~350 MB | GPU (CUDA) |
| CUDA runtime overhead | ~500 MB | GPU (CUDA) |
| **CV Pipeline Subtotal** | **~1.95 GB** | **GPU (CUDA)** |
| **Edge-TTS (Primary)** | **0 MB** | Cloud / Network |
| **pyttsx3 (Emergency)** | **0 MB** | CPU (Local SAPI5) |
| faster-whisper Small (INT8) | ~0.35 GB | GPU / CPU |
| **Full Real-time Stack Total** | **~2.30 GB** | **Comfortably fits in 6 GB!** |
| Qwen3-VL-2B (Phase 4 Scene Q&A) | ~2.50 GB | GPU (on-demand) |
| **Peak with VLM Active** | **~4.80 GB** | **Within 6 GB budget (< 80% usage)** |

> **Key Benefit:** Because both TTS backends consume 0 MB of VRAM, the CV pipeline runs at a rock-solid 30 FPS without GPU contention, and we never risk CUDA Out-Of-Memory errors.

---

## 6. Configuration (additions to `config.yaml`)

```yaml
nlp:
  # Language & Dual TTS
  language: "en"                          # Default language: "en", "hi", "gu"
  tts_primary_engine: "edge-tts"          # Primary online engine
  tts_emergency_engine: "pyttsx3"         # Emergency offline engine (critical hazards)
  tts_rate: "+10%"                        # Speed rate (+10%, +0%, -5%)
  tts_volume: "+0%"                       # Volume (+0%, -10%)
  tts_pitch: "+0Hz"                       # Pitch (+0Hz, +2Hz)

  # Edge-TTS Voice Mapping
  voices:
    en: "en-IN-NeerjaExpressiveNeural"    # Indian English (Expressive Female)
    hi: "hi-IN-SwaraNeural"               # Hindi (Female)
    gu: "gu-IN-DhwaniNeural"              # Gujarati (Female)

  # Emergency pyttsx3 parameters
  pyttsx3_rate: 150                       # Words per minute for critical alerts

  # STT
  stt_model: "small"                      # faster-whisper model size
  stt_device: "cuda"                      # "cuda" or "cpu"
  stt_compute_type: "int8"                # "int8" for minimal VRAM
  stt_silence_timeout: 2.0                # Seconds of silence before STT stops recording

  # Announcements & Smart Cooldown
  announce_hazard_depth_thresh: 0.5       # Only announce hazards closer than this
  announce_person_depth_thresh: 0.6       # Only announce people closer than this
  announce_hazard_cooldown: 15            # Seconds before re-announcing same hazard
  announce_person_cooldown: 30            # Seconds before re-announcing same person
  announce_depth_increase_thresh: 0.1     # Re-announce if depth_score increases by this much
  announce_ocr_cooldown: 30               # Seconds before re-announcing same OCR text

  # Scene Reasoner
  reasoner_mode: "json"                   # "json" (Phase 2) or "vlm" (Phase 4)
  vlm_model: "Qwen/Qwen3-VL-2B"           # Vision-language model (Phase 4)
  llm_model: "Qwen/Qwen3.5-4B"            # Text-only fallback (Phase 4)

  # Face Enrollment
  enrollment_enabled: true                # Enable interactive voice enrollment
  enrollment_cooldown: 60                 # Seconds before re-triggering for same object_id
```

---

## 7. File Structure

```
DL-miniProj-CV/
├── src/
│   ├── cv_pipeline/          ← existing CV modules (unchanged)
│   └── nlp_engine/           ← NEW: NLP layer modules
│       ├── __init__.py
│       ├── ingester.py       ← payload reader (stdin pipe / file tail)
│       ├── announcer.py      ← announcement manager (smart cooldown + priority routing)
│       ├── tts.py            ← Dual TTS manager (Edge-TTS primary + pyttsx3 emergency)
│       ├── stt.py            ← STT backend (faster-whisper)
│       ├── reasoner.py       ← scene Q&A (JSON templates / VLM)
│       ├── enrollment.py     ← interactive voice face enrollment
│       └── nlp_config.py     ← NLP-specific config loader
├── scripts/
│   ├── run_pipeline.py       ← existing CV entrypoint
│   └── run_nlp.py            ← NEW: NLP entrypoint
├── docs/NLPdocs/
│   └── nlp_plan.md           ← this document
```

---

## 8. Implementation Phases

### Phase 1 — Dual TTS & Core Announcements (Template-based, no LLM)
1. Build `ingester.py` — read JSON payloads from stdin pipe or file tail
2. Build `announcer.py` — smart cooldown event detection & priority triage
3. Build `tts.py` — Dual TTS manager:
   - Primary: `edge-tts` (English, Hindi, Gujarati) with natural speech rate/pitch
   - Emergency route: `pyttsx3` for immediate zero-latency critical alerts (`priority == "critical"`)
   - Network watchdog: automatically switch to `pyttsx3` if internet drops
4. Build `run_nlp.py` — CLI entrypoint
5. **Test:** Pipe CV output (`run_pipeline.py | run_nlp.py`) → verify spoken hazard/person/OCR alerts
6. **Deliverable:** Real-time multimodal spoken alerts in English/Hindi/Gujarati with zero GPU contention

### Phase 2 — Voice Input & Scene Q&A (Template-based)
1. Build `stt.py` — faster-whisper STT wrapper (INT8 on GPU/CPU)
2. Build `reasoner.py` — JSON payload template-based Q&A
3. Wire push-to-talk hotkey into `run_nlp.py`
4. **Test:** User asks "What's around me?" → system speaks structured scene description
5. **Deliverable:** Two-way voice interaction with structured scene understanding

### Phase 3 — Interactive Face Enrollment
1. Add `bbox` coordinates to CV payload (schema v1.1 bump)
2. Build `enrollment.py` — voice dialog ("Unknown, do you want to add them?") + auto-crop + FaceDB enrollment
3. **Test:** Unknown person → voice prompt → user names them → enrolled into `data/known_faces.pkl`
4. **Deliverable:** Hands-free face enrollment via voice

### Phase 4 — VLM-Powered Scene Understanding (If needed)
1. Integrate Qwen3-VL-2B into `reasoner.py` as a visual fallback
2. Pass `latest_frame.jpg` + JSON payload for open-ended questions (e.g., "What color is the car?", "Is it safe?")
3. **Test:** User asks complex visual questions → VLM answers
4. **Deliverable:** Rich visual scene reasoning without compromising steady-state CV FPS

---

## 9. Design Decisions & Status

| # | Topic | Decision | Status |
|---|---|---|---|
| 1 | **TTS Engine Selection** | **Dual TTS:** `edge-tts` (Primary, multilingual, 0 VRAM) + `pyttsx3` (Emergency offline routing) | **RESOLVED** (Verified on RTX 3050) |
| 2 | **TTS Latency & VRAM** | `edge-tts` gives ~1.5s latency, 0 MB VRAM. `pyttsx3` gives zero network risk for hazards. | **RESOLVED** |
| 3 | **Push-to-talk mechanism** | Keyboard hotkey (Space bar) / UI button | Decided for Phase 2 |
| 4 | **Payload bbox addition** | Add optional `bbox_xyxy` to `scene_understanding[]` in CV payload for auto-cropping | Scheduled for Phase 3 |
| 5 | **Audio library** | `pygame.mixer` (clean, cross-platform playback of generated MP3/WAV files) | **RESOLVED** |
| 6 | **Language switching** | Configurable via `config.yaml` (`en`, `hi`, `gu`) or voice command in Phase 2 | **RESOLVED** |

