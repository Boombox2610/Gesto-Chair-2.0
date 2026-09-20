# Gesto-Drive: Deep Learning CV Pipeline

<div align="center">

**A GPU-accelerated, multimodal scene-understanding engine for the Gesto-Drive wheelchair system.**

*Webcam / Video → YOLOE (Open-Vocabulary) + ByteTrack + MiDaS (Proximity Gated) + Face Recognition + EasyOCR → Structured JSON → NLP Layer*

</div>

---

## Overview

Gesto-Drive's CV pipeline acts as a **"dumb sensor"** — it observes the environment and emits a structured, versioned JSON payload every frame. All semantic reasoning lives in the NLP layer downstream. The CV system concerns itself with *what is there*, *where it is*, *how close it is*, *who it is*, *what text is visible*, and *what level of action priority* it carries.

```
Webcam/Video ──► YOLOE + ByteTrack ──► Spatial Mapper ──┐
                     │                                   │
                     ├───► person ROIs                   │
                     │     ▼                             │
                     │     Face Recognizer ──────────────┼──► Payload Builder ──► stdout (JSONL)
                     │                                   │                    ──► payloads.jsonl
                     ├───► proximity hazards/agents      │
                     │     ▼                             │
                     │     MiDaS Depth ──────────────────┤
                     │                                   │
                     └───► text-sign detections          │
                           ▼                             │
                           EasyOCR ──────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the full data-flow diagram.

---

## Quick Start

### Prerequisites
- Windows 10/11, Python 3.12, NVIDIA GPU (CUDA 12.x)
- Visual Studio C++ Build Tools **or** `conda` (for dlib)

### 1. Activate the virtual environment
```bash
.\venv\Scripts\activate
```

### 2. Install PyTorch (CUDA first — order matters)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install pipeline dependencies
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 4. Enroll known faces *(optional — needed for face recognition)*
```bash
python scripts/enroll_face.py enroll "Alice" path/to/alice.jpg
python scripts/enroll_face.py list
```

### 5. Run the pipeline

```bash
# Live Webcam (Outdoor Mode)
python scripts/run_pipeline.py

# Switch to Indoor Scene Mode
python scripts/run_pipeline.py --scene indoor

# Run on Sample Videos (Located in test-Vids/)
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-trial1.mp4 --scene outdoor
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-indoorTrial1.mp4 --scene indoor

# Offline Mode (after precomputing embeddings)
python scripts/run_pipeline.py --offline
```

### 6. Launch the Streamlit dashboard *(separate terminal)*
```bash
streamlit run dashboard/app.py
```

---

## Configuration (`config.yaml`)

All parameters, model variants, confidence thresholds, and YOLOE text prompts are centrally configurable via **`config.yaml`** at the project root:

- **Pipeline settings:** camera/video source, frame resolution, target FPS, spatial zones.
- **Model settings:** YOLOE weights, MiDaS depth area gates, OCR confidence, face recognition tolerance.
- **Vocabulary lists:** Edit, add, or remove outdoor/indoor classes and trigger categories directly in YAML without modifying Python code!

---

## Project Structure

```
DL-miniProj-CV/
├── config.yaml              ← central configuration & YOLOE vocabulary prompts
├── docs/                    ← documentation
│   ├── architecture.md      ← system diagram & conditional trigger data flow
│   ├── payload_schema_v1.md ← JSON schema contract for NLP layer
│   ├── setup_guide.md       ← Windows + CUDA install walkthrough & video tests
│   └── performance_notes.md ← FPS benchmarks, VRAM budget & tuning knobs
├── src/cv_pipeline/         ← core pipeline modules
│   ├── config.py            ← PipelineConfig dataclass + YAML loader
│   ├── vocabulary.py        ← open-vocabulary definitions & semantic gates
│   ├── capture.py           ← webcam/video capture + FPS monitor
│   ├── detector.py          ← YOLOE open-vocabulary detector + ByteTrack
│   ├── depth.py             ← MiDaS monocular depth estimation
│   ├── face_recognizer.py   ← face_recognition/dlib wrapper
│   ├── ocr.py               ← event-driven EasyOCR wrapper
│   ├── spatial.py           ← bbox → spatial zone mapping
│   ├── payload.py           ← JSON schema v1.0 builder with priority tagging
│   └── pipeline.py          ← orchestration loop
├── dashboard/app.py         ← Streamlit debug/demo UI
├── scripts/
│   ├── run_pipeline.py      ← CLI entrypoint
│   ├── precompute_embeddings.py ← YOLOE text embedding pre-cacher
│   └── enroll_face.py       ← face enrollment CLI
├── data/                    ← face DB and offline cached embeddings
├── test-Vids/               ← sample outdoor & indoor wheelchair trial videos
├── tests/                   ← unit tests (no GPU required)
├── requirements.txt
└── requirements-dev.txt
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Object Detection | Ultralytics YOLOE (`yoloe-11s-seg.pt`) | Zero-shot detection of 50+ custom hazards (potholes, tactile paving, glass doors, etc.) |
| Tracking | Ultralytics ByteTrack | Built-in, persistent `track_id` across frames |
| Depth Estimation | MiDaS Small (Proximity Gated) | Triggers on hazard area >4% or agent area >8%; falls back to lazy interval |
| Face Recognition | `face_recognition` / dlib | CPU-threaded on person ROIs; does not block the GPU pipeline |
| OCR | EasyOCR (Event-Driven) | Triggers only when sign/text classes are detected; 0 GPU compute otherwise |
| Semantic Priority | `priority` field tagging | Categorizes detections into `critical`, `standard`, `contextual` for NLP |
| Configuration | Root `config.yaml` | Single point of configuration for both runtime parameters and text prompts |

---

## Documentation

| Doc | Purpose |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full system diagram, module responsibilities, threading model |
| [`docs/payload_schema_v1.md`](docs/payload_schema_v1.md) | Complete JSON schema contract for NLP layer integration |
| [`docs/setup_guide.md`](docs/setup_guide.md) | Step-by-step Windows + CUDA + dlib install guide & sample video tests |
| [`docs/performance_notes.md`](docs/performance_notes.md) | FPS benchmarks, VRAM budget, degradation strategy, tuning knobs |
