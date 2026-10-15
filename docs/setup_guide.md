# Setup Guide: Gesto-Drive CV Pipeline (Windows + CUDA)

**Target environment:** Windows 10/11 · Python 3.12 · NVIDIA GPU (CUDA 12.x) · venv

---

## Prerequisites Checklist

Before running any install commands, verify the following:

| Requirement | How to check | Required version |
|---|---|---|
| Python | `python --version` | 3.12.x |
| NVIDIA GPU | `nvidia-smi` | Any CUDA-capable card |
| CUDA driver | `nvidia-smi` column `CUDA Version` | 12.x or higher |
| Git | `git --version` | Any recent version |
| pip | `pip --version` | 24.x+ |

> **Tested hardware:** NVIDIA GeForce RTX 3050 6 GB Laptop GPU, driver 592.27, CUDA 12.x

---

## Step 1 — Activate the Virtual Environment

A `venv` at `.\venv\` is already created for this project (Python 3.12.8).

```powershell
# From the project root (DL-miniProj-CV/)
.\venv\Scripts\activate
```

You should see `(venv)` in your prompt. All subsequent commands assume the venv is active.

---

## Step 2 — Install PyTorch with CUDA Support

> [!IMPORTANT]
> **Install PyTorch BEFORE running `pip install -r requirements.txt`.**
> If torch is installed without CUDA first, EasyOCR and Ultralytics will both pick up the CPU-only build and CUDA acceleration will not be available.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is active:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: True  /  NVIDIA GeForce RTX 3050 6GB Laptop GPU
```

If `False` is printed, the CPU-only wheel was installed. Uninstall and repeat with the correct index URL.

---

## Step 3 — Install dlib (for Face Recognition)

`face_recognition` wraps dlib, which requires a C++ compiler to build from source on Windows. **Two options — pick one:**

### Option A: pip (requires C++ Build Tools)

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)
   - In the installer, select **"Desktop development with C++"**
2. Install [CMake](https://cmake.org/download/) and ensure it is on your `PATH`
3. Then:
   ```powershell
   pip install dlib
   pip install face_recognition
   ```

### Option B: conda (recommended if you have Anaconda/Miniconda)

No compiler or CMake required.

```powershell
conda install -c conda-forge dlib
pip install face_recognition
```

Verify face_recognition is working:
```powershell
python -c "import face_recognition; print('face_recognition OK')"
```

---

## Step 4 — Install Pipeline Dependencies

```powershell
pip install -r requirements.txt
```

This installs: `opencv-python`, `numpy`, `ultralytics` (YOLOv8 + ByteTrack), `timm` (MiDaS backbone), `easyocr`.

---

## Step 5 — Install Dev & Dashboard Dependencies

```powershell
pip install -r requirements-dev.txt
```

This installs: `streamlit`, `altair`, `pandas`, `pytest`, `pytest-mock`.

---

## Step 6 — First-Run Model Downloads

On the first run, three model weights are downloaded automatically. You need an internet connection.

| Model | Size | Downloaded by |
|---|---|---|
| `yoloe-11s-seg.pt` | ~42 MB | `ultralytics` on first `YOLOE("yoloe-11s-seg.pt")` call |
| `mobileclip_blt.ts` (Text Encoder) | ~572 MB | `ultralytics` text encoder initialized on first YOLOE run |
| `yolov8n.pt` (Legacy) | ~6 MB | `ultralytics` on first `YOLO("yolov8n.pt")` call |
| `MiDaS_small` | ~80 MB | `torch.hub.load("intel-isl/MiDaS", "MiDaS_small")` |
| EasyOCR English model | ~40 MB | `easyocr.Reader(['en'])` on first instantiation |

To pre-download before going offline:
```powershell
# Precompute YOLOE embeddings and download the text encoder (mobileclip_blt.ts)
python scripts/precompute_embeddings.py

# Download legacy YOLOv8n
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Download MiDaS and EasyOCR
python -c "import torch; torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')"
python -c "import easyocr; easyocr.Reader(['en'], gpu=True)"
```

### Offline Mode
After weights are cached locally, use the `--offline` flag to skip all network calls:
```powershell
python scripts/run_pipeline.py --offline
```

---

## Step 7 — Camera Configuration

The pipeline defaults to camera index `0` (first webcam). If you have multiple cameras or the wrong one opens:

```powershell
# List available cameras (run this quick script)
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: OK')
        cap.release()
    else:
        print(f'Camera {i}: not found')
"
```

Set the correct index:
```powershell
python scripts/run_pipeline.py --camera 1
```

Or change the default in `config.yaml`:
```yaml
pipeline:
  camera_index: 0  # change 0 -> your index or video file path
```

---

## Step 8 — Enroll Known Faces

Skip this step if you don't need face recognition — the pipeline runs without a face DB.

```powershell
# Enroll a person
python scripts/enroll_face.py enroll "Alice" path\to\alice.jpg

# List all enrolled faces
python scripts/enroll_face.py list

# Delete an entry
python scripts/enroll_face.py delete "Alice"
```

**Photo requirements for best accuracy:**
- Front-facing, well-lit, single face in frame
- Minimum resolution: ~200×200 px on the face
- If `enroll` reports "No face found", the photo quality is the most common cause

The face database is stored at `data/known_faces.pkl` (gitignored).

---

## Step 9 — Run the Pipeline

```powershell
# Live Webcam (Default: CUDA, 640×480, stdout + JSONL, OpenCV window visible)
python scripts/run_pipeline.py

# Run on Sample Videos (Located in test-Vids/)
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-trial1.mp4 --scene outdoor
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-indoorTrial1.mp4 --scene indoor
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-indoorTrial2.mp4 --scene indoor

# Headless (no OpenCV window), file sink only
python scripts/run_pipeline.py --no-cv-window --sink file

# Force CPU (for testing without GPU)
python scripts/run_pipeline.py --device cpu

# Adjust FPS target (lowers interleaving aggressiveness)
python scripts/run_pipeline.py --fps-target 15.0
```

**Full CLI reference:**
```
python scripts/run_pipeline.py [--camera INDEX]
                               [--video PATH]
                               [--width WIDTH] [--height HEIGHT]
                               [--device cuda|cpu]
                               [--no-cv-window]
                               [--sink stdout|file|both]
                               [--output-file PATH.jsonl]
                               [--fps-target FLOAT]
                               [--scene outdoor|indoor]
                               [--legacy-detector]
                               [--offline]
```

---

## Step 10 — Run the NLP Engine (Voice Announcements)

The NLP Engine provides proactive spoken hazard alerts, OCR reading, and face announcements. It runs as a separate process and reads the output of the CV Pipeline.

Open a **second terminal** (venv active), then:

```powershell
# File-tail mode (reads payloads.jsonl)
python scripts/run_nlp.py --mode file-tail
```

**Alternative: Single-Terminal Piped Mode**
You can run both the CV Pipeline and the NLP Engine in a single terminal by piping the output:
```powershell
python scripts/run_pipeline.py --sink stdout | python scripts/run_nlp.py --mode pipe
```

### Testing the System (Phase 1 — Proactive Alerts)

**1. Live Testing (Webcam):**
Run the pipeline pointing at you. Hold up an object (e.g. your phone or a cup), move it close to the camera. The system should announce "Caution. [Class] ahead, [Position]". Hold up some text to the camera to test OCR.

### Testing Voice Q&A (Phase 2A — Push to Talk)

With the pipeline running, you can ask structured questions about the scene:
1. Press and **hold** the `F5` key.
2. A short beep will play indicating the microphone is listening.
3. Speak your question (e.g., *"What's around me?"*, *"Are there any hazards?"*, *"Is anyone nearby?"*).
4. Release the `F5` key.
5. The system will transcribe your speech and answer instantly using template matching.

*Note: You can switch languages mid-session by saying "Switch to Hindi" or "Switch to Gujarati" while holding F5.*

### Testing Open-Ended Visual Reasoning (Phase 2B — VLM Fallback)

If you ask an open-ended question that cannot be answered from bounding boxes alone:
1. Press and **hold** `F5`.
2. Ask any visual question:
   - *"What color is the chair?"*
   - *"Is it safe to cross?"*
   - *"What is that person holding?"*
3. Release `F5`.
4. The system will immediately acknowledge: **"Let me take a look..."** (<0.5s auditory feedback).
5. The pipeline automatically captures `latest_frame.jpg`, queries **Qwen3-VL-2B-Instruct** on-demand, speaks the reasoned answer, and immediately frees VRAM.

*(Optional: Run `python scripts/run_nlp.py --mode file-tail --no-vlm` if you want to test template matching only without VLM).*

### Testing Interactive Face Enrollment (Phase 3 — Hands-Free Voice Enrollment)

When running the pipeline on a scene or video with people, the system automatically detects unknown persons and offers to enroll them:

1. As an unknown person enters the near field (`depth_score >= 0.5`), the system announces:
   > *"Unknown person detected on your Center-Left. Would you like to add them?"*
2. **Confirming Enrollment:**
   - Hold `F5`, say: **"Yes"** (or *"Haan"* in Hindi / Gujarati).
   - Release `F5`.
3. **Providing Name:**
   - The system asks: > *"What is their name?"*
   - Hold `F5`, say: **"Alex"** (or any name, e.g., *"Rahul"*, *"Priya"*).
   - Release `F5`.
4. **Automatic Face Registration:**
   - The system says: > *"Adding Alex..."*
   - It captures the frame, crops the person's face from their bounding box, calculates face encodings, and saves to `data/known_faces.pkl`.
   - The system confirms: > *"Alex has been added to your recognized faces."*
5. **Live Verification:**
   - As the video continues playing, the CV pipeline automatically reloads `data/known_faces.pkl` without restarting!
   - On upcoming frames where this person appears, the system will now proactively announce:
     > *"Alex is nearby, on your [Position]."*
6. **Declining or Cancelling:**
   - If you say **"No"**, **"Skip"**, or remain silent for 10 seconds, the prompt cancels and enters a 60-second cooldown so you are not interrupted repeatedly.

**2. Video Testing (test-Vids):**
We provide test videos to simulate real-world scenarios without leaving your desk.

Run the CV pipeline on a test video (Terminal 1):
```powershell
# Outdoor Scene
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-trial1.mp4 --scene outdoor

# Indoor Scenes
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-indoorTrial1.mp4 --scene indoor
python scripts/run_pipeline.py --video test-Vids/gestoChair-CV-indoorTrial2.mp4 --scene indoor
```

Run the NLP engine (Terminal 2):
```powershell
python scripts/run_nlp.py --mode file-tail
```
*(Make sure your volume is up. You will hear announcements for hazards and OCR text as they appear in the video).*

---

## Step 11 — Launch the Streamlit Dashboard

Open a **second terminal** (venv active), then:

```powershell
streamlit run dashboard/app.py
```

Streamlit will open `http://localhost:8501` in your browser automatically. The dashboard reads from `payloads.jsonl` and refreshes every 500 ms. The pipeline must be running simultaneously for live data.

---

## Verify the Full Stack

Run this end-to-end sanity check:

```powershell
# 1. Schema compliance (should output valid JSON, no errors)
python scripts/run_pipeline.py --sink stdout --no-cv-window | python -m json.tool

# 2. Unit tests (no camera/GPU needed)
python -m pytest tests/ -v
```

---

## Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | venv not activated | Run `.\venv\Scripts\activate` |
| `torch.cuda.is_available()` returns `False` | CPU-only torch installed | Reinstall with `--index-url https://download.pytorch.org/whl/cu121` |
| `pip install dlib` fails with compiler error | Missing Visual C++ Build Tools | Use Option B (conda) in Step 3 |
| `Could not open webcam` | Wrong camera index | Run the camera-listing script in Step 7 |
| EasyOCR first-run very slow | Downloading model weights | Run pre-download command in Step 6 |
| `torch.hub.load` fails with SSL/connection error | Network or GitHub rate limit | Use `--offline` if weights already cached; otherwise retry |
| Streamlit dashboard shows no data | `payloads.jsonl` not found | Ensure pipeline is running with `--sink file` or `--sink both` |
