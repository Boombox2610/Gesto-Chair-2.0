# Phase 3 — Interactive Voice Face Enrollment: Implementation Plan

> **Status:** ✅ COMPLETE — Interactive Voice Face Enrollment fully implemented and tested  
> **Created:** 2026-10-15  
> **Parent doc:** [nlp_plan.md](./nlp_plan.md)  
> **Depends on:** Phase 1 (Dual TTS) & Phase 2 (STT & Pull Architecture) — ✅ BOTH COMPLETE  
> **Scope:** Hands-free voice enrollment of unknown people detected in the camera view — ✅ COMPLETE  

---

## 1. Phase 3 Objective

Give visually impaired users the ability to **identify and enroll new people using only voice prompts and push-to-talk (`F5`)**. 

When an unknown person is detected nearby:
1. The system alerts the user: *"Unknown person detected on your Left. Would you like to add them?"*
2. If the user confirms (*"Yes"* / *"Haan"*), the system asks: *"What is their name?"*
3. The user speaks the name (*"Rahul"*).
4. The system automatically captures `latest_frame.jpg`, crops the person's face, computes face encodings, updates the face database (`data/known_faces.pkl`), and confirms: *"Rahul has been added."*
5. The CV pipeline immediately recognizes that person on subsequent frames without requiring a restart.

---

## 2. User Interaction Flow (Dialog State Machine)

```
[Normal Operation]
       │
       ▼
Unknown Person Detected (depth_score >= 0.5, no recognized_face, not on cooldown)
       │
       ▼
TTS: "Unknown person detected on your [Position]. Would you like to add them?"
State: AWAIT_CONFIRMATION (Timeout: 10s)
       │
       ├── User says "No" / "Skip" / "Nahi" OR Timeout expires
       │     └─► Set 60s cooldown for this object_id
       │     └─► Return to IDLE
       │
       └── User holds F5 and says "Yes" / "Sure" / "Haan" / "Ha"
             │
             ▼
      TTS: "What is their name?"
      State: AWAIT_NAME (Timeout: 10s)
             │
             ▼
      User holds F5 and speaks name: "Rahul"
             │
             ▼
      State: ENROLLING
      1. Trigger Pull Handshake: Write frame_request.flag -> wait for latest_frame.jpg
      2. Locate face inside the person's bbox in latest_frame.jpg
      3. Compute 128-d face encoding via face_recognition
      4. Save face crop to data/sample_faces/rahul.jpg
      5. Atomically update data/known_faces.pkl
      6. CV Pipeline auto-reloads FaceDB on mtime change
             │
             ▼
      TTS: "Rahul has been added."
      State: IDLE (object_id now recognized as Rahul)
```

---

## 3. Architecture & Required Module Changes

### 3.1 CV Pipeline — Payload Schema v1.1 Update

To crop the unknown person accurately from `latest_frame.jpg`, the NLP engine needs the bounding box coordinates.

**File:** `src/cv_pipeline/payload.py` & `src/cv_pipeline/pipeline.py`

In `scene_understanding[]`:
```json
{
  "object_id": 42,
  "class": "Person",
  "confidence": 0.88,
  "position": "Center-Left",
  "depth_score": 0.62,
  "distance_confidence": "relative",
  "priority": "standard",
  "bbox": [120, 85, 340, 520]
}
```
* Bounding box format: `[x1, y1, x2, y2]` in pixel coordinates.
* Fully backward-compatible additive field.

---

### 3.2 CV Pipeline — Dynamic FaceDB Reloading

**File:** `src/cv_pipeline/face_recognizer.py`

Currently, `FaceRecognizer` loads `known_faces.pkl` once at initialization. 
We update `FaceRecognizer.recognize()` to check the file modification timestamp (`os.path.getmtime(self.db_path)`):
* If modified, reload `self.db` in-memory.
* This allows newly enrolled faces to be recognized on the very next frame without restarting the CV pipeline.

---

### 3.3 NLP Engine — Enrollment Controller (`src/nlp_engine/enrollment.py`)

**New Module:** `src/nlp_engine/enrollment.py`

#### Responsibilities:
1. **Dialog State Tracking:** Maintains current state (`IDLE`, `AWAIT_CONFIRMATION`, `AWAIT_NAME`, `ENROLLING`).
2. **Cooldown Management:** Tracks recently prompted `object_id`s with a 60-second cooldown so the user isn't spammed while walking past people.
3. **Face Extraction & Enrollment:**
   - Calls `vlm_engine._request_frame()` to get `latest_frame.jpg`.
   - Uses the person's `bbox` to crop the ROI.
   - Extracts face encoding using `face_recognition.face_encodings()`.
   - Saves cropped portrait to `data/sample_faces/<name>_<timestamp>.jpg`.
   - Updates `data/known_faces.pkl` via `FaceDB.enroll()`.
4. **Multilingual Prompts:** English, Hindi, Gujarati dialog templates.

---

### 3.4 Integration with STT & Hotkey Listener (`scripts/run_nlp.py`)

When the user presses `F5`:
1. If `enrollment_controller.is_active()`:
   - Route transcribed text directly to the enrollment dialog handler (e.g. check for "yes/no" or extract name).
2. Else:
   - Route text to `SceneReasoner.answer(text)` (normal Q&A).

---

## 4. Multilingual Dialog Templates

| Dialog Step | English | Hindi | Gujarati |
|---|---|---|---|
| **Prompt Enroll** | *"Unknown person detected on your {position}. Would you like to add them?"* | *"{position} तरफ एक अज्ञात व्यक्ति है। क्या आप उन्हें जोड़ना चाहते हैं?"* | *"{position} બાજુ એક અજાણી વ્યક્તિ છે. શું તમે તેમને ઉમેરવા માંગો છો?"* |
| **Ask Name** | *"What is their name?"* | *"उनका नाम क्या है?"* | *"તેમનું નામ શું છે?"* |
| **Success** | *"{name} has been added."* | *"{name} को जोड़ दिया गया है।"* | *"{name} ને ઉમેરી દેવામાં આવ્યા છે."* |
| **Cancelled** | *"Cancelled."* | *"रद्द कर दिया गया।"* | *"રદ કરવામાં આવ્યું."* |
| **Face Not Found** | *"Sorry, I couldn't see their face clearly. Please try again when facing them."* | *"क्षमा करें, चेहरा साफ़ नहीं दिखा। कृपया उनके सामने होकर पुनः प्रयास करें।"* | *"માફ કરશો, ચહેરો સ્પષ્ટ દેખાતો નથી. કૃપા કરીને ફરી પ્રયાસ કરો."* |

---

## 5. Implementation Steps (Ordered)

### Step 1: CV Pipeline Updates
- [x] Add `bbox` `[x1, y1, x2, y2]` to `scene_understanding` items in `payload.py`.
- [x] Add `mtime` cache invalidation to `FaceRecognizer` in `face_recognizer.py` so it reloads `known_faces.pkl` dynamically.

### Step 2: Build `src/nlp_engine/enrollment.py`
- [x] Implement `EnrollmentController` class with dialog states and timeouts.
- [x] Implement person trigger check: Person detection + depth >= 0.5 + no face ID + not on cooldown.
- [x] Implement face crop & enrollment logic using `FaceDB`.
- [x] Add multilingual dialog strings.

### Step 3: Wire into Announcement & NLP Loop
- [x] Hook `EnrollmentController` into `announcer.py` or `run_nlp.py`.
- [x] Handle push-to-talk routing in `run_nlp.py` when an enrollment dialog is awaiting user response.

### Step 4: Testing & Verification
- [x] Unit tests in `tests/test_enrollment.py` (5 passing tests).
- [x] Full test suite (20 passing tests).
- [x] Updated `docs/setup_guide.md` with voice enrollment commands.
