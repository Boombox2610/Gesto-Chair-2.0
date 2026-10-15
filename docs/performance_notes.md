# Performance Notes: Gesto-Drive CV Pipeline

**Hardware baseline:** NVIDIA GeForce RTX 3050 6 GB Laptop GPU · Python 3.12 · Windows 11

---

## 1. FPS Expectations by Configuration

All estimates at **640×480 resolution** on the RTX 3050 Laptop GPU. Actual numbers will vary with scene complexity (number of detected objects) and CPU thermal state.

| Configuration | Expected FPS | Notes |
|---|---|---|
| YOLOE-11s only (every frame) | 40–55 | Baseline — no depth, no face, no OCR |
| YOLOE-11s + ByteTrack | 32–45 | Core tracking loop |
| + Conditional MiDaS (~every 6th frame avg) | 28–38 | Depth triggers only on hazard/agent proximity |
| + Event-driven EasyOCR (sign-triggered only) | 27–36 | OCR adds zero cost when no signs are present |
| + Face recognition every 5th frame (dlib CPU) | **22–30** | **Realistic full-pipeline steady-state** |
| All models every frame (no interleaving) | 5–10 | Not recommended; for benchmarking only |
| CPU-only (no GPU) | 1–4 | Fallback mode — `"degraded"` status likely |

> **Target FPS:** 25 FPS minimum steady-state. The full pipeline is expected to comfortably meet this on GPU.

---

## 2. VRAM Budget (RTX 3050 6 GB)

| Model | Approx. VRAM Usage |
|---|---|
| YOLOE-11s-seg | ~700 MB |
| MiDaS Small | ~400 MB |
| EasyOCR English | ~350 MB |
| OS + CUDA runtime overhead | ~500 MB |
| **Total estimated peak** | **~1.95 GB** |

The RTX 3050 6 GB has **~4.0 GB headroom** above estimated peak usage. CUDA OOM is unlikely under normal operation but the pipeline includes a guard (see Section 5).

> **dlib (face_recognition) runs on CPU and uses no VRAM.** It uses ~50–100 MB of system RAM per active recognition call.

---

## 3. Conditional Triggering Strategy

The pipeline does not run all models every frame. This is the **semantic scheduling** logic:

| Model | Schedule | Rationale |
|---|---|---|
| YOLOE + ByteTrack | Every frame | Core detection must be real-time for tracking continuity |
| Spatial zone mapping | Every frame | Zero-cost computation (pure arithmetic on bbox) |
| MiDaS depth | Proximity Gated | Triggers when a hazard is >4% of frame or agent >8%, else falls back to every 6th frame |
| Face recognition | Every 5th frame, person-group gated | dlib is CPU-bound; face identity doesn't change frame-to-frame |
| EasyOCR | Event-Driven (ROI-triggered) | Fires only when YOLOE detects text-bearing labels (stop sign, traffic light, wet floor sign). Zero cost otherwise |

**Why conditionally interleave?**
- Keeps average FPS high (~30 FPS) despite using a heavier YOLOE detector
- Reduces thermal load on the laptop GPU during extended sessions
- Allows the Streamlit dashboard and JSONL file I/O to share CPU cycles without stalling the CUDA pipeline

---

## 4. Automatic Degradation

If the rolling 30-frame FPS average drops below `fps_target` (default: 10.0) for **5 consecutive frames**, the pipeline automatically:

1. Increases `depth_fallback_interval` by +2
2. Increases `ocr_every_n_frames` to 20 (legacy fallback)
3. Sets `status` → `"degraded"` in the payload
4. Logs a warning to stderr

The degraded schedule is maintained until FPS recovers above `fps_target` for 10 consecutive frames, at which point the original schedule is restored and `status` returns to `"ok"`.

**Common degradation triggers:**
- Background processes consuming GPU (Windows Update, another CUDA app)
- Thermal throttling on a laptop after extended use
- High object count in frame (YOLO scales with number of detections)
- First-time EasyOCR model load (slow cold start — only on first frame after OCR trigger)

---

## 5. CUDA Out-of-Memory Guard

Every CUDA inference call is wrapped:

```python
try:
    result = model(frame_tensor)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    result = model_cpu(frame_tensor)   # CPU fallback for this frame
    errors.append("CUDA OOM — fell back to CPU for this frame")
```

This prevents a total pipeline crash if another application steals VRAM. The `errors` field in the payload will contain the OOM message.

---

## 6. Tuning Knobs

All parameters live in `src/cv_pipeline/config.py`. Key performance-related values:

| Parameter | Default | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `depth_fallback_interval` | `6` | Lower VRAM/CPU load, less accurate depth | More accurate depth, higher GPU load |
| `ocr_every_n_frames` | `10` | Less CPU/GPU OCR load | More frequent OCR, higher compute cost |
| `face_every_n_frames` | `5` | Less dlib CPU load | More frequent face ID, lower latency |
| `fps_target` | `10.0` | Stricter degradation trigger | Looser degradation trigger |
| `frame_width` / `frame_height` | `640` / `480` | Lower res → higher FPS | Higher res → better detection, lower FPS |
| `yolo_conf_thresh` | `0.4` | Fewer detections → faster payload assembly | More detections, more compute |
| `ocr_conf_thresh` | `0.7` | Fewer OCR results in payload | More OCR results (incl. potential garbage) |
| `face_tolerance` | `0.5` | More false positives (too low = strict) | More false negatives (too high = lenient) |

---

## 7. Resolution Trade-offs

| Resolution | Expected YOLO FPS | Detection Quality |
|---|---|---|
| 320×240 | ~120+ | Poor — misses small/distant objects |
| **640×480** | **50–70** (YOLO only) | **Good — recommended default** |
| 1280×720 | ~25–35 | Better at distance, slower |
| 1920×1080 | ~10–15 | Best quality, near target FPS limit |

MiDaS and EasyOCR accept the full frame; resolution directly impacts their inference time too.

---

## 8. Threading Overhead

Face recognition (dlib) and OCR (EasyOCR) run in a `ThreadPoolExecutor(max_workers=2)`. The overhead of submitting and retrieving futures is ~0.2–1 ms per frame — negligible compared to model inference time.

If a worker has not completed by the time the main loop reaches payload assembly:
- The **previous cached result** is used for that module
- A timeout warning is added to `errors[]`
- The worker result is discarded (not queued for the next frame)

This prevents cumulative lag from building up across frames.

---

## 9. Profiling Guidance

To measure per-module timing, set `PIPELINE_PROFILE=1` before running:

```powershell
$env:PIPELINE_PROFILE = "1"
python scripts/run_pipeline.py --no-cv-window --sink stdout
```

This will print per-frame timing to stderr in the format:
```
[PROFILE] frame=4821  yolo=12ms  depth=28ms  face=41ms  ocr=--  payload=1ms  total=82ms
```

`--` indicates the module was skipped this frame (interleaving). Use this to identify which model is the per-frame bottleneck.

---

## 10. CPU-Only Mode

If you need to run without a GPU (e.g., on a different machine for testing):

```powershell
python scripts/run_pipeline.py --device cpu
```

Expected performance in CPU-only mode on a modern laptop CPU (approximate):

| Configuration | Expected FPS |
|---|---|
| YOLO only | 8–15 |
| YOLO + MiDaS (every 3rd frame) | 4–8 |
| Full pipeline | 2–5 |

In CPU mode, `"degraded"` status is the normal operating state. The interleaving schedule still applies and provides the best possible FPS under the constraint.
