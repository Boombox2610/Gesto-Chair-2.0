# Payload Schema — v1.0

**Contract between the CV Pipeline and the NLP Layer.**

This document is the authoritative reference for the JSON payload emitted by `payload.py` on every processed frame. The NLP layer must consume this contract. Changes to field names, types, or semantics require a schema version bump to `v1.1` or higher.

---

## Full Schema Example

```json
{
  "schema_version": "1.0",
  "frame_id": 4821,
  "timestamp": 1723354753.482,
  "status": "ok",
  "fps": 26.5,
  "scene_understanding": [
    {
      "object_id": 17,
      "class": "Person",
      "confidence": 0.91,
      "position": "Center",
      "depth_score": 0.74,
      "distance_confidence": "relative",
      "priority": "standard"
    },
    {
      "object_id": 18,
      "class": "Pothole",
      "confidence": 0.85,
      "position": "Center-Left",
      "depth_score": 0.88,
      "distance_confidence": "relative",
      "priority": "critical"
    }
  ],
  "recognized_faces": [
    {
      "object_id": 17,
      "name": "Alice",
      "match_confidence": "high"
    }
  ],
  "environmental_text": [
    {
      "text": "STOP",
      "confidence": 0.92
    }
  ],
  "errors": []
}
```

---

## Field Reference

### Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `schema_version` | `string` | ✓ | Schema version string. Currently `"1.0"`. |
| `frame_id` | `integer` | ✓ | Monotonically increasing frame counter. Resets to `0` on pipeline restart. |
| `timestamp` | `float` | ✓ | Unix epoch timestamp (seconds + fractional) at frame capture time. |
| `status` | `string` | ✓ | Pipeline health. See **Status Values** below. |
| `fps` | `float` | ✓ | Rolling 30-frame FPS average at time of payload assembly. |
| `scene_understanding` | `array` | ✓ | List of detected + tracked objects. Empty array `[]` if no objects detected. |
| `recognized_faces` | `array` | ✓ | List of face-recognition results. Empty array `[]` if no persons or no enrolled faces. |
| `environmental_text` | `array` | ✓ | List of OCR-detected text regions. Empty array `[]` if OCR not triggered this frame. |
| `errors` | `array[string]` | ✓ | List of non-fatal error messages. Empty array `[]` in healthy operation. |

---

### `status` Values

| Value | Meaning |
|---|---|
| `"ok"` | FPS ≥ `fps_target`. All models running normally. |
| `"degraded"` | FPS < `fps_target` for 5+ consecutive frames. Model interleaving has been automatically increased. |
| `"error"` | A recoverable error occurred this frame (e.g. CUDA OOM, webcam read failure). Details in `errors[]`. |
| `"shutdown"` | Final payload emitted on clean exit (`Ctrl+C` or `q`). The NLP layer should treat this as a session-end signal. |

---

### `scene_understanding[]` Object

| Field | Type | Description |
|---|---|---|
| `object_id` | `integer` | Persistent track ID assigned by ByteTrack. Stable across frames for the same physical object. Negative IDs indicate ByteTrack warm-up (first 1–2 frames). |
| `class` | `string` | Detected class label (e.g. `"Person"`, `"Pothole"`, `"Speed Breaker"`, `"Stop Sign"`, `"Glass Door"`). |
| `confidence` | `float` | Detection confidence. Range `[0.0, 1.0]`. Only objects ≥ `yolo_conf_thresh` (default `0.4`) are included. |
| `position` | `string` | Horizontal spatial zone. One of: `"Left"`, `"Center-Left"`, `"Center"`, `"Center-Right"`, `"Right"`. Based on bbox center-x. |
| `depth_score` | `float` | Relative depth derived from MiDaS. Range `[0.0, 1.0]`. **1.0 = closest to camera. 0.0 = farthest.** See **Depth Score Semantics** below. |
| `distance_confidence` | `string` | Always `"relative"` in v1.0. Documents that `depth_score` is not a calibrated metric distance. |
| `priority` | `string` | Actionable semantic severity tag for the downstream NLP layer: `"critical"` (immediate hazard), `"standard"` (active agent / traffic control), or `"contextual"` (surrounding scene infrastructure). |

---

### `recognized_faces[]` Object

| Field | Type | Description |
|---|---|---|
| `object_id` | `integer` | **Cross-reference to `scene_understanding[].object_id`** of the enclosing `"Person"` detection. Allows the NLP layer to join: "Alice is in the Center at depth 0.74". |
| `name` | `string` | Enrolled face name, as registered via `scripts/enroll_face.py`. |
| `match_confidence` | `string` | `"high"` (dlib distance < 0.4) or `"low"` (dlib distance 0.4–`face_tolerance`). |

> **Unknown faces are omitted** from `recognized_faces` entirely. An absence of a `recognized_faces` entry for a given `object_id` means either the face was not recognised, or face recognition was not triggered this frame (frame-skip or YOLO gate). The NLP layer must not infer "this is an unknown person" from absence alone.

---

### `environmental_text[]` Object

| Field | Type | Description |
|---|---|---|
| `text` | `string` | OCR-extracted text string. Uppercased by EasyOCR for Latin text. |
| `confidence` | `float` | EasyOCR confidence score. Range `[0.0, 1.0]`. Only results ≥ `ocr_conf_thresh` (default `0.7`) are included. |

> **OCR is event-driven.** It fires only when YOLOE detects a text-bearing class (such as `"stop sign"`, `"traffic light"`, `"wet floor sign"`, `"exit sign"`, etc.). When OCR is not triggered, `environmental_text` is an empty array `[]` — it does **not** consume GPU compute or repeat previous frame results.

---

## Depth Score Semantics

> [!IMPORTANT]
> `depth_score` is **NOT a metric distance in metres**. MiDaS outputs a *relative* inverse-depth map. The `depth_score` is a per-frame, per-scene normalised value with no fixed correspondence to real-world distance.

**Interpretation guide:**
- `depth_score = 1.0` → object is the closest detectable thing in the scene this frame
- `depth_score = 0.0` → object is the farthest detectable thing in the scene this frame
- Scores are comparable **within a single frame** (object A at 0.8 is closer than object B at 0.4)
- Scores are **not reliably comparable across frames** (a score of 0.6 in frame 100 does not mean the same distance as 0.6 in frame 200, because MiDaS normalises each frame independently)

**Why not metric metres?**
True metric depth from a monocular camera requires known reference distances (e.g. known object size, ground plane homography, or stereo calibration). This is an open problem for v1. The `distance_confidence: "relative"` field documents this limitation explicitly in every payload.

**Original field name:** The original project spec used `distance_meters`. This was renamed to `depth_score` before any NLP consumer code was written, to prevent the NLP layer from treating relative scores as metric distances. Migration: `distance_meters` → `depth_score`.

---

## `errors[]` Format

Each element is a plain string describing the non-fatal error. Examples:

```json
"errors": [
  "CUDA OOM on frame 4821 — fell back to CPU for depth inference",
  "Face recognition worker timed out on frame 4821 — using cached result"
]
```

Fatal errors cause a final payload with `status: "error"` and then pipeline exit.

---

## Schema Versioning Rules

| Rule | Detail |
|---|---|
| **Version bump required** | Any change to a field name, removal of a field, or change to a field's type |
| **Version bump optional** | Adding a new optional field (backward-compatible) |
| **Bump format** | Increment minor (`1.0` → `1.1`) for additive changes; major (`1.0` → `2.0`) for breaking changes |
| **Version field** | `schema_version` is always the first field in the emitted JSON |
| **NLP consumer** | Must check `schema_version` before parsing. Reject payloads with unknown major versions. |

---

## Invariants the NLP Layer Can Rely On

1. Every payload always contains all 8 top-level keys — no optional top-level fields.
2. `scene_understanding`, `recognized_faces`, `environmental_text`, and `errors` are always arrays (never `null`).
3. `recognized_faces[].object_id` always matches a valid `object_id` in `scene_understanding[]` from the same payload.
4. `frame_id` is strictly monotonically increasing within a session.
5. The final payload of every session always has `status: "shutdown"`.
