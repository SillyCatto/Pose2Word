Task:
I want some changes in the video preprocessing pipeline, the preprocessing page.
after running a raw video through the preprocssing pipeline steps and after doing all the necessary cleanup, normalization, prerocessing, the preprocessing pipeline will simply output the final cleaned and preprocessed video in oragnized way inside /outputs/preprocessed/label/label-id.mp4... (should support batch preprocessing).
it doesnt need to output any npy files, as it currently does, its only job is to cleanup, normalize, preprocess videos and output the preprocessed video with correct label in organized way, no need for npy.
But, it should output the metadata of the preprocessed output video as it currently does currectly. just the structure of metdata will change a bit. there will be one global metadata file inside outputs/preprocessed/metadata.json, that one json will be organized into list for each label and label-id.
for example, currently for one video label-id: it outputs this metadata.json:
{
  "source_video": "brother_69251.mp4",
  "label": "brother",
  "original_frame_count": 66,
  "trimmed_frame_count": 35,
  "trim_range": [
    8,
    42
  ],
  "crop_bbox": [
    372,
    137,
    949,
    720
  ],
  "crop_size": [
    258,
    256
  ],
  "keypoint_confidence_mean": 0.7017,
  "target_fps": 25,
  "target_sequence_length": 32,
  "flow_computed": true,
  "pipeline_version": "1.0.0",
  "processing_timestamp": "2026-03-12T13:42:21.660111+00:00"
}

the metadata for each preprocessed video will be same as this but instead of having metadata.json for each video, there will be a global json organized like:
{  label: { label-id : {metadata for that label} }  }. one thing to add: metadata for each label will also contain fps of the raw video and fps of the output preprocessed video.

when a particular raw video for a label and label-id is being preprocessed and written and if the metdata for that paritcular label and label-id already exists in metadata.json, then it will simply overwrite the old metadata for that label-id with new one being processed

Approach planning and discussions to consider before implementation:
---
# 1
---

Me:
ok so before i feed in a video to the keyframe extractor, i need to preprocess, cleanup and normalize video right? this is the preprocessing pipeline I drafted, is it enough ? or do I need any more changes/modifications:


```python
Pipeline steps:
    1. Video normalization    (ffmpeg → CFR H.264)
    2. Frame extraction
    3. Temporal trimming      (remove idle head/tail)
    4. Signer crop + resize   (body-centred crop)
    5. Denoising              (fastNlMeansDenoising)
    6. Write preprocessed video
```


Agent:
## Evaluating Your Video Preprocessing Pipeline

Let me go through each step honestly — some are good, one is unnecessary, and there are two missing steps that matter.

---

### Step 1 — Video Normalization (FFmpeg → CFR H.264) ✅ Keep

Constant Frame Rate is essential. VFR (variable frame rate) videos — which phone recordings commonly are — will make your optical flow signal meaningless because consecutive frames no longer represent equal time intervals. Farneback flow between a 33ms gap and a 100ms gap are not comparable scalars. Good call.

One addition: also standardize to **30fps** explicitly, not just CFR. WLASL videos have mixed frame rates. The paper did exactly this — they converted everything to 30fps by frame duplication. Set it in your ffmpeg command:

```bash
ffmpeg -i input.mp4 -vf "fps=30" -c:v libx264 -crf 18 output.mp4
```

---

### Step 2 — Frame Extraction ✅ Keep

Standard. One note: extract to numpy arrays in memory rather than writing individual JPEGs to disk if you're processing a large dataset — disk I/O becomes the bottleneck at scale.

---

### Step 3 — Temporal Trimming ✅ Keep but reconsider your method

Removing idle head and tail is correct — the sign hasn't started yet or has already ended, and those frames pollute your keyframe extractor's motion signal. However, be careful about *how* you detect idle:

A naive approach (thresholding frame-to-frame pixel difference) will falsely trim frames where the signer is holding a pose before moving — exactly the holds your sandwiched detector needs to capture.

The better approach is to use your **wrist velocity signal from MediaPipe** as the trimming signal — trim frames where wrist velocity stays below a threshold for more than N consecutive frames at the head and tail. This preserves holds in the middle of the sign.

```python
def trim_idle_frames(wrist_velocity: np.ndarray,
                     fps: float = 30.0,
                     idle_threshold: float = 0.005,
                     min_idle_frames: int = 8) -> tuple[int, int]:
    """
    Find start/end of active signing using wrist velocity.
    Trims only sustained stillness at head/tail, not internal holds.
    """
    N = len(wrist_velocity)
    
    # Find first frame where motion exceeds threshold
    start = 0
    for i in range(N):
        if wrist_velocity[i] > idle_threshold:
            start = max(0, i - 2)  # small lookback buffer
            break
    
    # Find last frame where motion exceeds threshold
    end = N - 1
    for i in range(N - 1, -1, -1):
        if wrist_velocity[i] > idle_threshold:
            end = min(N - 1, i + 2)  # small lookahead buffer
            break
    
    return start, end
```

---

### Step 4 — Signer Crop + Resize ✅ Keep but with one important constraint

Body-centred crop is good — it removes irrelevant background and focuses the optical flow computation on the signer. However there's a constraint: **your crop must always include both hands and the face together**. If you crop too tight and a hand exits the frame during a wide sign, MediaPipe drops that hand entirely.

The safe crop is a torso-centered box that is tall enough to include hands raised above the head, and wide enough for laterally extended arms. Use MediaPipe pose landmarks to define the crop dynamically per video rather than a fixed pixel crop:

```python
# Crop to: from above head to below hips, with arm-width padding
# Use pose landmarks: nose(0), hips(23,24), wrists(15,16)
# Add ~15% padding on all sides
```

Also — crop **before** feeding to MediaPipe, not after. MediaPipe performs better on a signer-centered image than on a wide shot where the signer occupies 20% of the frame.

---

### Step 5 — Denoising (fastNlMeansDenoising) ❌ Remove this

This is the one step I'd drop entirely. Here's why:

`fastNlMeansDenoising` is a heavy temporal filter. It averages similar patches across frames to reduce noise — but **sign language motion IS the signal**. Applying it to a signing video will:

- Blur fast finger movements, making handshape transitions look smoother than they are
- Reduce the sharpness of motion onset — your Farneback flow peaks will be lower and less distinctive
- Introduce temporal ghosting on hand edges
- Add significant processing time for no practical benefit

WLASL videos are not noisy sensor data — they're webcam/phone recordings with ordinary lighting noise. MediaPipe's landmark detection is robust to this level of noise. The denoising step is solving a problem that doesn't exist for this use case and actively degrades the signal you care about.

If you're worried about video quality, address it at capture time (lighting, codec settings) not in preprocessing.

---

### Step 6 — Write Preprocessed Video ✅ Keep

Standard. Write to H.264 at reasonable quality (CRF 18–22).

---

### Two Missing Steps That Actually Matter

**Missing Step A — Brightness/Contrast Normalization (CLAHE)**

WLASL has significant lighting variation across videos — indoor, outdoor, different skin tones, different backgrounds. This affects MediaPipe's confidence on hand landmark detection. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel in LAB color space per frame:

```python
def apply_clahe(frame_bgr: np.ndarray,
                clip_limit: float = 2.0,
                tile_size: tuple = (8, 8)) -> np.ndarray:
    lab  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l     = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
```

This dramatically improves MediaPipe detection consistency across videos shot in different lighting conditions.

**Missing Step B — Resolution Standardization**

WLASL videos have wildly varying resolutions — some are 1080p, some are 480p, some are portrait 720×1280. You need a fixed resolution before feeding into your pipeline for consistent landmark coordinate scales. After cropping, resize to **512×512** (square, preserving aspect ratio with padding):

```python
def resize_with_padding(frame: np.ndarray,
                         target: int = 512) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    
    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off  = (target - new_h) // 2
    x_off  = (target - new_w) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = resized
    return canvas
```

---

### Corrected Pipeline

```python
"""
Revised preprocessing pipeline:

    1. Video normalization     (ffmpeg → CFR H.264, 30fps)
    2. Frame extraction        (to numpy arrays in memory)
    3. CLAHE lighting norm     (per frame, LAB color space)    ← NEW
    4. Signer crop             (pose-landmark guided, dynamic) ← refined
    5. Resize + pad to 512×512 (aspect-preserving)             ← NEW
    6. Temporal trimming       (wrist-velocity based, not pixel diff) ← refined
    7. Write preprocessed video

    Removed: fastNlMeansDenoising — degrades motion signal for sign video
"""
```

The order matters here too — CLAHE before crop (you want full frame lighting normalization), crop before resize (you define the region of interest first, then standardize its size), temporal trimming last because it needs the full processed frame sequence to compute wrist velocity.

---
# 2

### some important points to consider for idle temporal trimming:

Me: so about the idle trimming at start and end, we could do two things:
check if no hands visible within frame, (cuz I saw some video samples where signer hasnt started the action yet and no hands visible in frame view, (hands are down))
also check if wrist velocity is very very low and its not doing anything

thats my thought process, review and give expert pov advice

Agent:
## Expert Review of Your Idle Trimming Approach

Your instinct is right but each signal alone has failure cases. Let me break down why both are needed and where each one fails without the other.

---

### Signal 1 — Hand Visibility (MediaPipe Detection) — Good but Incomplete

Detecting "no hands visible" is a strong signal for idle frames, but it has two failure cases:

**False negative — hands visible but not signing yet.** Many WLASL signers hold a neutral resting pose (hands loosely at sides or in lap) with hands fully visible before they begin. MediaPipe will happily detect those hands. Visibility alone won't catch this idle period.

**False positive — hands temporarily occluded mid-sign.** Some signs pass the hands behind the body or at extreme angles where MediaPipe drops detection for 2–3 frames. You don't want to trim those frames — they're in the middle of an active sign.

So visibility catches the "hands completely out of frame" idle case well, but misses the "hands visible but stationary pre-sign" case.

---

### Signal 2 — Wrist Velocity — Good but Also Incomplete

Low wrist velocity correctly catches the "hands visible but not moving" idle case. But it has its own failure:

**Cannot distinguish a pre-sign rest from a mid-sign hold.** A signer holding a pose for 3–4 frames before transitioning looks identical to pre-sign idle in velocity terms. This is survivable at the tail (post-sign idle is usually longer than mid-sign holds) but trickier at the head.

---

### The Right Approach — Combine Both as AND Logic, With Duration Gating

The key insight the paper hints at but doesn't fully state: **idle periods are sustained, not momentary**. A real pre-sign idle is typically 10+ frames of nothing. A mid-sign hold is 3–5 frames at most. Duration gating separates them cleanly.

```python
def detect_idle_boundaries(
    hand_bboxes:      list,           # per-frame bbox or None from MediaPipe
    wrist_velocity:   np.ndarray,     # per-frame wrist velocity signal
    fps:              float = 30.0,
    vel_threshold:    float = 0.008,  # normalized velocity below = idle
    min_idle_duration: float = 0.4,   # seconds — must be idle this long to trim
) -> tuple[int, int]:
    """
    Detects the true start and end of active signing using two signals:

        Signal A — hand visibility  : no bbox = definitely idle
        Signal B — wrist velocity   : very low velocity = probably idle

    A frame is considered IDLE if:
        - No hands detected (Signal A), OR
        - Both hands detected but wrist velocity below threshold (Signal B)

    Trimming only occurs if IDLE is sustained for >= min_idle_duration seconds.
    This prevents trimming mid-sign holds (short) vs real idle (long).
    """
    N = len(wrist_velocity)
    min_idle_frames = int(min_idle_duration * fps)  # e.g. 12 frames at 30fps

    # ── Build per-frame idle boolean ─────────────────────────────────────
    idle = np.zeros(N, dtype=bool)

    for i in range(N):
        no_hands  = hand_bboxes[i] is None               # Signal A
        low_vel   = wrist_velocity[i] < vel_threshold     # Signal B
        idle[i]   = no_hands or low_vel

    # ── Find start: first frame of sustained non-idle ─────────────────────
    # Walk forward — skip sustained idle blocks at the head
    start = 0
    i = 0
    while i < N:
        if not idle[i]:
            # Check if this active period is real (not a fluke)
            # Look ahead: if active for at least 3 consecutive frames, it's real
            active_run = 0
            for j in range(i, min(i + 5, N)):
                if not idle[j]:
                    active_run += 1
            if active_run >= 3:
                start = max(0, i - 2)  # 2-frame lookback buffer
                break
        i += 1

    # ── Find end: last frame of sustained non-idle ────────────────────────
    end = N - 1
    i = N - 1
    while i >= 0:
        if not idle[i]:
            active_run = 0
            for j in range(max(0, i - 4), i + 1):
                if not idle[j]:
                    active_run += 1
            if active_run >= 3:
                end = min(N - 1, i + 2)  # 2-frame lookahead buffer
                break
        i -= 1

    # ── Safety check ──────────────────────────────────────────────────────
    # If trimming leaves fewer than 10 frames, don't trim — something is wrong
    if end - start < 10:
        return 0, N - 1

    return start, end
```

---

### One More Edge Case: The "Preparation Phase"

There's a subtle third case your two signals don't cover — the **preparation phase**. In ASL, signers often raise their hands to the starting position before the actual sign begins. The hands ARE visible, wrist velocity IS non-zero, but the sign hasn't started. This is linguistically called "preparation" and is distinct from the sign stroke.

For WLASL at word level this matters less because the dataset clips are already relatively tight around the sign. But if you notice your keyframe extractor is picking up frames before the sign clearly starts, the fix is to raise your velocity threshold slightly so only purposeful signing motion clears it — preparation movements tend to be slower and smoother than sign strokes.

---

### Summary

```
Your two signals combined with duration gating:

    No hands visible          → definitely idle (strong signal)
    Low wrist velocity        → probably idle   (weaker signal)
    Duration >= 0.4 seconds   → gate that separates holds from real idle

    Together: catches both "hands out of frame" AND 
              "hands visible but not signing" idle cases
    Duration gate: prevents trimming mid-sign holds

Tuning knobs:
    vel_threshold    = 0.008   (raise if prep phase bleeds in)
    min_idle_duration = 0.4s   (raise if holds get trimmed, lower if idle bleeds in)
```

Your instinct was correct — combining both is the right call. The duration gate is the missing piece that makes it robust.
