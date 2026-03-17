"""
Step 4 — Signer localization, crop, and trim-signal extraction.

Runs MediaPipe Pose on every frame (fast: ~10 ms/frame on CPU for a
pose_landmarker_lite model at typical WLASL resolution).

Two outputs are produced in one MediaPipe pass:

  1. **Stable crop bbox** — median across all detected poses, expanded by
     cfg.crop_expansion, clamped to frame bounds.

  2. **Per-frame trim signals** —
     - wrist_positions (T, 4) float32: [lx, ly, rx, ry] normalised [0, 1]
     - hand_in_frame  (T,)  bool:  True when at least one wrist is visible
       (wrist visibility > 0.3 in the pose model)

These are passed to temporal_trim.py to detect idle head/tail segments.

Running MediaPipe once and extracting both crop and trim signals is more
efficient than a separate crop pass + a separate hand-detection pass.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp

from .config import PipelineConfig

# Pose landmark indices used for bounding-box estimation:
# nose (0), shoulders (11, 12), elbows (13, 14), wrists (15, 16), hips (23, 24)
_BBOX_LM_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
_LEFT_WRIST = 15
_RIGHT_WRIST = 16

_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def _ensure_pose_model() -> str:
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "pose_landmarker_lite.task")
    if not os.path.exists(path):
        print("Downloading pose_landmarker_lite.task …")
        urllib.request.urlretrieve(_POSE_MODEL_URL, path)
    return path


def analyze_signer(
    frames: list[np.ndarray],
    cfg: PipelineConfig,
) -> tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]:
    """
    Run MediaPipe Pose on every frame and return:
        bbox            — (x1, y1, x2, y2) pixel crop region in original frames
        wrist_positions — (T, 4) float32 [lx, ly, rx, ry] normalised coords
        hand_in_frame   — (T,)  bool — True if at least one wrist is visible

    Args:
        frames: RGB uint8 frames (post-CLAHE, pre-crop).
        cfg:    PipelineConfig.
    """
    model_path = _ensure_pose_model()
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_poses=1,
    )

    T = len(frames)
    h, w = frames[0].shape[:2]

    bboxes: list[tuple[float, float, float, float]] = []
    wrist_positions = np.zeros((T, 4), dtype=np.float32)  # [lx, ly, rx, ry]
    wrist_vis = np.zeros((T, 2), dtype=np.float32)  # [left_vis, right_vis]

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for i, frame_rgb in enumerate(frames):
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_img)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks[0]  # first (and only) person

            # Collect coordinates for bbox
            xs = [lm[j].x for j in _BBOX_LM_IDS if lm[j].visibility > 0.3]
            ys = [lm[j].y for j in _BBOX_LM_IDS if lm[j].visibility > 0.3]
            if len(xs) >= 3:
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))

            # Wrist data for temporal trimming
            lw = lm[_LEFT_WRIST]
            rw = lm[_RIGHT_WRIST]
            wrist_positions[i] = [lw.x, lw.y, rw.x, rw.y]
            wrist_vis[i] = [lw.visibility, rw.visibility]

    # hand_in_frame: at least one wrist has visibility > 0.3
    hand_in_frame = wrist_vis.max(axis=1) > 0.3

    # --- Compute stable crop bbox ---
    if bboxes:
        arr = np.array(bboxes)
        xn, yn, xx, yx = np.median(arr, axis=0)

        # Expand around centre
        cx, cy = (xn + xx) / 2, (yn + yx) / 2
        bw = (xx - xn) * cfg.crop_expansion
        bh = (yx - yn) * cfg.crop_expansion

        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))
        bbox = (x1, y1, x2, y2)
    else:
        # Fallback: 80 % centre crop
        mx, my = int(w * 0.1), int(h * 0.1)
        bbox = (mx, my, w - mx, h - my)

    return bbox, wrist_positions, hand_in_frame


def crop_frames(
    frames: list[np.ndarray],
    bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    """Crop every frame to (x1, y1, x2, y2) pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return [f[y1:y2, x1:x2] for f in frames]
