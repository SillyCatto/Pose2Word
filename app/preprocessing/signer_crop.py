"""
Step 4 — Signer localization, crop, and trim-signal extraction.

Runs MediaPipe Pose on every frame (fast: ~10 ms/frame on CPU for a
pose_landmarker_lite model at typical WLASL resolution).

Two outputs are produced in one MediaPipe pass:

  1. **Adaptive crop bbox** — robust envelope over body anchors + forehead and
      hand trajectories, with asymmetric side/top/bottom margins and pixel
      cushion. This reduces clipping when hands rise above forehead level.

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

import numpy as np
import mediapipe as mp

from .config import PipelineConfig

# Pose landmark indices used for bounding-box estimation:
# nose (0), shoulders (11, 12), elbows (13, 14), wrists (15, 16), hips (23, 24)
_BBOX_LM_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
_NOSE = 0
_LEFT_EYE = 2
_RIGHT_EYE = 5
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_ELBOW = 13
_RIGHT_ELBOW = 14
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


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _estimate_forehead_point(lm, vis_thr: float) -> tuple[float, float] | None:
    """
    Estimate a forehead proxy from visible facial pose points.

    MediaPipe Pose does not expose an explicit forehead landmark, so we use the
    highest visible point among nose/eyes and project slightly upward based on
    head scale.
    """
    candidates = []
    for idx in (_NOSE, _LEFT_EYE, _RIGHT_EYE):
        if lm[idx].visibility > vis_thr:
            candidates.append((lm[idx].x, lm[idx].y))

    if not candidates:
        return None

    xs = [p[0] for p in candidates]
    ys = [p[1] for p in candidates]
    x = float(np.median(xs))
    top_y = float(min(ys))

    if (
        lm[_NOSE].visibility > vis_thr
        and lm[_LEFT_SHOULDER].visibility > vis_thr
        and lm[_RIGHT_SHOULDER].visibility > vis_thr
    ):
        shoulder_mid_y = (lm[_LEFT_SHOULDER].y + lm[_RIGHT_SHOULDER].y) * 0.5
        head_scale = max(0.02, shoulder_mid_y - lm[_NOSE].y)
    else:
        head_scale = 0.05

    y = top_y - 0.35 * head_scale
    return (_clip01(x), _clip01(y))


def _estimate_hand_tip(
    lm,
    wrist_idx: int,
    elbow_idx: int,
    vis_thr: float,
    extension: float,
) -> tuple[float, float] | None:
    """
    Estimate fingertip reach from wrist and elbow direction.

    Pose tracks wrists but not fingers; extending the forearm direction gives a
    lightweight proxy for how far fingers may extend beyond the wrist.
    """
    wrist = lm[wrist_idx]
    if wrist.visibility <= vis_thr:
        return None

    if lm[elbow_idx].visibility > vis_thr:
        elbow = lm[elbow_idx]
        x = wrist.x + extension * (wrist.x - elbow.x)
        y = wrist.y + extension * (wrist.y - elbow.y)
    else:
        x, y = wrist.x, wrist.y

    return (_clip01(x), _clip01(y))


def _legacy_bbox_from_pose(
    bboxes: list[tuple[float, float, float, float]],
    cfg: PipelineConfig,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """Fallback legacy bbox computation if adaptive signals are unavailable."""
    if bboxes:
        arr = np.array(bboxes, dtype=np.float32)
        xn, yn, xx, yx = np.median(arr, axis=0)

        cx, cy = (xn + xx) / 2.0, (yn + yx) / 2.0
        bw = (xx - xn) * cfg.crop_expansion
        bh = (yx - yn) * cfg.crop_expansion

        x1 = max(0, int((cx - bw / 2.0) * w))
        y1 = max(0, int((cy - bh / 2.0) * h))
        x2 = min(w, int((cx + bw / 2.0) * w))
        y2 = min(h, int((cy + bh / 2.0) * h))

        x1 = min(x1, w - 2)
        y1 = min(y1, h - 2)
        x2 = max(x2, x1 + 2)
        y2 = max(y2, y1 + 2)
        return (x1, y1, x2, y2)

    mx, my = int(w * 0.1), int(h * 0.1)
    return (mx, my, w - mx, h - my)


def _compute_adaptive_bbox(
    anchor_points: list[tuple[float, float]],
    motion_points: list[tuple[float, float]],
    forehead_ys: list[float],
    hand_top_ys: list[float],
    cfg: PipelineConfig,
    w: int,
    h: int,
) -> tuple[int, int, int, int] | None:
    """
    Compute robust asymmetric crop margins from trajectory envelopes.

    - anchor_points define the signer's stable body region.
    - motion_points expand side/top/bottom margins according to movement span.
    - forehead vs raised-hand delta adds extra top margin when hands go high.
    """
    if not anchor_points:
        return None

    anchors = np.array(anchor_points, dtype=np.float32)

    q_low = float(np.clip(cfg.crop_quantile_low, 0.0, 0.49))
    q_high = float(np.clip(cfg.crop_quantile_high, 0.51, 1.0))

    core_x1 = float(np.quantile(anchors[:, 0], q_low))
    core_x2 = float(np.quantile(anchors[:, 0], q_high))
    core_y1 = float(np.quantile(anchors[:, 1], q_low))
    core_y2 = float(np.quantile(anchors[:, 1], q_high))

    if core_x2 <= core_x1:
        core_x1, core_x2 = float(np.min(anchors[:, 0])), float(np.max(anchors[:, 0]))
    if core_y2 <= core_y1:
        core_y1, core_y2 = float(np.min(anchors[:, 1])), float(np.max(anchors[:, 1]))

    cx = float(np.quantile(anchors[:, 0], 0.5))
    cy = float(np.quantile(anchors[:, 1], 0.5))

    if motion_points:
        motion = np.array(motion_points, dtype=np.float32)
        mx_low = float(np.quantile(motion[:, 0], 0.05))
        mx_high = float(np.quantile(motion[:, 0], 0.95))
        my_low = float(np.quantile(motion[:, 1], 0.05))
        my_high = float(np.quantile(motion[:, 1], 0.95))
    else:
        mx_low, mx_high = core_x1, core_x2
        my_low, my_high = core_y1, core_y2

    left_reach = max(0.0, cx - mx_low)
    right_reach = max(0.0, mx_high - cx)
    up_reach = max(0.0, cy - my_low)
    down_reach = max(0.0, my_high - cy)

    left_margin = cfg.crop_base_margin_side + cfg.crop_motion_margin_scale * left_reach
    right_margin = (
        cfg.crop_base_margin_side + cfg.crop_motion_margin_scale * right_reach
    )
    top_margin = cfg.crop_base_margin_top + cfg.crop_motion_margin_scale * up_reach
    bottom_margin = (
        cfg.crop_base_margin_bottom + cfg.crop_motion_margin_scale * down_reach
    )

    if forehead_ys and hand_top_ys:
        forehead_ref = float(np.quantile(np.array(forehead_ys, dtype=np.float32), 0.5))
        hand_high = float(np.quantile(np.array(hand_top_ys, dtype=np.float32), 0.1))
        raise_delta = max(0.0, forehead_ref - hand_high)
        top_margin += cfg.crop_top_raise_scale * raise_delta

    px_x = float(cfg.crop_cushion_px) / float(max(1, w))
    px_y = float(cfg.crop_cushion_px) / float(max(1, h))

    left_margin += px_x
    right_margin += px_x
    top_margin += px_y
    bottom_margin += px_y

    x1n = _clip01(core_x1 - left_margin)
    x2n = _clip01(core_x2 + right_margin)
    y1n = _clip01(core_y1 - top_margin)
    y2n = _clip01(core_y2 + bottom_margin)

    # Global expansion keeps compatibility with the previous pipeline control.
    ex = max(1.0, float(cfg.crop_expansion))
    bx = x2n - x1n
    by = y2n - y1n
    ccx = (x1n + x2n) * 0.5
    ccy = (y1n + y2n) * 0.5
    x1n = _clip01(ccx - (bx * ex) * 0.5)
    x2n = _clip01(ccx + (bx * ex) * 0.5)
    y1n = _clip01(ccy - (by * ex) * 0.5)
    y2n = _clip01(ccy + (by * ex) * 0.5)

    # Enforce a minimum crop size so box does not collapse on sparse detections.
    min_frac = float(np.clip(cfg.crop_min_fraction, 0.05, 1.0))
    cur_w = x2n - x1n
    cur_h = y2n - y1n
    if cur_w < min_frac:
        half = min_frac * 0.5
        x1n = _clip01(ccx - half)
        x2n = _clip01(ccx + half)
    if cur_h < min_frac:
        half = min_frac * 0.5
        y1n = _clip01(ccy - half)
        y2n = _clip01(ccy + half)

    x1 = int(np.floor(x1n * w))
    y1 = int(np.floor(y1n * h))
    x2 = int(np.ceil(x2n * w))
    y2 = int(np.ceil(y2n * h))

    x1 = max(0, min(x1, w - 2))
    y1 = max(0, min(y1, h - 2))
    x2 = max(x1 + 2, min(x2, w))
    y2 = max(y1 + 2, min(y2, h))

    return (x1, y1, x2, y2)


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

    vis_thr = float(cfg.crop_visibility_threshold)

    bboxes: list[tuple[float, float, float, float]] = []
    anchor_points: list[tuple[float, float]] = []
    motion_points: list[tuple[float, float]] = []
    forehead_ys: list[float] = []
    hand_top_ys: list[float] = []
    wrist_positions = np.zeros((T, 4), dtype=np.float32)  # [lx, ly, rx, ry]
    wrist_vis = np.zeros((T, 2), dtype=np.float32)  # [left_vis, right_vis]

    with mp.tasks.vision.PoseLandmarker.create_from_options(options) as detector:
        for i, frame_rgb in enumerate(frames):
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_img)

            if not result.pose_landmarks:
                continue

            lm = result.pose_landmarks[0]  # first (and only) person

            # Stable body anchors for robust core crop envelope
            visible_points: list[tuple[float, float]] = []
            xs = []
            ys = []
            for idx in _BBOX_LM_IDS:
                if lm[idx].visibility > vis_thr:
                    x = _clip01(lm[idx].x)
                    y = _clip01(lm[idx].y)
                    xs.append(x)
                    ys.append(y)
                    visible_points.append((x, y))

            anchor_points.extend(visible_points)
            if len(xs) >= 3:
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))

            # Forehead proxy improves top margin decisions.
            forehead = _estimate_forehead_point(lm, vis_thr)
            if forehead is not None:
                anchor_points.append(forehead)
                motion_points.append(forehead)
                forehead_ys.append(float(forehead[1]))

            # Wrist + fingertip-reach proxies drive asymmetric motion margins.
            for wrist_idx, elbow_idx in (
                (_LEFT_WRIST, _LEFT_ELBOW),
                (_RIGHT_WRIST, _RIGHT_ELBOW),
            ):
                wrist = lm[wrist_idx]
                if wrist.visibility > vis_thr:
                    wx = _clip01(wrist.x)
                    wy = _clip01(wrist.y)
                    motion_points.append((wx, wy))
                    hand_top_ys.append(wy)

                hand_tip = _estimate_hand_tip(
                    lm,
                    wrist_idx,
                    elbow_idx,
                    vis_thr,
                    float(cfg.crop_hand_tip_extension),
                )
                if hand_tip is not None:
                    motion_points.append(hand_tip)
                    hand_top_ys.append(float(hand_tip[1]))

            # Wrist data for temporal trimming
            lw = lm[_LEFT_WRIST]
            rw = lm[_RIGHT_WRIST]
            wrist_positions[i] = [lw.x, lw.y, rw.x, rw.y]
            wrist_vis[i] = [lw.visibility, rw.visibility]

    # hand_in_frame: at least one wrist has visibility > 0.3
    hand_in_frame = wrist_vis.max(axis=1) > vis_thr

    bbox = _compute_adaptive_bbox(
        anchor_points,
        motion_points,
        forehead_ys,
        hand_top_ys,
        cfg,
        w,
        h,
    )
    if bbox is None:
        bbox = _legacy_bbox_from_pose(bboxes, cfg, w, h)

    return bbox, wrist_positions, hand_in_frame


def crop_frames(
    frames: list[np.ndarray],
    bbox: tuple[int, int, int, int],
) -> list[np.ndarray]:
    """Crop every frame to (x1, y1, x2, y2) pixel coordinates."""
    x1, y1, x2, y2 = bbox
    return [f[y1:y2, x1:x2] for f in frames]
