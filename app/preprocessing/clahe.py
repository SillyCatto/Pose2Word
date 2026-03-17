"""
Step 3 — Per-frame CLAHE (Contrast Limited Adaptive Histogram Equalization).

Applied to the full frame BEFORE signer cropping so the entire image
benefits from consistent lighting normalization.  WLASL videos have wide
lighting variation (indoor/outdoor, skin tones, camera types) — CLAHE
dramatically improves MediaPipe pose detection consistency across videos.

Approach:
  1. Convert RGB → BGR → LAB color space.
  2. Apply CLAHE to the L (luminance) channel only — colour is preserved.
  3. Convert back LAB → BGR → RGB.
"""

import cv2
import numpy as np

from .config import PipelineConfig


def apply_clahe(frames: list[np.ndarray], cfg: PipelineConfig) -> list[np.ndarray]:
    """
    Apply per-frame CLAHE to the luminance channel (LAB color space).

    Args:
        frames: List of RGB uint8 numpy arrays.
        cfg:    PipelineConfig (clahe_clip_limit, clahe_tile_size).

    Returns:
        List of CLAHE-normalised RGB uint8 frames.
    """
    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=cfg.clahe_tile_size,
    )
    out: list[np.ndarray] = []
    for frame_rgb in frames:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe.apply(l)
        lab_eq = cv2.merge([l, a, b])
        bgr_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        out.append(cv2.cvtColor(bgr_eq, cv2.COLOR_BGR2RGB))
    return out
