"""
Step 2 — Frame extraction and raw video probing.

Decodes a normalized video into a list of RGB uint8 numpy arrays held
entirely in memory.  A separate fps probe reads the source video's native
frame rate (stored in metadata for reference).
"""

import cv2
import numpy as np
from pathlib import Path


def extract_frames(video_path: Path) -> list[np.ndarray]:
    """
    Decode a video file into a list of RGB uint8 numpy arrays.

    Args:
        video_path: Path to the (normalized) video file.

    Returns:
        List of frames, each shape (H, W, 3) dtype uint8 in RGB order.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def probe_fps(video_path: Path) -> float:
    """
    Return the declared frame rate of a video file.

    Used to record the raw video's native fps in the output metadata.

    Returns:
        fps as float, or 0.0 if unreadable.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps > 0 else 0.0
