"""
Keyframe Extraction Pipeline
=============================
Ported from asl_keyframe_extractor.py CLI tool — exact same algorithms.

Extracts 8–15 semantically meaningful keyframes from word-level ASL videos
using a four-signal hybrid approach:

    Signal 1 — Farneback Optical Flow magnitude (hand-region masked)
    Signal 2 — MediaPipe Wrist Velocity
    Signal 3 — MediaPipe Handshape Change (finger joint delta)
    Signal 4 — Flow Direction Reversal (catches circular/reversing signs)

Keyframe selection detects:
    - Motion transitions  (peaks of fused motion signal)
    - Held poses          (valleys of motion signal)
    - Sandwiched holds    (short holds flanked by two motion bursts)
"""

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# Type alias for optional progress callbacks: fn(fraction: float, message: str)
ProgressCallback = Optional[Callable[[float, str], None]]


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """All tunable parameters for the keyframe extraction pipeline."""

    # Keyframe count bounds
    min_keyframes: int = 8
    max_keyframes: int = 15

    # Signal fusion weights (must sum to 1.0)
    weight_flow_magnitude: float = 0.30
    weight_flow_direction: float = 0.15
    weight_wrist_velocity: float = 0.25
    weight_handshape:      float = 0.30

    # Smoothing sigma applied to the transition signal.
    # Hold signal uses smooth_sigma * 0.5 to preserve short holds.
    smooth_sigma: float = 2.0

    # Peak detection minimum prominence
    peak_prominence: float = 0.04

    # Minimum gap between keyframes (fraction of total frames)
    min_gap_fraction: float = 0.04

    # Sandwiched hold detector sensitivity.
    # Lower = more sensitive (catches subtle holds). Higher = stricter.
    sandwiched_hold_threshold: float = 0.12

    # MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)
    mediapipe_complexity: int = 1

    # Hand bounding-box padding (fraction of frame size)
    hand_bbox_padding: float = 0.08


@dataclass
class PipelineResult:
    """Structured result from a single keyframe extraction run."""
    keyframe_indices: list[int]
    keyframe_times:   list[float]  # seconds
    keyframe_images:  list[np.ndarray]  # BGR frames
    fps:              float
    total_frames:     int
    fused_transition: np.ndarray
    fused_hold:       np.ndarray
    hands_detected:   int          # frames where at least one hand was found
    video_stem:       str = ""


# ─────────────────────────────────────────────
# Video I/O
# ─────────────────────────────────────────────

def load_video(video_path: str) -> tuple[list[np.ndarray], float]:
    """
    Load all frames from a video file.

    Returns (frames_bgr, fps).
    Raises FileNotFoundError if video can't be opened.
    Raises ValueError if video has < 4 frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if len(frames) < 4:
        raise ValueError(f"Video too short: only {len(frames)} frames found.")

    return frames, fps


# ─────────────────────────────────────────────
# Signal 1 & 4 — Farneback Optical Flow
# ─────────────────────────────────────────────

class FlowSignalExtractor:
    """
    CPU-friendly dense optical flow via OpenCV Farneback.

    Produces two per-frame signals:
        flow_magnitude : mean flow magnitude inside hand bbox (motion energy)
        flow_direction : angular velocity between frames (catches reversals)
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

    def _to_gray(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _hand_mask(self, frame_shape: tuple, bbox: Optional[tuple]) -> np.ndarray:
        H, W = frame_shape[:2]
        if bbox is None:
            return np.ones((H, W), dtype=bool)

        pad = self.config.hand_bbox_padding
        x0 = max(0.0, bbox[0] - pad);  y0 = max(0.0, bbox[1] - pad)
        x1 = min(1.0, bbox[2] + pad);  y1 = min(1.0, bbox[3] + pad)

        mask = np.zeros((H, W), dtype=bool)
        mask[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = True
        return mask

    def extract(
        self,
        frames: list[np.ndarray],
        hand_bboxes: list[Optional[tuple]],
    ) -> tuple[np.ndarray, np.ndarray]:
        N = len(frames)
        flow_magnitude = np.zeros(N)
        flow_direction = np.zeros(N)
        prev_angle = None

        for i in range(1, N):
            g1 = self._to_gray(frames[i - 1])
            g2 = self._to_gray(frames[i])

            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )

            u, v = flow[..., 0], flow[..., 1]
            bbox = hand_bboxes[i] or hand_bboxes[i - 1]
            mask = self._hand_mask(frames[i].shape, bbox)

            # Signal 1 — magnitude
            mag = np.sqrt(u ** 2 + v ** 2)
            flow_magnitude[i] = mag[mask].mean() if mask.any() else mag.mean()

            # Signal 4 — direction reversal
            mu = u[mask].mean() if mask.any() else u.mean()
            mv = v[mask].mean() if mask.any() else v.mean()
            angle = float(np.arctan2(mv, mu))

            if prev_angle is not None:
                delta = abs(angle - prev_angle)
                if delta > np.pi:
                    delta = 2 * np.pi - delta
                flow_direction[i] = delta
            prev_angle = angle

        return flow_magnitude, flow_direction


# ─────────────────────────────────────────────
# Signal 2 & 3 — MediaPipe Landmark Signals
# ─────────────────────────────────────────────

class LandmarkSignalExtractor:
    """
    Runs MediaPipe Holistic on every frame and extracts:
        wrist_velocity  : L2 displacement of wrist landmarks between frames
        handshape_score : L2 displacement of finger-joint landmarks
        hand_bboxes     : normalized (x0, y0, x1, y1) per frame (or None)
        landmarks_seq   : (N, 126) raw landmark array
    """

    FINGER_ALL = list(range(1, 21))  # all non-wrist hand landmarks (1–20)

    def __init__(self, config: PipelineConfig):
        self.config = config
        # Lazy import to avoid top-level mediapipe crash if not installed
        from mediapipe.python.solutions.holistic import Holistic as MpHolistic
        self.holistic = MpHolistic(
            static_image_mode=False,
            model_complexity=config.mediapipe_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _hand_to_vec(self, hand_lms) -> np.ndarray:
        if hand_lms is None:
            return np.zeros(63)
        return np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark]
        ).flatten()

    def _hand_bbox(self, hand_lms) -> Optional[tuple]:
        if hand_lms is None:
            return None
        xs = [lm.x for lm in hand_lms.landmark]
        ys = [lm.y for lm in hand_lms.landmark]
        return (min(xs), min(ys), max(xs), max(ys))

    def _merge_bboxes(self, b1, b2) -> Optional[tuple]:
        boxes = [b for b in (b1, b2) if b is not None]
        if not boxes:
            return None
        xs = [b[0] for b in boxes] + [b[2] for b in boxes]
        ys = [b[1] for b in boxes] + [b[3] for b in boxes]
        return (min(xs), min(ys), max(xs), max(ys))

    def extract(
        self, frames: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
        N = len(frames)
        landmarks_seq = np.zeros((N, 126))
        hand_bboxes: list[Optional[tuple]] = []

        for i, frame in enumerate(frames):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = self.holistic.process(rgb)

            lh = self._hand_to_vec(res.left_hand_landmarks)
            rh = self._hand_to_vec(res.right_hand_landmarks)
            landmarks_seq[i] = np.concatenate([lh, rh])

            bbox = self._merge_bboxes(
                self._hand_bbox(res.left_hand_landmarks),
                self._hand_bbox(res.right_hand_landmarks),
            )
            hand_bboxes.append(bbox)

        # Signal 2 — wrist velocity
        wrists = landmarks_seq[:, [0, 1, 2, 63, 64, 65]]
        wrist_velocity = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(wrists, axis=0), axis=1)]
        )

        # Signal 3 — handshape change (finger joints, no wrist)
        def finger_indices(offset):
            return [offset + j * 3 + k for j in self.FINGER_ALL for k in range(3)]

        fingers = landmarks_seq[:, finger_indices(0) + finger_indices(63)]
        handshape_score = np.concatenate(
            [[0.0], np.linalg.norm(np.diff(fingers, axis=0), axis=1)]
        )

        return wrist_velocity, handshape_score, hand_bboxes, landmarks_seq

    def close(self):
        self.holistic.close()


# ─────────────────────────────────────────────
# Signal Utilities
# ─────────────────────────────────────────────

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    mn, mx = signal.min(), signal.max()
    return (signal - mn) / (mx - mn + 1e-8)


def smooth_signal(signal: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_filter1d(signal.astype(float), sigma=sigma)


# ─────────────────────────────────────────────
# Signal Fusion
# ─────────────────────────────────────────────

def fuse_signals(
    flow_mag:  np.ndarray,
    flow_dir:  np.ndarray,
    wrist_vel: np.ndarray,
    handshape: np.ndarray,
    config:    PipelineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (fused_transition, fused_hold).

    fused_transition : high = movement / handshape change
    fused_hold       : high = stillness / held pose

    Hold signal uses tighter smoothing (0.5×) to preserve short holds.
    """
    fm = smooth_signal(normalize_signal(flow_mag),  config.smooth_sigma)
    fd = smooth_signal(normalize_signal(flow_dir),  config.smooth_sigma)
    wv = smooth_signal(normalize_signal(wrist_vel), config.smooth_sigma)
    hs = smooth_signal(normalize_signal(handshape), config.smooth_sigma)

    w = config
    fused_transition = (
        w.weight_flow_magnitude * fm +
        w.weight_flow_direction * fd +
        w.weight_wrist_velocity * wv +
        w.weight_handshape      * hs
    )

    # Tighter sigma for hold signal so short holds aren't smoothed away
    hold_sigma = max(1.0, config.smooth_sigma * 0.5)
    wv_tight   = smooth_signal(normalize_signal(wrist_vel), hold_sigma)
    hs_tight   = smooth_signal(normalize_signal(handshape), hold_sigma)
    fused_hold = 1.0 - smooth_signal(
        normalize_signal(0.5 * wv_tight + 0.5 * hs_tight), hold_sigma
    )

    return normalize_signal(fused_transition), normalize_signal(fused_hold)


# ─────────────────────────────────────────────
# Sandwiched Hold Detector
# ─────────────────────────────────────────────

def detect_sandwiched_holds(
    motion_signal:  np.ndarray,
    min_gap:        int,
    drop_threshold: float = 0.12,
) -> list[int]:
    """
    Finds holds sandwiched between two motion peaks.

    For every consecutive peak pair, locates the deepest point between them.
    If it drops enough below the flanking peak average, it's a real hold.
    """
    peaks, _ = find_peaks(motion_signal, distance=min_gap, prominence=0.05)

    sandwiched = []
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]

        if p2 - p1 < 3:
            continue

        segment          = motion_signal[p1: p2 + 1]
        local_min_offset = int(np.argmin(segment))
        local_min_idx    = p1 + local_min_offset

        min_val  = motion_signal[local_min_idx]
        peak_avg = (motion_signal[p1] + motion_signal[p2]) / 2.0

        if peak_avg - min_val >= drop_threshold:
            sandwiched.append(local_min_idx)

    return sandwiched


# ─────────────────────────────────────────────
# Keyframe Selection
# ─────────────────────────────────────────────

def select_keyframes(
    fused_transition: np.ndarray,
    fused_hold:       np.ndarray,
    config:           PipelineConfig,
    wrist_velocity:   Optional[np.ndarray] = None,
    handshape_score:  Optional[np.ndarray] = None,
) -> list[int]:
    """
    Select keyframe indices via four complementary strategies:

        1. Peaks of fused_transition  → motion onsets / handshape transitions
        2. Peaks of fused_hold        → prominent held poses
        3. Sandwiched hold detection  → short holds between motion bursts
        4. Trim / pad to [min_keyframes, max_keyframes]
    """
    N       = len(fused_transition)
    min_gap = max(2, int(N * config.min_gap_fraction))
    target  = (config.min_keyframes + config.max_keyframes) // 2

    selected = {0, N - 1}

    # 1. Transition peaks
    trans_peaks, trans_props = find_peaks(
        fused_transition, distance=min_gap, prominence=config.peak_prominence
    )
    if len(trans_peaks):
        order = np.argsort(trans_props["prominences"])[::-1]
        selected.update(trans_peaks[order][: target // 2])

    # 2. Hold peaks
    hold_peaks, hold_props = find_peaks(
        fused_hold, distance=min_gap, prominence=config.peak_prominence
    )
    if len(hold_peaks):
        order = np.argsort(hold_props["prominences"])[::-1]
        selected.update(hold_peaks[order][: target // 2])

    # 3. Sandwiched holds (uses raw unsmoothed signals)
    if wrist_velocity is not None and handshape_score is not None:
        raw_motion = normalize_signal(
            0.5 * normalize_signal(wrist_velocity) +
            0.5 * normalize_signal(handshape_score)
        )
        sandwiched = detect_sandwiched_holds(
            raw_motion,
            min_gap=min_gap,
            drop_threshold=config.sandwiched_hold_threshold,
        )
        if sandwiched:
            selected.update(sandwiched)

    sorted_idx = sorted(selected)

    # 4a. Trim if above max — drop lowest-scoring middle frames first
    while len(sorted_idx) > config.max_keyframes:
        middle = sorted_idx[1:-1]
        scores = [fused_transition[i] + fused_hold[i] for i in middle]
        worst  = middle[int(np.argmin(scores))]
        sorted_idx.remove(worst)

    # 4b. Pad if below min — insert highest-scoring unselected frame
    while len(sorted_idx) < config.min_keyframes:
        combined  = fused_transition + fused_hold
        remaining = [i for i in range(N) if i not in set(sorted_idx)]
        if not remaining:
            break
        best       = remaining[int(np.argmax([combined[i] for i in remaining]))]
        sorted_idx = sorted(sorted_idx + [best])

    return sorted_idx


# ─────────────────────────────────────────────
# Frame Saving
# ─────────────────────────────────────────────

def save_individual_keyframes(
    frames:     list[np.ndarray],
    indices:    list[int],
    output_dir: str,
) -> list[str]:
    """
    Save each selected keyframe as frame_0.png, frame_1.png, …
    Returns list of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for rank, idx in enumerate(indices):
        out_path = os.path.join(output_dir, f"frame_{rank}.png")
        cv2.imwrite(out_path, frames[idx])
        paths.append(out_path)
    return paths


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    config: Optional[PipelineConfig] = None,
    progress_cb: ProgressCallback = None,
) -> PipelineResult:
    """
    Full ASL keyframe extraction pipeline.

    Parameters
    ----------
    video_path  : path to input video file
    config      : pipeline tuning parameters (defaults used if None)
    progress_cb : optional callback(fraction, message) for UI progress bars

    Returns
    -------
    PipelineResult with keyframe indices, images, scores, and metadata.
    """
    if config is None:
        config = PipelineConfig()

    def _progress(frac: float, msg: str):
        if progress_cb:
            progress_cb(frac, msg)

    stem = Path(video_path).stem

    # Stage 1 — Load video
    _progress(0.0, "Loading video…")
    frames, fps = load_video(video_path)
    N = len(frames)

    # Stage 2 — MediaPipe landmark signals
    _progress(0.10, f"MediaPipe Holistic on {N} frames…")
    lm_extractor = LandmarkSignalExtractor(config)
    wrist_velocity, handshape_score, hand_bboxes, landmarks_seq = \
        lm_extractor.extract(frames)
    lm_extractor.close()

    hands_detected = sum(1 for b in hand_bboxes if b is not None)

    # Stage 3 — Optical flow
    _progress(0.50, f"Farneback flow on {N - 1} frame pairs…")
    flow_extractor = FlowSignalExtractor(config)
    flow_magnitude, flow_direction = flow_extractor.extract(frames, hand_bboxes)

    # Stage 4 — Signal fusion + keyframe selection
    _progress(0.85, "Fusing signals and selecting keyframes…")
    fused_transition, fused_hold = fuse_signals(
        flow_magnitude, flow_direction,
        wrist_velocity, handshape_score,
        config,
    )

    keyframe_indices = select_keyframes(
        fused_transition, fused_hold, config,
        wrist_velocity=wrist_velocity,
        handshape_score=handshape_score,
    )

    keyframe_images = [frames[i] for i in keyframe_indices]

    _progress(1.0, f"Done — {len(keyframe_indices)} keyframes selected")

    return PipelineResult(
        keyframe_indices=keyframe_indices,
        keyframe_times=[round(i / fps, 3) for i in keyframe_indices],
        keyframe_images=keyframe_images,
        fps=fps,
        total_frames=N,
        fused_transition=fused_transition,
        fused_hold=fused_hold,
        hands_detected=hands_detected,
        video_stem=stem,
    )


# ─────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────

def discover_videos(
    dataset_dir: str,
) -> list[tuple[str, str, str]]:
    """
    Walk a dataset directory structured as:
        dataset_dir / label / video_file.mp4

    Returns list of (video_path, label, video_stem) tuples,
    sorted by label then filename.
    """
    root = Path(dataset_dir)
    videos = []

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for video_file in sorted(label_dir.iterdir()):
            if video_file.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"):
                videos.append((str(video_file), label, video_file.stem))

    return videos


def run_batch(
    dataset_dir: str,
    output_root: str,
    config: Optional[PipelineConfig] = None,
    video_cb: Optional[Callable[[int, int, str, str], None]] = None,
) -> list[dict]:
    """
    Process all videos in a dataset directory.

    Output structure:
        output_root / label / video_stem / frame_0.png, frame_1.png, …

    Parameters
    ----------
    dataset_dir : root directory containing label subdirectories
    output_root : root output directory for keyframes
    config      : pipeline parameters
    video_cb    : callback(current_idx, total, label, video_stem) per video

    Returns
    -------
    List of result dicts: {label, video_stem, status, n_keyframes, error}
    """
    if config is None:
        config = PipelineConfig()

    videos = discover_videos(dataset_dir)
    if not videos:
        raise FileNotFoundError(f"No videos found in: {dataset_dir}")

    results = []

    for i, (video_path, label, video_stem) in enumerate(videos):
        if video_cb:
            video_cb(i, len(videos), label, video_stem)

        out_dir = os.path.join(output_root, label, video_stem)

        try:
            result = run_pipeline(video_path, config)
            save_individual_keyframes(
                result.keyframe_images,
                list(range(len(result.keyframe_images))),
                out_dir,
            )
            results.append({
                "label": label,
                "video_stem": video_stem,
                "status": "✓",
                "n_keyframes": len(result.keyframe_indices),
                "error": None,
            })
        except Exception as e:
            results.append({
                "label": label,
                "video_stem": video_stem,
                "status": "✗",
                "n_keyframes": 0,
                "error": str(e),
            })

    return results
