"""
Step 6 — Temporal trimming using a dual-signal approach.

Pre/post-sign rest frames (hands down, not signing) are removed by
combining two complementary signals and applying a duration gate that
distinguishes true idle from mid-sign holds.

────────────────────────────────────────────────────────────────────────
Signal A — Hand visibility (MediaPipe wrist landmark visibility < 0.3)
  Catches: hands completely out of frame (signer hasn't raised hands yet)
  Misses:  hands visible but stationary at rest before sign begins

Signal B — Wrist velocity below threshold
  Catches: hands visible but not moving (pre/post-sign neutral pose)
  Misses:  mid-sign holds (intentional pauses) which also have low velocity

Duration gate — idle must persist for ≥ min_idle_duration seconds
  Separates: pre/post-sign rest (long, 0.5–2 s) from
             mid-sign holds (short, 0.1–0.3 s)
────────────────────────────────────────────────────────────────────────

A frame is IDLE if Signal A OR Signal B is true.
Trimming only occurs when IDLE is *sustained* for ≥ min_idle_frames.
"""

import numpy as np

from .config import PipelineConfig


def compute_wrist_velocity(wrist_positions: np.ndarray) -> np.ndarray:
    """
    Per-frame wrist velocity as the max displacement magnitude across
    both wrists from the previous frame.

    Args:
        wrist_positions: (T, 4) float32 — [lx, ly, rx, ry] in [0, 1] coords.

    Returns:
        velocity: (T,) float32 — displacement magnitude per frame.
                  Frame 0 is always 0.
    """
    T = len(wrist_positions)
    velocity = np.zeros(T, dtype=np.float32)
    for i in range(1, T):
        dl = float(np.linalg.norm(wrist_positions[i, :2] - wrist_positions[i - 1, :2]))
        dr = float(np.linalg.norm(wrist_positions[i, 2:] - wrist_positions[i - 1, 2:]))
        velocity[i] = max(dl, dr)
    return velocity


def detect_idle_boundaries(
    wrist_velocities: np.ndarray,
    hand_in_frame: np.ndarray,
    fps: float,
    cfg: PipelineConfig,
) -> tuple[int, int]:
    """
    Find the inclusive [start, end] frame range of active signing.

    Algorithm:
        1. Build per-frame idle boolean (Signal A OR Signal B).
        2. Walk forward from frame 0, skipping idle frames.
           Accept the first position that has ≥ 3 consecutive active frames
           (prevents reacting to a single fluke frame).
        3. Mirror the walk from the tail.
        4. Apply a 2-frame look-back/look-ahead buffer to avoid clipping
           the natural onset/offset ramp.
        5. Safety: if result is shorter than min_active_frames, keep everything.

    Args:
        wrist_velocities: (T,) per-frame wrist velocity from compute_wrist_velocity.
        hand_in_frame:    (T,) bool — True when wrist visible.
        fps:              Frame rate (used to convert duration → frames).
        cfg:              PipelineConfig.

    Returns:
        (start, end) inclusive indices.
    """
    N = len(wrist_velocities)
    min_idle_frames = max(1, int(cfg.trim_min_idle_duration * fps))

    # Build idle boolean
    idle = np.zeros(N, dtype=bool)
    for i in range(N):
        no_hand = not bool(hand_in_frame[i])  # Signal A
        low_vel = float(wrist_velocities[i]) < cfg.trim_vel_threshold  # Signal B
        idle[i] = no_hand or low_vel

    # ── Find start: skip sustained idle at head ──────────────────────────
    start = 0
    i = 0
    while i < N:
        if not idle[i]:
            # Require ≥ 3 consecutive active frames to confirm onset
            active_run = sum(1 for j in range(i, min(i + 5, N)) if not idle[j])
            if active_run >= 3:
                start = max(0, i - 2)  # 2-frame lookback buffer
                break
        i += 1

    # ── Find end: skip sustained idle at tail ────────────────────────────
    end = N - 1
    i = N - 1
    while i >= 0:
        if not idle[i]:
            active_run = sum(1 for j in range(max(0, i - 4), i + 1) if not idle[j])
            if active_run >= 3:
                end = min(N - 1, i + 2)  # 2-frame lookahead buffer
                break
        i -= 1

    # Safety: if trimming leaves too few frames, keep everything
    if end - start + 1 < cfg.min_active_frames:
        return 0, N - 1

    return int(start), int(end)
