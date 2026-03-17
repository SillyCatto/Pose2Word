"""
Pipeline configuration — all tunable parameters in one place.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    """All preprocessing pipeline parameters."""

    # --- Paths ---
    dataset_dir: Path = Path("dataset/raw_video_data")
    labels_csv: Path = Path("dataset/raw_video_data/labels.csv")
    output_dir: Path = Path("outputs/preprocessed")
    normalized_video_dir: Path = Path("outputs/_normalized_videos")

    # --- Step 1: Video Normalization (ffmpeg) ---
    target_fps: int = 30
    crf_quality: int = 20
    ffmpeg_preset: str = "medium"

    # --- Step 3: CLAHE Lighting Normalization ---
    clahe_clip_limit: float = 2.0
    clahe_tile_size: tuple = field(default_factory=lambda: (8, 8))

    # --- Step 4: Signer Localization ---
    crop_expansion: float = 1.3  # bbox expansion around detected body
    pose_sample_step: int = 3  # run pose every N frames for bbox stability

    # --- Step 5: Output Resolution ---
    output_size: int = 512  # square output — 512×512, aspect-preserving

    # --- Step 6: Temporal Trimming (dual signal) ---
    # Signal A: wrist landmark visibility < 0.3  → no hand visible
    # Signal B: wrist velocity < vel_threshold   → hand stationary
    # Gate: idle must be sustained >= min_idle_duration seconds to trim
    trim_vel_threshold: float = 0.008  # normalized velocity (0–1 coord space)
    trim_min_idle_duration: float = 0.4  # seconds
    min_active_frames: int = 10  # minimum frames to keep after trimming

    # --- Device (for future GPU-accelerated steps) ---
    device: str = "auto"

    def resolve_device(self) -> str:
        """Determine the best available compute device."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
