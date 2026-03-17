"""
app.preprocessing
=================
Video preprocessing pipeline for sign-language recognition.

Pipeline version: 2.0.0

Steps:
  1. Probe raw FPS
  2. Extract frames (cv2, normalised to target_fps)
  3. CLAHE lighting normalisation (full frame)
  4. Signer analysis — crop bbox + wrist signals
  5. Crop to signer bbox
  6. Resize + pad to output_size × output_size
  7. Temporal trim (dual signal: wrist velocity + hand visibility)
  8. Write output video (ffmpeg stdin pipe)
  9. Upsert global metadata.json

Public API:
    from app.preprocessing import PipelineConfig, PreprocessingPipeline
"""

from .config import PipelineConfig
from .runner import PreprocessingPipeline

__all__ = ["PipelineConfig", "PreprocessingPipeline"]
