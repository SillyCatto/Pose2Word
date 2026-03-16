"""
Inference helpers for upload-and-predict sign recognition.

This module keeps the prediction path close to the existing preprocessing
pipeline so uploaded videos are converted to the same 258-dim keypoint format
used by the classifier.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.preprocessing.config import PipelineConfig
from app.preprocessing.denoise import denoise_frames
from app.preprocessing.frame_extraction import extract_frames, trim_idle_segments
from app.preprocessing.keypoint_extraction import extract_keypoints
from app.preprocessing.sequence_normalizer import normalize_sequence_length
from app.preprocessing.signer_crop import compute_signer_bbox, crop_and_resize
from app.preprocessing.video_normalizer import normalize_video

from model.sign_classifier import create_model


DEFAULT_CLASSES = [
    "brother",
    "call",
    "drink",
    "go",
    "help",
    "man",
    "mother",
    "no",
    "short",
    "tall",
    "what",
    "who",
    "why",
    "woman",
    "yes",
]


@dataclass
class InferenceResult:
    predicted_label: str
    predicted_index: int
    confidence: float
    top_k: list[dict]
    is_confident: bool
    metadata: dict


def resolve_class_names(
    checkpoint_config: dict | None = None,
    manual_labels: str | None = None,
) -> list[str]:
    """Resolve class names from manual input, training data, or defaults."""
    if manual_labels:
        labels = [label.strip() for label in manual_labels.split(",") if label.strip()]
        if labels:
            return labels

    checkpoint_config = checkpoint_config or {}
    landmarks_dir = checkpoint_config.get("landmarks_dir")
    if landmarks_dir:
        class_dirs = [path for path in Path(landmarks_dir).iterdir() if path.is_dir()] if Path(landmarks_dir).exists() else []
        class_names = sorted(path.name for path in class_dirs)
        if class_names:
            return class_names

    num_classes = int(checkpoint_config.get("num_classes", len(DEFAULT_CLASSES)))
    if num_classes == len(DEFAULT_CLASSES):
        return DEFAULT_CLASSES.copy()

    return [f"class_{index}" for index in range(num_classes)]


def load_checkpoint(checkpoint_path: str | Path, device: str | None = None) -> tuple[torch.nn.Module, dict, list[str], str]:
    """Load a saved checkpoint and rebuild the matching model."""
    resolved_device = _resolve_device(device)
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)
    config = checkpoint.get("config", {})

    model_type = config.get("model_type", "lstm")
    num_classes = int(config.get("num_classes", len(DEFAULT_CLASSES)))
    model_kwargs = _extract_model_kwargs(model_type, config)

    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        device=resolved_device,
        **model_kwargs,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = resolve_class_names(config)
    if len(class_names) != num_classes:
        class_names = [f"class_{index}" for index in range(num_classes)]

    return model, config, class_names, resolved_device


def build_inference_features(
    video_path: str | Path,
    cfg: PipelineConfig | None = None,
) -> tuple[np.ndarray, dict]:
    """Run one uploaded video through the preprocessing path used for inference."""
    cfg = cfg or PipelineConfig()
    video_path = Path(video_path)

    with tempfile.TemporaryDirectory(prefix="pose2word_predict_") as tmp_dir:
        normalized_path = Path(tmp_dir) / video_path.name
        ok = normalize_video(video_path, normalized_path, cfg)
        if not ok:
            raise RuntimeError("Video normalization failed. Check ffmpeg availability.")

        frames = extract_frames(normalized_path)
        original_count = len(frames)
        if original_count == 0:
            raise RuntimeError("No frames could be decoded from the uploaded video.")

        frames, trim_start, trim_end = trim_idle_segments(frames, cfg)
        trimmed_count = len(frames)
        if trimmed_count == 0:
            raise RuntimeError("No active signing segment remained after trimming.")

        bbox = compute_signer_bbox(frames, cfg)
        frames = crop_and_resize(frames, bbox, cfg.crop_short_side)
        crop_h, crop_w = frames[0].shape[:2]

        frames = denoise_frames(frames, cfg)
        keypoints, kp_confidence = extract_keypoints(frames, cfg)
        data = normalize_sequence_length(
            frames,
            keypoints,
            flow_vectors=None,
            flow_magnitudes=None,
            masks=None,
            cfg=cfg,
        )

    metadata = {
        "original_frame_count": original_count,
        "trimmed_frame_count": trimmed_count,
        "trim_range": [int(trim_start), int(trim_end)],
        "crop_bbox": [int(value) for value in bbox],
        "crop_size": [int(crop_h), int(crop_w)],
        "keypoint_confidence_mean": round(float(kp_confidence), 4),
        "target_sequence_length": int(cfg.target_sequence_length),
        "target_fps": int(cfg.target_fps),
    }
    return data["keypoints"].astype(np.float32), metadata


def predict_video(
    video_path: str | Path,
    checkpoint_path: str | Path,
    *,
    manual_labels: str | None = None,
    confidence_threshold: float = 0.55,
    cfg: PipelineConfig | None = None,
    device: str | None = None,
    top_k: int = 3,
) -> InferenceResult:
    """Preprocess a video, run inference, and return ranked predictions."""
    model, checkpoint_config, class_names, resolved_device = load_checkpoint(
        checkpoint_path, device=device
    )
    if manual_labels:
        class_names = resolve_class_names(checkpoint_config, manual_labels=manual_labels)

    keypoints, metadata = build_inference_features(video_path, cfg=cfg)

    landmarks = torch.from_numpy(keypoints).unsqueeze(0).to(resolved_device)
    flow = torch.zeros((1, landmarks.shape[1] - 1, 32), dtype=torch.float32).to(
        resolved_device
    )

    with torch.no_grad():
        logits = model(landmarks, flow)
        probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    top_indices = np.argsort(probabilities)[::-1][: max(1, top_k)]
    ranked = [
        {
            "label": class_names[int(index)] if int(index) < len(class_names) else f"class_{int(index)}",
            "index": int(index),
            "confidence": float(probabilities[int(index)]),
        }
        for index in top_indices
    ]

    best = ranked[0]
    metadata["model_type"] = checkpoint_config.get("model_type", "lstm")
    metadata["device"] = resolved_device

    return InferenceResult(
        predicted_label=best["label"],
        predicted_index=best["index"],
        confidence=best["confidence"],
        top_k=ranked,
        is_confident=best["confidence"] >= confidence_threshold,
        metadata=metadata,
    )


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _extract_model_kwargs(model_type: str, config: dict) -> dict:
    if model_type == "lstm":
        return {
            "hidden_dim": int(config.get("hidden_dim", 256)),
        }
    if model_type == "transformer":
        return {
            "d_model": int(config.get("d_model", 256)),
            "nhead": int(config.get("nhead", 8)),
            "num_layers": int(config.get("num_layers", 4)),
            "dim_feedforward": int(config.get("dim_feedforward", 1024)),
            "dropout": float(config.get("dropout", 0.1)),
        }
    if model_type == "hybrid":
        return {
            "hidden_dim": int(config.get("hidden_dim", 256)),
            "num_lstm_layers": int(config.get("num_lstm_layers", 2)),
            "num_attention_heads": int(config.get("num_attention_heads", 4)),
            "dropout": float(config.get("dropout", 0.3)),
        }
    return {}