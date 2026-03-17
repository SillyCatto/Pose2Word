"""
Step 8 — Output video storage and global metadata management.

Directory layout:
    outputs/preprocessed/
        metadata.json          ← single global file for all processed videos
        <label>/
            <video_stem>.mp4   ← one preprocessed video per sample

metadata.json structure:
    {
        "<label>": {
            "<video_stem>": { ...per-video metadata dict... }
        }
    }

If an entry for <label>/<video_stem> already exists it is overwritten
(supports re-processing without manual cleanup).
"""

import json
from pathlib import Path

_METADATA_FILE = "metadata.json"


def _load_global_metadata(output_dir: Path) -> dict:
    path = output_dir / _METADATA_FILE
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_global_metadata(output_dir: Path, data: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _METADATA_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def upsert_metadata(
    output_dir: Path,
    label: str,
    video_name: str,
    metadata: dict,
) -> None:
    """
    Insert or overwrite the metadata entry for one video in metadata.json.

    Args:
        output_dir: Root output directory (e.g. outputs/preprocessed/).
        label:      Class label string (e.g. "brother").
        video_name: Original filename (e.g. "brother_69251.mp4").
        metadata:   Dict with all per-video stats.
    """
    global_meta = _load_global_metadata(output_dir)
    stem = Path(video_name).stem

    if label not in global_meta:
        global_meta[label] = {}

    global_meta[label][stem] = metadata
    _save_global_metadata(output_dir, global_meta)


def sample_exists(output_dir: Path, label: str, video_name: str) -> bool:
    """
    Return True if the preprocessed .mp4 for this video already exists.

    Note: we check the video file, not the metadata, so a partial run
    that only wrote metadata but no video is correctly treated as missing.
    """
    stem = Path(video_name).stem
    return (output_dir / label / f"{stem}.mp4").exists()
