"""
Steps 5 & 7 — Aspect-preserving resize and H.264 video writing.

Resize + letterbox:
    Scales the frame so the longer dimension equals output_size,
    then pads the shorter dimension with black bars to make it square.
    This avoids squishing the signer's body proportions.

Video writing:
    Pipes raw RGB frames directly to ffmpeg via stdin, encoding to
    H.264 with configurable CRF and preset.  This avoids writing
    intermediate image files to disk.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .video_normalizer import _get_ffmpeg


def resize_with_padding(frame: np.ndarray, target: int) -> np.ndarray:
    """
    Scale frame so that max(H, W) == target, then centre-pad to target×target.

    Args:
        frame:  RGB uint8 numpy array.
        target: Output size in pixels (both width and height).

    Returns:
        RGB uint8 numpy array of shape (target, target, 3).
    """
    h, w = frame.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    y_off = (target - new_h) // 2
    x_off = (target - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def write_video(
    frames: list[np.ndarray],
    output_path: Path,
    fps: int,
    cfg: PipelineConfig,
) -> tuple[bool, str | None]:
    """
    Encode a list of RGB uint8 frames to an H.264 .mp4 file via ffmpeg stdin.

    Frame data is streamed as raw RGB bytes — no temporary image files.

    Args:
        frames:       List of (H, W, 3) uint8 RGB arrays (all same size).
        output_path:  Destination .mp4 path (parent directory created if needed).
        fps:          Output frame rate.
        cfg:          PipelineConfig (crf_quality, ffmpeg_preset).

    Returns:
        Tuple of (success, error_message).
    """
    if not frames:
        return False, "No frames provided for video encoding."

    output_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    ffmpeg = _get_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-crf",
        str(cfg.crf_quality),
        "-preset",
        cfg.ffmpeg_preset,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]

    proc: subprocess.Popen | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if proc.stdin is None:
            return False, "ffmpeg stdin pipe was not created."

        for frame_rgb in frames:
            proc.stdin.write(frame_rgb.tobytes())

        proc.stdin.close()
        proc.stdin = None
        proc.wait(timeout=120)

        stderr_text = ""
        if proc.stderr is not None:
            stderr_text = proc.stderr.read().decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            if stderr_text:
                return False, stderr_text
            return False, f"ffmpeg exited with code {proc.returncode}."

        if not output_path.exists():
            return (
                False,
                f"ffmpeg exited successfully but did not create {output_path}.",
            )

        if output_path.stat().st_size <= 0:
            return False, f"ffmpeg created an empty output file at {output_path}."

        return True, None
    except BrokenPipeError:
        stderr_text = _read_stderr(proc)
        if stderr_text:
            return False, stderr_text
        return False, "ffmpeg closed stdin before all frames were written."
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        stderr_text = _read_stderr(proc)
        if stderr_text:
            return False, f"ffmpeg timed out while finishing output: {stderr_text}"
        return False, "ffmpeg timed out while finishing output."
    except Exception as exc:
        return False, str(exc)
    finally:
        if proc is not None and proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc is not None and proc.stderr is not None:
            try:
                proc.stderr.close()
            except Exception:
                pass


def _read_stderr(proc: subprocess.Popen | None) -> str:
    if proc is None or proc.stderr is None:
        return ""
    try:
        return proc.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""
