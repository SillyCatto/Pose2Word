"""
Manual Keyframe Selection Page

Upload a video → every frame is shown as a clickable thumbnail.
Click to toggle frames on/off, then save the selection as ground-truth data.

Ground truth is saved as:
  <output_dir>/<video_name>/
      frames/          ← JPEG images of selected frames
      ground_truth.json  ← {video, fps, total_frames, selected_indices: [...]}
      ground_truth.npy   ← numpy int array of selected indices
"""

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core.video_utils import get_frames_from_video, get_video_info
from ._folder_browser import folder_input_with_browse


# ── session-state keys ───────────────────────────────────────────────────────
_ALL_FRAMES = "mk_all_frames"
_SELECTED = "mk_selected_indices"  # set[int]
_VIDEO_NAME = "mk_video_name"
_VIDEO_INFO = "mk_video_info"
_OUTPUT_DIR = "mk_output_dir"


def _init():
    defaults = {
        _ALL_FRAMES: None,
        _SELECTED: set(),
        _VIDEO_NAME: "",
        _VIDEO_INFO: {},
        _OUTPUT_DIR: "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── helpers ──────────────────────────────────────────────────────────────────


def _toggle(idx: int):
    sel: set = st.session_state[_SELECTED]
    if idx in sel:
        sel.discard(idx)
    else:
        sel.add(idx)


def _save_ground_truth(output_dir: str) -> tuple[bool, str]:
    frames: list = st.session_state[_ALL_FRAMES]
    sel_indices = sorted(st.session_state[_SELECTED])
    video_name: str = st.session_state[_VIDEO_NAME]
    info: dict = st.session_state[_VIDEO_INFO]

    if not sel_indices:
        return False, "No frames selected."
    if not output_dir:
        return False, "Please set an output directory."

    output_dir = os.path.expanduser(output_dir)
    video_stem = Path(video_name).stem
    save_root = Path(output_dir) / video_stem
    frames_dir = save_root / "frames"

    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return False, f"Cannot create directory: {e}"

    # Save JPEG frames
    for idx in sel_indices:
        bgr = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frames_dir / f"frame_{idx:05d}.jpg"), bgr)

    # Save JSON manifest
    manifest = {
        "video": video_name,
        "fps": info.get("fps", 0),
        "total_frames": len(frames),
        "selected_count": len(sel_indices),
        "selected_indices": sel_indices,
    }
    with open(save_root / "ground_truth.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Save numpy array
    np.save(str(save_root / "ground_truth.npy"), np.array(sel_indices, dtype=np.int32))

    return True, str(save_root)


# ── main render ──────────────────────────────────────────────────────────────


def render():
    _init()

    st.caption(
        "Upload a video, pick your keyframes manually, and save them as ground-truth data."
    )

    # ── 1. VIDEO UPLOAD ───────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload Video", type=["mp4", "mov", "avi", "mkv"], key="mk_uploader"
    )

    if uploaded is None:
        st.info("Upload a video to get started.")
        return

    # ── 2. LOAD FRAMES (only when video changes) ──────────────────────────────
    load_key = uploaded.name

    if st.session_state.get("mk_load_key") != load_key:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        with st.spinner("Reading video…"):
            info = get_video_info(tmp_path)
            frames = get_frames_from_video(tmp_path)

        os.remove(tmp_path)

        st.session_state[_ALL_FRAMES] = frames
        st.session_state[_SELECTED] = set()
        st.session_state[_VIDEO_NAME] = uploaded.name
        st.session_state[_VIDEO_INFO] = info
        st.session_state["mk_load_key"] = load_key

    frames: list = st.session_state[_ALL_FRAMES]
    info: dict = st.session_state[_VIDEO_INFO]
    selected: set = st.session_state[_SELECTED]

    if not frames:
        st.error("Could not read any frames from this video.")
        return

    # ── 3. VIDEO SUMMARY ─────────────────────────────────────────────────────
    h, w = frames[0].shape[:2]
    st.caption(
        f"📹 **{uploaded.name}**  |  "
        f"native {info['width']}×{info['height']} @ {info['fps']:.1f} fps  →  "
        f"**{len(frames)} frames** loaded @ {w}×{h}"
    )

    # ── 4. QUICK SELECTION HELPERS ────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 2])
    with col_a:
        if st.button("✅ Select All"):
            st.session_state[_SELECTED] = set(range(len(frames)))
            st.rerun()
    with col_b:
        if st.button("❌ Clear All"):
            st.session_state[_SELECTED] = set()
            st.rerun()
    with col_c:
        every_n = st.number_input(
            "Every N frames",
            min_value=2,
            max_value=len(frames),
            value=5,
            step=1,
            key="mk_every_n",
        )
    with col_d:
        if st.button(f"⚡ Auto-select every {int(every_n)} frames"):
            st.session_state[_SELECTED] = set(range(0, len(frames), int(every_n)))
            st.rerun()

    selected_count = len(selected)
    st.markdown(f"**{selected_count} / {len(frames)} frames selected**")

    # ── 5. FRAME GRID ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎞️ Click a frame to toggle selection")
    st.caption("🟩 Green border = selected  |  Click again to deselect")

    COLS = 6
    rows = (len(frames) + COLS - 1) // COLS

    for row in range(rows):
        cols = st.columns(COLS)
        for col_idx, col in enumerate(cols):
            frame_idx = row * COLS + col_idx
            if frame_idx >= len(frames):
                break
            is_selected = frame_idx in selected
            with col:
                # Annotate thumbnail with selection border
                thumb = frames[frame_idx].copy()
                if is_selected:
                    border_color = (50, 205, 50)  # green
                    cv2.rectangle(
                        thumb,
                        (0, 0),
                        (thumb.shape[1] - 1, thumb.shape[0] - 1),
                        border_color,
                        thickness=max(4, thumb.shape[0] // 30),
                    )

                label = f"{'✅ ' if is_selected else ''}#{frame_idx}"
                st.image(thumb, caption=label, use_container_width=True)

                if st.button(
                    "Deselect" if is_selected else "Select",
                    key=f"mk_btn_{frame_idx}",
                    type="primary" if is_selected else "secondary",
                ):
                    _toggle(frame_idx)
                    st.rerun()

    # ── 6. SAVE GROUND TRUTH ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💾 Save Ground Truth")

    if selected_count == 0:
        st.warning("Select at least one frame before saving.")
    else:
        st.success(
            f"**{selected_count} frames** ready to save: {sorted(selected)[:10]}{'…' if selected_count > 10 else ''}"
        )

    folder_input_with_browse(
        "Output directory",
        session_key=_OUTPUT_DIR,
        placeholder="/absolute/path/to/ground_truth_output",
        dialog_title="Select Ground Truth Output Folder",
    )

    if st.button("💾 Save Ground Truth", type="primary", disabled=selected_count == 0):
        ok, msg = _save_ground_truth(st.session_state[_OUTPUT_DIR])
        if ok:
            st.success(f"✅ Saved to `{msg}`")
            st.markdown(f"""
            **Files written:**
            - `{msg}/frames/frame_XXXXX.jpg` × {selected_count}
            - `{msg}/ground_truth.json`
            - `{msg}/ground_truth.npy`
            """)
        else:
            st.error(msg)

    # ── 7. PREVIEW SELECTED FRAMES ────────────────────────────────────────────
    if selected_count > 0:
        with st.expander(
            f"🔍 Preview {selected_count} selected frames", expanded=False
        ):
            prev_cols = st.columns(min(selected_count, 5))
            for i, idx in enumerate(sorted(selected)):
                with prev_cols[i % 5]:
                    st.image(
                        frames[idx], caption=f"Frame #{idx}", use_container_width=True
                    )
