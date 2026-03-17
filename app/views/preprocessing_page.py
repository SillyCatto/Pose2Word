"""
Video Preprocessing Page — integrated with the preprocessing pipeline v2.0.0.

Provides a Streamlit UI for:
  - Configuring all pipeline parameters
  - Processing a single video or the full dataset
  - Viewing progress and quality reports
  - Previewing preprocessed .mp4 outputs
"""

import csv
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from app.preprocessing import PreprocessingPipeline
from app.preprocessing.config import PipelineConfig
from app.preprocessing.quality_checks import generate_dataset_report
from app.preprocessing.storage import sample_exists


# ── Session state keys ───────────────────────────────────────────────────────
_PREFIX = "pp_"


def _key(name: str) -> str:
    return f"{_PREFIX}{name}"


def _init():
    defaults = {
        _key("mode"): "single",
        _key("dataset_dir"): "dataset/raw_video_data",
        _key("output_dir"): "outputs/preprocessed",
        _key("preview_video_path"): None,
        _key("preview_metadata"): None,
        _key("batch_results"): None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Main render ──────────────────────────────────────────────────────────────


def render():
    _init()

    st.caption(
        "Run the full preprocessing pipeline on WLASL sign language videos — "
        "normalize, CLAHE, crop, temporally trim, and write 512×512 video."
    )

    # --- Mode selector ---
    mode = st.radio(
        "Processing Mode",
        ["🎬 Single Video", "📦 Batch (Full Dataset)", "📊 Quality Report"],
        horizontal=True,
        key=_key("mode_radio"),
    )

    st.divider()

    # --- Shared config sidebar ---
    cfg = _render_config()

    if mode == "🎬 Single Video":
        _render_single_video(cfg)
    elif mode == "📦 Batch (Full Dataset)":
        _render_batch(cfg)
    else:
        _render_report(cfg)


# ── Configuration panel ──────────────────────────────────────────────────────


def _render_config() -> PipelineConfig:
    """Render pipeline configuration controls and return a PipelineConfig."""
    with st.expander("⚙️ Pipeline Configuration", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Video Normalization**")
            target_fps = st.number_input("Target FPS", 15, 60, 30, key=_key("fps"))
            crf = st.slider(
                "CRF Quality",
                18,
                28,
                20,
                help="Lower = better quality, larger files",
                key=_key("crf"),
            )

            st.markdown("**CLAHE Lighting Normalization**")
            clahe_clip = st.slider(
                "CLAHE clip limit",
                0.5,
                8.0,
                2.0,
                0.5,
                help="Controls contrast enhancement strength. 2.0 is a sensible default.",
                key=_key("clahe_clip"),
            )

            st.markdown("**Output Resolution**")
            output_size = st.selectbox(
                "Output size (px)",
                [256, 384, 512],
                index=2,
                help="Square output resolution. 512 is recommended.",
                key=_key("output_size"),
            )

        with col2:
            st.markdown("**Signer Crop**")
            crop_exp = st.slider(
                "Bbox expansion",
                1.0,
                2.0,
                1.3,
                0.05,
                help="1.3 = 30% padding around detected body",
                key=_key("crop_exp"),
            )
            crop_cushion_px = st.slider(
                "Safety cushion (px)",
                0,
                64,
                18,
                1,
                help="Extra pixel padding on all sides after adaptive margins.",
                key=_key("crop_cushion_px"),
            )
            crop_hand_tip_extension = st.slider(
                "Hand reach extension",
                0.0,
                1.0,
                0.35,
                0.05,
                help=(
                    "Extends wrist->elbow direction to estimate fingertip reach; "
                    "helps keep raised fingers above forehead in frame."
                ),
                key=_key("crop_hand_tip_extension"),
            )

            st.markdown("**Temporal Trimming (dual signal)**")
            trim_vel = st.slider(
                "Wrist velocity threshold",
                0.001,
                0.05,
                0.008,
                0.001,
                help="Frames with wrist velocity < threshold are considered idle",
                key=_key("trim_vel"),
                format="%.3f",
            )
            trim_min_idle = st.slider(
                "Min idle duration (s)",
                0.1,
                1.0,
                0.4,
                0.1,
                help="Idle must last ≥ this many seconds before trimming",
                key=_key("trim_min_idle"),
            )
            trim_min_frames = st.number_input(
                "Min active frames after trim",
                0,
                30,
                10,
                key=_key("trim_min_frames"),
                help="Reject trim and keep original boundaries if too few frames remain",
            )

            st.markdown("**Device**")
            device = st.selectbox(
                "Compute device",
                ["auto", "cuda", "mps", "cpu"],
                key=_key("device"),
            )

        dataset_dir = st.text_input(
            "Dataset directory", "dataset/raw_video_data", key=_key("ds_dir")
        )
        output_dir = st.text_input(
            "Output directory", "outputs/preprocessed", key=_key("out_dir")
        )

    cfg = PipelineConfig(
        dataset_dir=Path(dataset_dir),
        labels_csv=Path(dataset_dir) / "labels.csv",
        output_dir=Path(output_dir),
        normalized_video_dir=Path(output_dir) / "_normalized_videos",
        target_fps=target_fps,
        crf_quality=crf,
        clahe_clip_limit=clahe_clip,
        output_size=output_size,
        crop_expansion=crop_exp,
        crop_cushion_px=int(crop_cushion_px),
        crop_hand_tip_extension=float(crop_hand_tip_extension),
        trim_vel_threshold=trim_vel,
        trim_min_idle_duration=trim_min_idle,
        min_active_frames=trim_min_frames,
        device=device,
    )
    return cfg


# ── Single video mode ────────────────────────────────────────────────────────


def _render_single_video(cfg: PipelineConfig):
    st.subheader("🎬 Process a Single Video")

    input_method = st.radio(
        "Video source",
        ["Upload file", "Select from dataset"],
        horizontal=True,
        key=_key("input_method"),
    )

    video_path: Path | None = None
    label = ""
    video_name = ""

    if input_method == "Upload file":
        uploaded = st.file_uploader(
            "Upload a sign language video",
            type=["mp4", "mov", "avi", "mkv"],
            key=_key("upload"),
        )
        if uploaded:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(uploaded.read())
            tmp.close()
            video_path = Path(tmp.name)
            video_name = uploaded.name
            label = st.text_input(
                "Label (class name)", value="unknown", key=_key("label")
            )
    else:
        # Pick from the dataset's labels.csv
        labels_path = cfg.labels_csv
        if not labels_path.exists():
            st.warning(f"labels.csv not found at `{labels_path}`")
            return

        rows = _read_labels(labels_path)
        if not rows:
            st.warning("labels.csv is empty.")
            return

        options = [f"{r['label']}/{r['video_name']}" for r in rows]
        choice = st.selectbox("Select video", options, key=_key("video_sel"))
        idx = options.index(choice)
        row = rows[idx]
        video_path = cfg.dataset_dir / row["path"]
        label = row["label"]
        video_name = row["video_name"]

        if video_path is None or not video_path.exists():
            st.error(f"File not found: `{video_path}`")
            return

        st.success(f"Selected: **{label}/{video_name}**")

    if video_path is None:
        return

    skip_existing = st.checkbox(
        "Skip if already processed",
        value=True,
        key=_key("skip_existing"),
    )

    # --- Process button ---
    if st.button(
        "▶️ Run Preprocessing Pipeline", type="primary", key=_key("run_single")
    ):
        _run_single_pipeline(cfg, video_path, label, video_name, skip_existing)

    # --- Preview section ---
    if st.session_state[_key("preview_video_path")] is not None:
        _render_preview()


def _run_single_pipeline(
    cfg: PipelineConfig,
    video_path: Path,
    label: str,
    video_name: str,
    skip_existing: bool,
):
    """Execute the pipeline with step-by-step status updates."""

    if skip_existing and sample_exists(cfg.output_dir, label, video_name):
        st.info(
            f"⊘ Already processed: `{label}/{video_name}`. "
            "Uncheck 'Skip if already processed' to reprocess."
        )
        stem = Path(video_name).stem
        output_path = cfg.output_dir / label / f"{stem}.mp4"
        if output_path.exists():
            st.session_state[_key("preview_video_path")] = output_path
        return

    _TOTAL = 9
    progress = st.progress(0.0, text="Starting pipeline…")
    status = st.empty()

    def step_cb(step: int, total: int, msg: str) -> None:
        frac = (step - 1) / total
        progress.progress(frac, text=f"Step {step}/{total} — {msg}")
        status.info(f"Step {step}/{total} — {msg}")

    pipeline = PreprocessingPipeline(cfg)

    try:
        result = pipeline.process_single_video(
            video_path,
            label,
            video_name,
            skip_existing=False,
            step_cb=step_cb,
        )

        progress.progress(1.0, text="Done!")

        if result.get("success"):
            stem = Path(video_name).stem
            output_path = cfg.output_dir / label / f"{stem}.mp4"
            orig = result.get("original_frame_count", "?")
            trim = result.get("trimmed_frame_count", "?")
            status.success(
                f"✅ Saved to `{output_path}`  "
                f"({trim} frames kept from {orig} original)"
            )
            st.session_state[_key("preview_video_path")] = output_path
            st.session_state[_key("preview_metadata")] = result
        else:
            status.error(f"❌ Pipeline failed: {result.get('error')}")

    except Exception as exc:
        progress.progress(1.0, text="Error")
        status.error(f"❌ Pipeline failed: {exc}")
        st.exception(exc)


def _sample_frames_from_video(video_path: Path, n: int = 8) -> list:
    """Read up to n uniformly-spaced frames (RGB) from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(i * total / n) for i in range(min(n, total))]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _render_preview():
    """Display preview of preprocessed output."""
    st.divider()
    st.subheader("🔍 Preview")

    meta = st.session_state[_key("preview_metadata")]
    if meta:
        mcols = st.columns(4)
        mcols[0].metric("Original Frames", meta.get("original_frame_count", "—"))
        mcols[1].metric("After Trim", meta.get("trimmed_frame_count", "—"))
        crop = meta.get("crop_size", ["—", "—"])
        mcols[2].metric("Crop Size", f"{crop[0]}×{crop[1]}")
        out = meta.get("output_size", ["—", "—"])
        mcols[3].metric("Output Size", f"{out[0]}×{out[1]}")

    video_path = st.session_state[_key("preview_video_path")]
    if video_path and Path(video_path).exists():
        st.markdown("**Sampled frames from preprocessed video:**")
        sampled = _sample_frames_from_video(video_path, n=8)
        if sampled:
            # Get total frame count for captions
            cap = cv2.VideoCapture(str(video_path))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            cols = st.columns(len(sampled))
            for i, frame in enumerate(sampled):
                frame_idx = int(i * total_f / len(sampled))
                with cols[i]:
                    st.image(frame, caption=f"#{frame_idx}", use_container_width=True)

    if meta:
        with st.expander("📋 Full Metadata"):
            st.json({k: v for k, v in meta.items() if k != "success"})


# ── Batch mode ───────────────────────────────────────────────────────────────


def _render_batch(cfg: PipelineConfig):
    st.subheader("📦 Batch Process — Full Dataset")

    labels_path = cfg.labels_csv
    if not labels_path.exists():
        st.error(f"labels.csv not found at `{labels_path}`")
        return

    rows = _read_labels(labels_path)
    total = len(rows)

    # Dataset overview
    label_counts: dict[str, int] = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1

    st.markdown(f"**{total} videos** across **{len(label_counts)} classes**")
    with st.expander("Class distribution"):
        for cls, cnt in sorted(label_counts.items()):
            st.text(f"  {cls:15s}  {cnt} videos")

    skip_existing = st.checkbox(
        "Skip already processed",
        value=True,
        key=_key("batch_skip_existing"),
    )

    if st.button("🚀 Start Batch Processing", type="primary", key=_key("run_batch")):
        _run_batch(cfg, rows, skip_existing)

    results = st.session_state.get(_key("batch_results"))
    if results:
        _render_batch_results(results)


def _run_batch(
    cfg: PipelineConfig,
    rows: list[dict],
    skip_existing: bool,
):
    """Process every video in labels.csv with live progress."""
    pipeline = PreprocessingPipeline(cfg)
    total = len(rows)

    progress = st.progress(0.0, text=f"Processing 0/{total}…")
    status_counts: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}
    all_results: list[dict] = []

    for idx, row in enumerate(rows):
        video_name = row["video_name"]
        label = row["label"]
        rel_path = row["path"]
        video_path = cfg.dataset_dir / rel_path

        if not video_path.exists():
            result = {
                "status": "error",
                "video": video_name,
                "error": "file not found",
            }
        else:
            raw = pipeline.process_single_video(
                video_path,
                label,
                video_name,
                skip_existing=skip_existing,
            )
            if raw.get("skipped"):
                result = {"status": "skipped", "video": video_name}
            elif raw.get("success"):
                result = {"status": "ok", "video": video_name, **raw}
            else:
                result = {
                    "status": "error",
                    "video": video_name,
                    "error": raw.get("error", "unknown"),
                }

        status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
        all_results.append(result)

        frac = (idx + 1) / total
        icon = {"ok": "✓", "skipped": "⊘", "error": "✗"}.get(result["status"], "?")
        progress.progress(frac, text=f"{icon} [{idx + 1}/{total}] {label}/{video_name}")

    progress.progress(1.0, text="Batch complete!")

    st.session_state[_key("batch_results")] = {
        "total": total,
        "processed": status_counts.get("ok", 0),
        "skipped": status_counts.get("skipped", 0),
        "errors": status_counts.get("error", 0),
        "results": all_results,
    }


def _render_batch_results(summary: dict):
    st.divider()
    st.subheader("📊 Batch Results")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", summary["total"])
    c2.metric("Processed", summary["processed"])
    c3.metric("Skipped", summary["skipped"])
    c4.metric("Errors", summary["errors"])

    errors = [r for r in summary["results"] if r["status"] == "error"]
    if errors:
        with st.expander(f"⚠️ {len(errors)} errors", expanded=True):
            for e in errors:
                st.text(f"  ✗ {e['video']}: {e.get('error', 'unknown')}")


# ── Quality report mode ──────────────────────────────────────────────────────


def _render_report(cfg: PipelineConfig):
    st.subheader("📊 Quality Report")

    if not cfg.output_dir.exists():
        st.info("No preprocessed data found. Run the pipeline first.")
        return

    if st.button("🔄 Generate Report", key=_key("gen_report")):
        with st.spinner("Scanning preprocessed data…"):
            report = generate_dataset_report(cfg.output_dir)
        st.session_state[_key("report_data")] = report

    report = st.session_state.get(_key("report_data"))
    if not report:
        st.info("Click 'Generate Report' to scan preprocessed data.")
        return

    # report is List[SampleReport]
    total = len(report)
    passed = sum(1 for r in report if r.passed)
    failed_list = [r for r in report if not r.passed]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Samples", total)
    c2.metric("Passed", passed)
    c3.metric("Failed", len(failed_list))

    # Per-class counts
    class_counts: dict[str, int] = {}
    for r in report:
        class_counts[r.label] = class_counts.get(r.label, 0) + 1

    if class_counts:
        st.markdown("**Per-class counts:**")
        ncols = min(5, len(class_counts))
        cols = st.columns(ncols)
        for i, (cls, cnt) in enumerate(sorted(class_counts.items())):
            cols[i % ncols].metric(cls, cnt)

    if failed_list:
        st.markdown("---")
        st.markdown(f"**⚠️ {len(failed_list)} failed samples:**")
        for r in failed_list:
            st.text(f"  ✗ {r.label}/{r.video_name}: {'; '.join(r.warnings)}")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _read_labels(labels_path: Path) -> list[dict]:
    with open(labels_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
