"""
CLI entry point for the preprocessing pipeline.

Usage:
    uv run python -m app.preprocessing [OPTIONS]

Examples:
    # Run entire dataset with default settings
    uv run python -m app.preprocessing

    # Re-process even if outputs already exist
    uv run python -m app.preprocessing --no-skip-existing

    # Custom output resolution and directory
    uv run python -m app.preprocessing --output-size 256 --output-dir outputs/small

    # Single label
    uv run python -m app.preprocessing --label who

    # Print dataset quality report only (no processing)
    uv run python -m app.preprocessing --report
"""

import argparse
import logging
import sys
from pathlib import Path

from .config import PipelineConfig
from .quality_checks import generate_dataset_report
from .runner import PreprocessingPipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m app.preprocessing",
        description="Preprocess sign-language videos (pipeline v2.0.0)",
    )
    p.add_argument(
        "--dataset-dir",
        default="dataset/raw_video_data",
        metavar="DIR",
        help="Root directory of raw video data (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/preprocessed",
        metavar="DIR",
        help="Output root directory (default: %(default)s)",
    )
    p.add_argument(
        "--output-size",
        type=int,
        default=512,
        metavar="PX",
        help="Square output resolution in pixels (default: %(default)s)",
    )
    p.add_argument(
        "--target-fps",
        type=int,
        default=30,
        metavar="FPS",
        help="Output video frame rate (default: %(default)s)",
    )
    p.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        metavar="LIMIT",
        help="CLAHE clip limit (default: %(default)s)",
    )
    p.add_argument(
        "--label",
        default=None,
        metavar="LABEL",
        help="Process only this class label (e.g. 'who')",
    )
    p.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-process videos that already have output files",
    )
    p.add_argument(
        "--report",
        action="store_true",
        help="Print a quality report for already-processed videos and exit",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return p


def _progress(idx: int, total: int, name: str) -> None:
    width = len(str(total))
    print(f"  [{idx:{width}d}/{total}] {name}", flush=True)


def main() -> int:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cfg = PipelineConfig(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        output_size=args.output_size,
        target_fps=args.target_fps,
        clahe_clip_limit=args.clahe_clip,
    )

    pipeline = PreprocessingPipeline(cfg)

    # ── Report mode ──────────────────────────────────────────────────────────
    if args.report:
        reports = generate_dataset_report(cfg.output_dir)
        total = len(reports)
        passed = sum(1 for r in reports if r.passed)
        failed_list = [r for r in reports if not r.passed]
        print(f"\nDataset quality report — {passed}/{total} passed\n")
        if failed_list:
            print("FAILED samples:")
            for r in failed_list:
                warnings = "; ".join(r.warnings)
                print(f"  {r.label}/{r.video_name}: {warnings}")
        trimmed = sum(1 for r in reports if r.trimmed_frames > 0)
        print(f"\nTrimmed {trimmed}/{total} samples.")
        return 0 if not failed_list else 1

    # ── Processing mode ───────────────────────────────────────────────────────
    skip_existing = not args.no_skip_existing

    if args.label:
        # Single label
        label_dir = cfg.dataset_dir / args.label
        video_files = sorted(label_dir.glob("*.mp4"))
        total = len(video_files)
        if total == 0:
            print(f"No .mp4 files found under {label_dir}", file=sys.stderr)
            return 1

        print(f"Processing label '{args.label}' — {total} videos …\n")
        processed = skipped = failed = trimmed_count = 0

        for idx, video_path in enumerate(video_files, start=1):
            _progress(idx, total, video_path.name)
            result = pipeline.process_single_video(
                video_path,
                args.label,
                video_path.name,
                skip_existing=skip_existing,
            )
            if result.get("skipped"):
                skipped += 1
            elif result.get("success"):
                processed += 1
                if result.get("trim_range"):
                    trimmed_count += 1
            else:
                failed += 1
                print(f"    ERROR: {result.get('error')}", file=sys.stderr)

        print(
            f"\nDone. processed={processed} skipped={skipped} "
            f"failed={failed} trimmed={trimmed_count}"
        )
    else:
        # Entire dataset
        print(f"Processing entire dataset under '{cfg.dataset_dir}' …\n")
        summary = pipeline.run(
            skip_existing=skip_existing,
            progress_cb=_progress,
        )
        print(
            f"\nDone. total={summary['total']} "
            f"processed={summary['processed']} "
            f"skipped={summary['skipped']} "
            f"failed={summary['failed']} "
            f"trimmed={summary['trimmed_count']}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
