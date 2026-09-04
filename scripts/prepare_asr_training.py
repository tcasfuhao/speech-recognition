from __future__ import annotations

# Standard libraries
import argparse
import sys

from pathlib import Path

# Ensure project root is on PYTHONPATH for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Third-party libraries
import pandas as pd
import yaml

# Local imports
from src.data import schema
from src.data.split import (
    SplitRatios,
    build_split_summary,
    save_split_summary,
    split_by_group,
    split_rows,
)
from src.prep.eaf_ingest import (
    DEFAULT_AUDIO_EXTS,
    IngestConfig,
    ingest_eaf_directory,
)


def _project_root() -> Path:
    return PROJECT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument("--start_stage", type=int, default=1)
    ap.add_argument("--stop_stage", type=int, default=None)

    ap.add_argument("--data_root", type=str, help="Root of the external dataset")
    ap.add_argument("--annotations_dir", type=str, default=None)
    ap.add_argument("--audio_root", type=str, default=None)
    ap.add_argument("--clips_dir", type=str, default=None)
    ap.add_argument("--logs_dir", type=str, default="logs/prep")

    ap.add_argument("--include_tier_regex", type=str, default=None)
    ap.add_argument("--exclude_tier_regex", type=str, default=None)
    ap.add_argument("--exclude_annotation_path_regex", type=str, default=None)
    ap.add_argument("--merge_same_time_annotations", action="store_true")
    ap.add_argument("--annotation_joiner", type=str, default="")

    ap.add_argument("--min_dur_ms", type=int, default=200)
    ap.add_argument("--max_dur_ms", type=int, default=30000)

    ap.add_argument("--session_id_from", type=str, default="parent_dir")
    ap.add_argument("--audio_exts", type=str, nargs="*", default=None)

    ap.add_argument("--split_by_col", type=str, default=None)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--allow_row_fallback", action="store_true")

    args = ap.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(args, k, v)

        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config, sort_keys=False))

    root = _project_root()
    if args.data_root is None:
        raise ValueError("data_root must be provided via CLI or config file")

    data_root = Path(args.data_root).expanduser()
    annotations_dir = (
        Path(args.annotations_dir).expanduser()
        if args.annotations_dir
        else data_root
    )
    audio_root = Path(args.audio_root).expanduser() if args.audio_root else data_root
    clips_dir = (
        Path(args.clips_dir).expanduser()
        if args.clips_dir
        else data_root / "processed" / "splits"
    )
    logs_dir = Path(args.logs_dir).expanduser()
    if not logs_dir.is_absolute():
        logs_dir = root / logs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)

    audio_exts = args.audio_exts
    if audio_exts:
        audio_exts = tuple(ext if ext.startswith(".") else f".{ext}" for ext in audio_exts)
    else:
        audio_exts = None

    ###########################################################################
    # Stage 1: Ingest EAF + extract segments
    ###########################################################################
    if args.start_stage <= 1 and (args.stop_stage is None or args.stop_stage >= 1):
        ingest_cfg = IngestConfig(
            annotations_dir=str(annotations_dir),
            audio_root=str(audio_root),
            clips_dir=str(clips_dir),
            include_tier_regex=args.include_tier_regex,
            exclude_tier_regex=args.exclude_tier_regex,
            exclude_annotation_path_regex=args.exclude_annotation_path_regex,
            merge_same_time_annotations=args.merge_same_time_annotations,
            annotation_joiner=args.annotation_joiner,
            min_dur_ms=args.min_dur_ms,
            max_dur_ms=args.max_dur_ms,
            session_id_from=args.session_id_from,
            audio_exts=audio_exts or tuple(DEFAULT_AUDIO_EXTS),
        )

        seg_df, skip_df = ingest_eaf_directory(ingest_cfg)
        seg_df.to_csv(logs_dir / "metadata.csv", index=False)
        skip_df.to_csv(logs_dir / "skip_metadata.csv", index=False)

        print(f"Wrote {len(seg_df)} segments to {logs_dir / 'metadata.csv'}")
        print(f"Wrote {len(skip_df)} skipped rows to {logs_dir / 'skip_metadata.csv'}")

    ###########################################################################
    # Stage 2: X/Y/Z splits
    ###########################################################################
    if args.start_stage <= 2 and (args.stop_stage is None or args.stop_stage >= 2):
        metadata_path = logs_dir / "metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata.csv at {metadata_path}")

        df = pd.read_csv(metadata_path)
        ratios = SplitRatios(
            train=float(args.train_ratio),
            dev=float(args.dev_ratio),
            test=float(args.test_ratio),
        )

        dur_col = schema.pick_col(
            df, schema.DUR_COL_CANDIDATES, required=False
        )

        if args.split_by_col:
            try:
                train_df, dev_df, test_df = split_by_group(
                    df,
                    split_col=args.split_by_col,
                    seed=args.split_seed,
                    ratios=ratios,
                )
            except ValueError:
                if not args.allow_row_fallback:
                    raise
                print(
                    "[WARN] Group split failed, falling back to row-level split."
                )
                train_df, dev_df, test_df = split_rows(
                    df, seed=args.split_seed, ratios=ratios
                )
                args.split_by_col = None
        else:
            train_df, dev_df, test_df = split_rows(
                df, seed=args.split_seed, ratios=ratios
            )

        splits_dir = logs_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(splits_dir / "train.csv", index=False)
        dev_df.to_csv(splits_dir / "dev.csv", index=False)
        test_df.to_csv(splits_dir / "test.csv", index=False)

        summary = build_split_summary(
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            split_col=args.split_by_col,
            dur_col=dur_col,
        )
        summary["split_seed"] = int(args.split_seed)
        summary["ratios"] = {
            "train": ratios.train,
            "dev": ratios.dev,
            "test": ratios.test,
        }

        save_split_summary(str(splits_dir / "split_summary.json"), summary)

        print("Split summary:", summary)


if __name__ == "__main__":
    main()
