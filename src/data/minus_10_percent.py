from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import SplitRatios, build_split_summary, save_split_summary, split_rows
from src.prep.run_paths import new_prep_run_id, validate_prep_run_id


DEFAULT_CONFIG = PROJECT_ROOT / "config/minus_10_percent/cer90.yaml"


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_language_name(language: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", language.lower()).strip("_")
    if not safe:
        raise ValueError("language must contain at least one letter or number")
    return safe


def _load_config(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}")
    required = {"language", "transcription_dir"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Missing required config keys in {config_path}: {', '.join(missing)}")
    return config


def _write_note(
    output_dir: Path,
    *,
    language: str,
    transcription_dir: Path,
    source_path: Path,
    total_rows: int,
    removed_rows: int,
) -> None:
    retained_rows = total_rows - removed_rows
    output_dir.joinpath("NOTE.md").write_text(
        "# CER-filtered training data\n\n"
        f"This directory contains a 10% CER-filtered version of the {language} "
        "imported transcriptions. The noisiest scored rows were removed before "
        "the train/development/test split was made.\n\n"
        f"- Source transcription directory: `{transcription_dir}`\n"
        f"- Source scored transcription file: `{source_path}`\n"
        f"- Original scored rows: {total_rows}\n"
        f"- Removed noisy rows: {removed_rows}\n"
        f"- Retained rows: {retained_rows}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a timestamped 10% CER-filtered transcription manifest set."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--transcription-dir", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--remove-fraction", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--dev-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    config = _load_config(config_path)
    language = args.language or str(config["language"])
    transcription_dir = _resolve_path(args.transcription_dir or config["transcription_dir"])
    scored_filename = str(config.get("scored_filename", "preds_scored.csv"))
    source_path = transcription_dir / scored_filename
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Expected imported scored transcriptions at {source_path}"
        )

    run_id = validate_prep_run_id(args.run_id) if args.run_id else new_prep_run_id()
    configured_output = args.output_dir or config.get("output_dir")
    output_dir = (
        _resolve_path(str(configured_output))
        if configured_output
        else PROJECT_ROOT / "logs/prep" / f"{_safe_language_name(language)}_{run_id}"
    )
    if output_dir.exists():
        raise FileExistsError(f"CER-filter output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    # Every output artifact is deliberately derived from this fresh run directory.
    default_summary_path = output_dir / "cer_filter_summary.json"
    metadata_out = output_dir / "metadata.csv"
    removed_out = output_dir / "removed_noisy_top10.csv"
    splits_dir = output_dir / "splits"

    remove_fraction = float(
        args.remove_fraction
        if args.remove_fraction is not None
        else config.get("remove_fraction", 0.10)
    )
    if not 0 < remove_fraction < 1:
        raise ValueError("remove_fraction must be greater than 0 and less than 1")

    df = pd.read_csv(source_path)
    if "cer" not in df.columns:
        raise ValueError(f"Expected a cer column in {source_path}")
    if df.empty:
        raise ValueError(f"No scored transcriptions found in {source_path}")

    removed_n = math.ceil(len(df) * remove_fraction)
    ranked = df.assign(_cer_rank=df["cer"].fillna(1.0)).sort_values(
        "_cer_rank", ascending=False, kind="mergesort"
    )
    removed_df = ranked.head(removed_n).drop(columns=["_cer_rank"])
    kept_df = ranked.iloc[removed_n:].drop(columns=["_cer_rank"])

    split_seed = int(args.split_seed if args.split_seed is not None else config.get("split_seed", 42))
    ratios = SplitRatios(
        train=float(args.train_ratio if args.train_ratio is not None else config.get("train_ratio", 0.8)),
        dev=float(args.dev_ratio if args.dev_ratio is not None else config.get("dev_ratio", 0.1)),
        test=float(args.test_ratio if args.test_ratio is not None else config.get("test_ratio", 0.1)),
    )
    train_df, dev_df, test_df = split_rows(kept_df, seed=split_seed, ratios=ratios)

    splits_dir.mkdir()
    kept_df.to_csv(metadata_out, index=False)
    removed_df.to_csv(removed_out, index=False)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    dev_df.to_csv(splits_dir / "dev.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    summary = {
        "source_transcription_dir": str(transcription_dir),
        "source_preds_scored": str(source_path),
        "total_scored": len(df),
        "removed_count": removed_n,
        "retained_count": len(kept_df),
        "removed_fraction": removed_n / len(df),
        "requested_remove_fraction": remove_fraction,
        "note": "CER NaN values are treated as 1.0 when ranking noisy rows.",
    }
    default_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    split_summary = build_split_summary(train_df=train_df, dev_df=dev_df, test_df=test_df)
    split_summary["split_seed"] = split_seed
    split_summary["ratios"] = {
        "train": ratios.train,
        "dev": ratios.dev,
        "test": ratios.test,
    }
    save_split_summary(str(splits_dir / "split_summary.json"), split_summary)
    _write_note(
        output_dir,
        language=language,
        transcription_dir=transcription_dir,
        source_path=source_path,
        total_rows=len(df),
        removed_rows=removed_n,
    )

    print(f"CER-filter run ID: {run_id}")
    print(f"CER-filter output: {output_dir}")
    print(f"Kept {len(kept_df)} rows; removed {removed_n} noisy rows")


if __name__ == "__main__":
    main()
