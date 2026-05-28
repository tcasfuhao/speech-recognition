from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import SplitRatios, build_split_summary, save_split_summary, split_rows


DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "data/processed/splits_cer90/cer_filter_summary.json"
DEFAULT_METADATA_OUT = PROJECT_ROOT / "data/processed/metadata_cer90.csv"
DEFAULT_REMOVED_OUT = PROJECT_ROOT / "data/processed/splits_cer90/removed_noisy_top10.csv"
DEFAULT_SPLITS_DIR = PROJECT_ROOT / "data/processed/splits_cer90"


def _resolve_path(path_str: str | None, default_path: Path) -> Path:
	if path_str is None:
		return default_path

	path = Path(path_str)
	if path.is_absolute():
		return path

	return PROJECT_ROOT / path


def main() -> None:
	ap = argparse.ArgumentParser(
		description="Remove the worst CER rows using a saved filter summary."
	)
	ap.add_argument(
		"--summary",
		default=str(DEFAULT_SUMMARY_PATH),
		help="Path to cer_filter_summary.json",
	)
	ap.add_argument(
		"--metadata-out",
		default=str(DEFAULT_METADATA_OUT),
		help="Where to write the filtered metadata CSV",
	)
	ap.add_argument(
		"--removed-out",
		default=str(DEFAULT_REMOVED_OUT),
		help="Where to write the removed rows CSV",
	)
	ap.add_argument(
		"--splits-dir",
		default=str(DEFAULT_SPLITS_DIR),
		help="Where to write train/dev/test CSVs",
	)
	ap.add_argument("--split-seed", type=int, default=42)
	ap.add_argument("--train-ratio", type=float, default=0.8)
	ap.add_argument("--dev-ratio", type=float, default=0.1)
	ap.add_argument("--test-ratio", type=float, default=0.1)
	args = ap.parse_args()

	summary_path = _resolve_path(args.summary, DEFAULT_SUMMARY_PATH)
	metadata_out = _resolve_path(args.metadata_out, DEFAULT_METADATA_OUT)
	removed_out = _resolve_path(args.removed_out, DEFAULT_REMOVED_OUT)
	splits_dir = _resolve_path(args.splits_dir, DEFAULT_SPLITS_DIR)

	summary = json.loads(summary_path.read_text(encoding="utf-8"))
	scored_path = _resolve_path(summary["source_preds_scored"], PROJECT_ROOT)
	df = pd.read_csv(scored_path)

	if "cer" not in df.columns:
		raise ValueError(f"Expected a cer column in {scored_path}")

	# Match the note in the summary: NaN CER is treated as 1.0 for ranking.
	df["_cer_rank"] = df["cer"].fillna(1.0)

	removed_n = int(summary["removed_count"])
	removed_df = df.sort_values("_cer_rank", ascending=False).head(removed_n).copy()
	kept_df = df.drop(index=removed_df.index).drop(columns=["_cer_rank"])
	removed_df = removed_df.drop(columns=["_cer_rank"])

	ratios = SplitRatios(
		train=float(args.train_ratio),
		dev=float(args.dev_ratio),
		test=float(args.test_ratio),
	)
	train_df, dev_df, test_df = split_rows(
		kept_df,
		seed=int(args.split_seed),
		ratios=ratios,
	)

	metadata_out.parent.mkdir(parents=True, exist_ok=True)
	removed_out.parent.mkdir(parents=True, exist_ok=True)
	splits_dir.mkdir(parents=True, exist_ok=True)

	kept_df.to_csv(metadata_out, index=False)
	removed_df.to_csv(removed_out, index=False)
	train_df.to_csv(splits_dir / "train.csv", index=False)
	dev_df.to_csv(splits_dir / "dev.csv", index=False)
	test_df.to_csv(splits_dir / "test.csv", index=False)

	split_summary = build_split_summary(
		train_df=train_df,
		dev_df=dev_df,
		test_df=test_df,
	)
	split_summary["split_seed"] = int(args.split_seed)
	split_summary["ratios"] = {
		"train": ratios.train,
		"dev": ratios.dev,
		"test": ratios.test,
	}
	save_split_summary(str(splits_dir / "split_summary.json"), split_summary)

	print(f"Wrote filtered metadata: {metadata_out}")
	print(f"Wrote removed rows: {removed_out}")
	print(f"Wrote CER-90 splits to: {splits_dir}")
	print(f"Kept {len(kept_df)} rows, removed {len(removed_df)} rows")


if __name__ == "__main__":
	main()