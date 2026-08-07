from __future__ import annotations

# Standard libraries
import argparse
from pathlib import Path

# Third-party libraries
import pandas as pd
import yaml

# Local imports
import src.utils.io as io
import src.data.schema as schema

from src.evaluation.metrics import cer, strip_whitespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument(
        "--preds", help="Path to preds.tsv produced by transcribe.py"
    )
    ap.add_argument(
        "--out_dir", default="outputs",
        help="Directory to write scored preds + summaries"
    )

    ap.add_argument(
        "--strip_punct", action="store_true",
        help="Strip punctuation before CER"
    )
    ap.add_argument(
        "--empty_ref_policy", default="skip", choices=["skip", "zero", "raise"]
    )

    ap.add_argument(
        "--extra_metadata",
        help="Optional CSV/TSV with extra metadata to merge"
    )
    ap.add_argument(
        "--merge_on", default="segment_path", help="Column to merge on"
    )
    ap.add_argument(
        "--group_by",
        action="append", default=[],
        help="Column to summarize by; can be passed multiple times",
    )

    args = ap.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(args, k, v)

        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config, sort_keys=False))

    required = ["preds"]
    for r in required:
        if getattr(args, r) is None:
            raise ValueError(f"{r} must be provided via CLI or config file")

    args.preds = io.expand_path(args.preds)
    args.out_dir = io.expand_path(args.out_dir)
    args.extra_metadata = io.expand_path(args.extra_metadata)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.preds)

    if args.extra_metadata:
        meta_df = pd.read_csv(args.extra_metadata, sep=None, engine="python")

        if args.merge_on not in df.columns:
            raise ValueError(
                f"Merge column {args.merge_on} not found in preds"
            )

        if args.merge_on not in meta_df.columns:
            raise ValueError(
                f"Merge column {args.merge_on} not found in metadata"
            )

        df = df.merge(meta_df, on=args.merge_on, how="left")

    dur_col = schema.pick_col(
        df, schema.DUR_COL_CANDIDATES, required=False
    )
    try:
        hyp_col = schema.pick_col(
            df, schema.HYP_COL_CANDIDATES
        )
    except ValueError as exc:
        available = list(df.columns)
        if "loss" in available and "eval_loss" in available:
            raise ValueError(
                "The preds file looks like a training log (train_log.tsv). "
                "evaluate_preds expects the preds CSV created by "
                "src.inference.transcribe (with pred_text/ref_text columns)."
            ) from exc
        raise
    loc_col = schema.pick_col(
        df, schema.LOCATION_COL_CANDIDATES, required=False
    )
    model_col = schema.pick_col(
        df, schema.MODEL_COL_CANDIDATES, required=False
    )
    rec_col = schema.pick_col(
        df, schema.RECORDING_COL_CANDIDATES, required=False
    )
    try:
        ref_col = schema.pick_col(
            df, schema.TEXT_COL_CANDIDATES
        )
    except ValueError as exc:
        raise ValueError(
            "No reference text column found. "
            "Make sure the preds CSV includes ref_text (or text/reference)."
        ) from exc
    sess_col = schema.pick_col(
        df, schema.SESSION_COL_CANDIDATES, required=False
    )
    spk_col = schema.pick_col(
        df, schema.SPEAKER_COL_CANDIDATES, required=False
    )

    cer_vals = []

    for ref, hyp in zip(df[ref_col], df[hyp_col]):

        ref = strip_whitespace(ref) if pd.notna(ref) else ""
        hyp = strip_whitespace(hyp) if pd.notna(hyp) else ""

        cer_vals.append(
            cer(
                ref,
                hyp,
                strip_punct=args.strip_punct,
                empty_ref_policy=args.empty_ref_policy,
            )
        )

    df[ref_col] = df[ref_col].map(lambda value: strip_whitespace(value) if pd.notna(value) else "")
    df[hyp_col] = df[hyp_col].map(lambda value: strip_whitespace(value) if pd.notna(value) else "")
    df["cer"] = cer_vals

    scored_path = out_dir / "preds_scored.csv"
    df.to_csv(scored_path, index=False)

    overall = pd.DataFrame(
        [{
            "n_rows": int(df.shape[0]),
            "n_scored": int(df["cer"].notna().sum()),
            dur_col: float(df[dur_col].sum()) if dur_col else None,
            "mean_cer": float(df["cer"].dropna().mean()),
            "median_cer": float(df["cer"].dropna().median()),
        }]
    )

    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    def _group_summary(group_col: str, out_name: str):
        if group_col is None:
            return

        g = (
            df.groupby(group_col, dropna=False)
            .agg(
                n_scored=("cer", "count"),
                dur=(dur_col, "sum") if dur_col else ("cer", "count"),
                mean_cer=("cer", "mean"),
                median_cer=("cer", "median"),
            )
            .reset_index()
            .sort_values("mean_cer", ascending=True)
        )

        if dur_col:
            g.rename(columns={"dur": dur_col}, inplace=True)

        g.to_csv(out_dir / out_name, index=False)

    print(f"Wrote: {scored_path}")
    print(f"Wrote: {out_dir / 'summary_overall.csv'}")

    group_cols = list(
        dict.fromkeys([spk_col, rec_col, model_col, loc_col, sess_col])
    )
    if args.group_by:
        for col in args.group_by:
            if col not in group_cols:
                group_cols.append(col)

    for col in group_cols:
        if col not in df.columns:
            print(
                f"Warning: Grouping column {col} not found in data;"
                f" skipping group summary for this column"
            )

            continue

        _group_summary(col, f"summary_by_{col}.csv")
        print(f"Wrote: {out_dir / f'summary_by_{col}.csv'}")


if __name__ == "__main__":
    main()
