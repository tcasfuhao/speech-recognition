from __future__ import annotations

# Standard libraries
import argparse
import csv
import time

from datetime import datetime
from pathlib import Path

# Third-party libraries
import pandas as pd
import torch
import yaml

from tqdm import tqdm

# Local imports
import src.data.schema as schema
import src.utils.io as io

from src.inference.backends.factory import load_asr_backend
from src.utils.speech import load_audio_mono_16k
from src.evaluation.metrics import prepare_asr_text
from src.utils.text_policy import resolve_remove_spaces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument("--model_id_or_path", type=str, help="HF model id or path")

    ap.add_argument("--metadata", type=str, help="Path to metadata.csv")
    ap.add_argument(
        "--metadata_delimiter", type=str, default=",",
        help="Delimiter for metadata file (default: ',')"
    )
    ap.add_argument(
        "--utt_root", type=str, default=None,
        help="Optional root to prepend to relative paths"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="Optional limit for smoke test"
    )

    ap.add_argument("--out_dir", help="Output run directory")

    ap.add_argument(
        "--lm_path", type=str, default=None, help="Path to KenLM binary"
    )
    ap.add_argument("--lm_weight", type=float, default=0.5, help="LM weight")
    ap.add_argument("--beam_width", type=int, default=50)
    ap.add_argument(
        "--remove_spaces",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the saved model text policy (default: read it from the model)",
    )

    args = ap.parse_args()
    cli_remove_spaces = args.remove_spaces

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(args, k, v)

        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config, sort_keys=False))

    if cli_remove_spaces is not None:
        args.remove_spaces = cli_remove_spaces

    required = ["model_id_or_path", "metadata", "out_dir"]
    for r in required:
        if getattr(args, r) is None:
            raise ValueError(f"{r} must be provided via CLI or config file")

    args.model_id_or_path = io.expand_path(args.model_id_or_path)
    args.metadata = io.expand_path(args.metadata)
    args.out_dir = io.expand_path(args.out_dir)
    remove_spaces = resolve_remove_spaces(args.model_id_or_path, args.remove_spaces)
    print(f"ASR remove_spaces policy: {remove_spaces}")

    if args.utt_root is not None:
        args.utt_root = io.expand_path(args.utt_root)

    if args.lm_path is not None:
        args.lm_path = io.expand_path(args.lm_path)
        lm_tag = f"{Path(args.lm_path).stem}_w{args.lm_weight}_bw{args.beam_width}"
    else:
        lm_tag = None

    model_name = args.model_id_or_path.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    if lm_tag is None:
        preds_csv = out_dir / f"preds-{model_name}.csv"
        error_tsv = out_dir / f"transcription_errors-{model_name}.tsv"
    else:
        preds_csv = out_dir / f"preds-{model_name}-{lm_tag}.csv"
        error_tsv = out_dir / f"transcription_errors-{model_name}-{lm_tag}.tsv"

    df = pd.read_csv(args.metadata, delimiter=args.metadata_delimiter)

    end_col = schema.pick_col(
        df, schema.END_COL_CANDIDATES, required=False
    )
    loc_col = schema.pick_col(
        df, schema.LOCATION_COL_CANDIDATES, required=False
    )
    path_col = schema.pick_col(df, schema.PATH_COL_CANDIDATES)
    rec_col = schema.pick_col(
        df, schema.RECORDING_COL_CANDIDATES, required=False
    )
    sess_col = schema.pick_col(
        df, schema.SESSION_COL_CANDIDATES, required=False
    )
    spk_col = schema.pick_col(
        df, schema.SPEAKER_COL_CANDIDATES, required=False
    )
    start_col = schema.pick_col(
        df, schema.START_COL_CANDIDATES, required=False
    )
    text_col = schema.pick_col(
        df, schema.TEXT_COL_CANDIDATES, required=False
    )

    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    error_log = open(error_tsv, "a", newline="", encoding="utf-8")
    error_writer = csv.writer(error_log, delimiter="\t")

    if error_log.tell() == 0:
        error_writer.writerow([
            "timestamp",
            "segment_path",
            "recording_id",
            "start_ms",
            "end_ms",
            "text",
            "error_type",
            "error_msg",
        ])

    backend = load_asr_backend(
        model_id=args.model_id_or_path,
        device=device,
        dtype=dtype,
        lm_path=args.lm_path,
        lm_weight=args.lm_weight,
        beam_width=args.beam_width,
    )

    out_exists = preds_csv.exists()
    out_f = open(preds_csv, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        out_f,
        fieldnames=[
            "segment_path",
            "session_id",
            "recording_id",
            "start_ms",
            "end_ms",
            "dur_ms",
            "location",
            "speaker_id",
            "ref_text",
            "pred_text",
            "model_id",
            "device",
            "runtime_sec",
        ],
    )
    if not out_exists:
        writer.writeheader()

    n = 0
    for _, row in tqdm(df.iterrows(), desc="Transcribing", total=len(df)):
        seg_path = str(row[path_col])
        seg_path_fs = io.resolve_audio_path(seg_path, args.utt_root)

        ref_text = prepare_asr_text(
            str(row[text_col])
            if text_col and pd.notna(row[text_col])
            else "",
            remove_spaces,
        )

        session_id = str(row[sess_col]) if sess_col and pd.notna(row[sess_col]) else ""
        recording_id = str(row[rec_col]) if rec_col and pd.notna(row[rec_col]) else ""
        start_ms = int(row[start_col]) if start_col and pd.notna(row[start_col]) else ""
        end_ms = int(row[end_col]) if end_col and pd.notna(row[end_col]) else ""
        location = str(row[loc_col]) if loc_col and pd.notna(row[loc_col]) else ""
        speaker_id = str(row[spk_col]) if spk_col and pd.notna(row[spk_col]) else ""

        t0 = time.time()
        wav, _ = load_audio_mono_16k(seg_path_fs)
        dur_ms = float(wav.numel()) / 16000.0 * 1000

        try:
            result = backend.transcribe(wav)
            pred = prepare_asr_text(result.text, remove_spaces)
        except Exception as e:
            print("ERROR on:", seg_path_fs)
            error_writer.writerow([
                datetime.now().isoformat(),
                seg_path_fs,
                row.get(rec_col, "") if rec_col else "",
                row.get(start_col, "") if start_col else "",
                row.get(end_col, "") if end_col else "",
                row.get(text_col, "") if text_col else "",
                type(e).__name__,
                str(e),
            ])
            error_log.flush()
            continue

        rt = time.time() - t0

        writer.writerow(
            {
                "segment_path": seg_path,
                "session_id": session_id,
                "recording_id": recording_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "dur_ms": f"{dur_ms:.3f}",
                "location": location,
                "speaker_id": speaker_id,
                "ref_text": ref_text,
                "pred_text": pred,
                "model_id": args.model_id_or_path,
                "device": device,
                "runtime_sec": f"{rt:.3f}",
            }
        )
        out_f.flush()

        n += 1
        if args.limit and n >= args.limit:
            break

    out_f.close()

    print(f"Done. Wrote predictions to {preds_csv}")


if __name__ == "__main__":
    main()
