from __future__ import annotations

# Standard libraries
import argparse
import json
import random

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml

from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
)

# Local imports
import src.utils.io as io
import src.data.schema as schema
import src.data.split as split

from src.evaluation.metrics import cer, prepare_asr_text
from src.finetune.asr_config import write_experiment_summary
from src.utils.text_policy import write_text_policy


def build_char_vocab(texts: List[str]) -> Dict[str, int]:
    chars = set()
    for t in texts:
        chars.update(list(str(t)))
    if " " in chars:
        chars.remove(" ")
        chars.add("|")
    vocab = sorted(chars)

    vocab_dict = {c: i for i, c in enumerate(vocab)}
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    return vocab_dict


class ASRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_root: Optional[str],
        path_col: str,
        text_col: str,
        target_sr: int = 16000,
        min_dur_sec: float = 0.20,
        max_dur_sec: float = 30.0,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_root = audio_root
        self.path_col = path_col
        self.text_col = text_col
        self.target_sr = target_sr
        self.min_dur_sec = min_dur_sec
        self.max_dur_sec = max_dur_sec

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = io.resolve_audio_path(
            str(row[self.path_col]), self.audio_root
        )
        text = str(row[self.text_col])

        waveform, sr = torchaudio.load(wav_path)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform, sr, self.target_sr
            )
            sr = self.target_sr

        dur = waveform.numel() / float(sr)
        if dur < self.min_dur_sec or dur > self.max_dur_sec:
            return None

        return {"input_values": waveform, "text": text, "path": wav_path}


@dataclass
class DataCollatorCTC:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features: List[dict]):
        features = [f for f in features if f is not None]
        if len(features) == 0:
            return {}

        input_values = [f["input_values"].numpy() for f in features]
        texts = [f["text"] for f in features]

        batch = self.processor(
            input_values,
            sampling_rate=16000,
            return_tensors="pt",
            padding=self.padding,
        )

        labels_batch = self.processor.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        batch["labels"] = labels
        return batch


def compute_metrics_factory(processor: Wav2Vec2Processor):
    def compute_metrics(pred):
        predictions = pred.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_ids = (
            np.argmax(predictions, axis=-1)
            if np.asarray(predictions).ndim == 3
            else predictions
        )
        pred_str = processor.batch_decode(pred_ids)

        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        cers = []

        for r, h in zip(label_str, pred_str):
            cers.append(
                cer(
                    r,
                    h,
                    strip_punct=True,
                    empty_ref_policy="skip",
                )
            )

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "cer": _mean(cers),
        }

    return compute_metrics


def preprocess_logits_for_metrics(logits, _labels):
    """Keep CTC evaluation memory bounded by retaining token IDs, not logits."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)


def sanity_check_tokenizer(processor, texts, max_examples=10):
    tokenizer = processor.tokenizer
    unk_id = tokenizer.unk_token_id

    print("\n[Tokenizer sanity check]")

    sample = texts.iloc[0]
    print("Sample text:", sample)
    print("Tokens:", tokenizer.tokenize(sample))
    print("IDs:", tokenizer(sample).input_ids)
    print("Decoded:", tokenizer.decode(tokenizer(sample).input_ids))
    print()

    n_rows_with_unk = 0
    examples_with_unk = []

    for t in texts.astype(str):
        ids = tokenizer(t).input_ids
        if unk_id in ids:
            n_rows_with_unk += 1
            if len(examples_with_unk) < max_examples:
                examples_with_unk.append((t, ids, tokenizer.decode(ids)))

    print(f"Rows with [UNK]: {n_rows_with_unk} / {len(texts)}")

    if examples_with_unk:
        print("\nExamples containing [UNK]:")
        for t, ids, dec in examples_with_unk:
            print(t)
            print(ids)
            print(dec)
            print("---")

    if n_rows_with_unk > 0:
        raise ValueError(
            f"Tokenizer produced [UNK] in {n_rows_with_unk}/{len(texts)} rows. "
            "Fix the vocabulary or normalization before training."
        )


def _load_split_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument("--metadata", help="Path to metadata.csv")
    ap.add_argument("--audio_root", default=None)

    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--dev_csv", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)

    ap.add_argument("--model_id", help="HF model id, e.g. facebook/mms-1b-all")

    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--split_by_col", type=str, default=None)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--dev_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--allow_row_fallback", action="store_true")

    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--eval_accumulation_steps", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    ap.add_argument("--out_dir", help="Output run directory")
    ap.add_argument("--remove_spaces", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(args, k, v)

        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config, sort_keys=False))

    required = ["model_id", "out_dir"]
    for r in required:
        if getattr(args, r) is None:
            raise ValueError(f"{r} must be provided via CLI or config file")

    args.audio_root = io.expand_path(args.audio_root)
    args.metadata = io.expand_path(args.metadata)
    args.model_id = io.expand_path(args.model_id)
    args.out_dir = io.expand_path(args.out_dir)

    args.train_csv = io.expand_path(args.train_csv)
    args.dev_csv = io.expand_path(args.dev_csv)
    args.test_csv = io.expand_path(args.test_csv)

    # The dispatcher validates architecture and storage before this point.

    model_name = args.model_id.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_name}_{timestamp}"

    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train_seed)

    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )

    if args.train_csv and args.dev_csv and args.test_csv:
        train_df = _load_split_csv(args.train_csv)
        dev_df = _load_split_csv(args.dev_csv)
        test_df = _load_split_csv(args.test_csv)
    else:
        if args.metadata is None:
            raise ValueError(
                "Provide --metadata or the full set of --train_csv/--dev_csv/--test_csv"
            )

        df = pd.read_csv(args.metadata)
        path_col = schema.pick_col(df, schema.PATH_COL_CANDIDATES)
        text_col = schema.pick_col(df, schema.TEXT_COL_CANDIDATES)
        df[text_col] = df[text_col].astype(str)
        df = df[df[text_col].str.len() > 0].copy()

        ratios = split.SplitRatios(
            train=float(args.train_ratio),
            dev=float(args.dev_ratio),
            test=float(args.test_ratio),
        )

        if args.split_by_col:
            try:
                train_df, dev_df, test_df = split.split_by_group(
                    df,
                    split_col=args.split_by_col,
                    seed=args.split_seed,
                    ratios=ratios,
                )
            except ValueError:
                if not args.allow_row_fallback:
                    raise
                print("[WARN] Group split failed, falling back to row-level.")
                train_df, dev_df, test_df = split.split_rows(
                    df, seed=args.split_seed, ratios=ratios
                )
                args.split_by_col = None
        else:
            train_df, dev_df, test_df = split.split_rows(
                df, seed=args.split_seed, ratios=ratios
            )

        split_summary = split.build_split_summary(
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            split_col=args.split_by_col,
            dur_col=schema.pick_col(df, schema.DUR_COL_CANDIDATES, required=False),
        )
        split_summary["split_seed"] = int(args.split_seed)
        split_summary["ratios"] = {
            "train": ratios.train,
            "dev": ratios.dev,
            "test": ratios.test,
        }
        (out_dir / "split_summary.json").write_text(
            json.dumps(split_summary, indent=2),
            encoding="utf-8",
        )

    path_col = schema.pick_col(train_df, schema.PATH_COL_CANDIDATES)
    text_col = schema.pick_col(train_df, schema.TEXT_COL_CANDIDATES)
    target_transform = lambda value: prepare_asr_text(value, args.remove_spaces)
    train_df[text_col] = train_df[text_col].astype(str).map(target_transform)
    dev_df[text_col] = dev_df[text_col].astype(str).map(target_transform)
    test_df[text_col] = test_df[text_col].astype(str).map(target_transform)

    if getattr(args, "max_train_samples", None):
        train_df = train_df.head(int(args.max_train_samples)).copy()
    if getattr(args, "max_eval_samples", None):
        dev_df = dev_df.head(int(args.max_eval_samples)).copy()
        test_df = test_df.head(int(args.max_eval_samples)).copy()

    vocab = build_char_vocab(train_df[text_col].tolist())
    vocab_path = out_dir / "vocab.json"
    vocab_path.write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_file=str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="" if args.remove_spaces else "|",
        replace_word_delimiter_char="" if args.remove_spaces else " ",
        do_lower_case=False,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)

    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    sanity_check_tokenizer(processor, train_df[text_col])

    config = AutoConfig.from_pretrained(args.model_id)
    config.vocab_size = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.ctc_zero_infinity = True
    config.remove_spaces = bool(args.remove_spaces)

    model = AutoModelForCTC.from_pretrained(
        args.model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )

    model.freeze_feature_encoder()
    model.gradient_checkpointing_enable()

    train_ds = ASRDataset(train_df, args.audio_root, path_col, text_col)
    dev_ds = ASRDataset(dev_df, args.audio_root, path_col, text_col)
    test_ds = ASRDataset(test_df, args.audio_root, path_col, text_col)

    collator = DataCollatorCTC(processor=processor)

    train_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_accumulation_steps=args.eval_accumulation_steps,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        seed=args.train_seed,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics_factory(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience)
        ],
    )

    trainer.train()

    best_info = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }

    (out_dir / "best_checkpoint.json").write_text(
        json.dumps(best_info, indent=2),
        encoding="utf-8",
    )

    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(out_dir / "train_log.tsv", sep="\t", index=False)

    final_dir = out_dir / "best"
    final_dir.mkdir(exist_ok=True, parents=True)
    trainer.save_model(str(final_dir))
    processor.save_pretrained(str(final_dir))
    write_text_policy(final_dir, args.remove_spaces)

    test_metrics = trainer.evaluate(test_ds)

    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    write_experiment_summary(
        "ctc", run_name,
        {"model_id": args.model_id, "output_dir": str(out_dir), "best": best_info, "test": test_metrics},
    )

    print("Done. Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
