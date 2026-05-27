from __future__ import annotations

# Standard libraries
import argparse
import json
import random

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Third-party libraries
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml

from torch.utils.data import Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Local imports
import src.data.schema as schema
import src.data.split as split
import src.utils.io as io

from src.evaluation.metrics import cer, normalize_text


class SpeechSeq2SeqDataset(Dataset):
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
        wav_path = io.resolve_audio_path(str(row[self.path_col]), self.audio_root)
        text = normalize_text(str(row[self.text_col]))

        waveform, sr = torchaudio.load(wav_path)
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.target_sr)
            sr = self.target_sr

        dur = waveform.numel() / float(sr)
        if dur < self.min_dur_sec or dur > self.max_dur_sec:
            return None

        return {
            "audio": waveform,
            "text": text,
            "path": wav_path,
        }


@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: AutoProcessor
    padding: bool = True
    input_dtype: torch.dtype = torch.float32

    def _max_input_length(self) -> Optional[int]:
        feature_extractor = self.processor.feature_extractor
        return getattr(feature_extractor, "n_samples", None)

    def __call__(self, features: List[dict]):
        features = [f for f in features if f is not None]
        if len(features) == 0:
            return {}

        audio = [f["audio"].numpy() for f in features]
        texts = [f["text"] for f in features]

        max_input_length = self._max_input_length()
        batch = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length" if max_input_length is not None else self.padding,
            max_length=max_input_length,
            truncation=True,
            return_attention_mask=True,
        )

        batch["input_features"] = batch["input_features"].to(self.input_dtype)

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


def load_config(args):
    if not args.config:
        return args

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key, value in config.items():
        setattr(args, key, value)

    print(f"Loaded config from {args.config}:")
    print(yaml.dump(config, sort_keys=False))
    return args


def load_split_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path else None


def compute_metrics_factory(processor: AutoProcessor):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        pred_ids = np.asarray(pred_ids)

        label_ids = np.asarray(pred.label_ids).copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        cers_no_space = []
        cers_with_space = []

        for ref, hyp in zip(label_str, pred_str):
            cers_no_space.append(
                cer(
                    ref,
                    hyp,
                    strip_punct=True,
                    remove_whitespace=True,
                    empty_ref_policy="skip",
                )
            )
            cers_with_space.append(
                cer(
                    ref,
                    hyp,
                    strip_punct=True,
                    remove_whitespace=False,
                    empty_ref_policy="skip",
                )
            )

        def _mean(vals):
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else 0.0

        return {
            "cer": _mean(cers_no_space),
            "cer_with_space": _mean(cers_with_space),
        }

    return compute_metrics


def maybe_set_generation_prompt(model, processor, language: Optional[str], task: str):
    if not language:
        return

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "set_prefix_tokens"):
        tokenizer.set_prefix_tokens(language=language, task=task)

    if hasattr(model.generation_config, "language"):
        model.generation_config.language = language
    if hasattr(model.generation_config, "task"):
        model.generation_config.task = task
    if hasattr(model.config, "language"):
        model.config.language = language
    if hasattr(model.config, "task"):
        model.config.task = task


def _get_model_torch_dtype(args) -> Optional[torch.dtype]:
    if getattr(args, "bf16", False):
        return torch.bfloat16
    if getattr(args, "fp16", False):
        return torch.float16
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument("--metadata", help="Path to metadata.csv")
    ap.add_argument("--audio_root", default=None)

    ap.add_argument("--train_csv", type=str, default=None)
    ap.add_argument("--dev_csv", type=str, default=None)
    ap.add_argument("--test_csv", type=str, default=None)

    ap.add_argument("--model_id", help="HF model id, e.g. openai/whisper-large-v3")
    ap.add_argument("--language", type=str, default=None)
    ap.add_argument("--task", type=str, default="transcribe")
    ap.add_argument("--generation_max_length", type=int, default=256)

    ap.add_argument("--train_seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--freeze_encoder", action="store_true")

    ap.add_argument("--out_dir", help="Output run directory")

    args = load_config(ap.parse_args())

    required = ["model_id", "out_dir"]
    for required_key in required:
        if getattr(args, required_key) is None:
            raise ValueError(f"{required_key} must be provided via CLI or config file")

    args.audio_root = io.expand_path(args.audio_root)
    args.metadata = io.expand_path(args.metadata)
    args.model_id = io.expand_path(args.model_id)
    args.out_dir = io.expand_path(args.out_dir)

    args.train_csv = io.expand_path(args.train_csv)
    args.dev_csv = io.expand_path(args.dev_csv)
    args.test_csv = io.expand_path(args.test_csv)

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
        train_df = load_split_csv(args.train_csv)
        dev_df = load_split_csv(args.dev_csv)
        test_df = load_split_csv(args.test_csv)
    else:
        if args.metadata is None:
            raise ValueError(
                "Provide --metadata or the full set of --train_csv/--dev_csv/--test_csv"
            )

        df = pd.read_csv(args.metadata)
        path_col = schema.pick_col(df, schema.PATH_COL_CANDIDATES)
        text_col = schema.pick_col(df, schema.TEXT_COL_CANDIDATES)
        df[text_col] = df[text_col].astype(str).map(normalize_text)
        df = df[df[text_col].str.len() > 0].copy()

        ratios = split.SplitRatios(train=0.8, dev=0.1, test=0.1)
        if hasattr(args, "split_seed") and getattr(args, "split_seed") is not None:
            split_seed = int(args.split_seed)
        else:
            split_seed = int(args.train_seed)

        if getattr(args, "split_by_col", None):
            try:
                train_df, dev_df, test_df = split.split_by_group(
                    df,
                    split_col=args.split_by_col,
                    seed=split_seed,
                    ratios=ratios,
                )
            except ValueError:
                if not getattr(args, "allow_row_fallback", False):
                    raise
                print("[WARN] Group split failed, falling back to row-level.")
                train_df, dev_df, test_df = split.split_rows(
                    df,
                    seed=split_seed,
                    ratios=ratios,
                )
        else:
            train_df, dev_df, test_df = split.split_rows(
                df,
                seed=split_seed,
                ratios=ratios,
            )

    path_col = schema.pick_col(train_df, schema.PATH_COL_CANDIDATES)
    text_col = schema.pick_col(train_df, schema.TEXT_COL_CANDIDATES)
    train_df[text_col] = train_df[text_col].astype(str).map(normalize_text)
    dev_df[text_col] = dev_df[text_col].astype(str).map(normalize_text)
    test_df[text_col] = test_df[text_col].astype(str).map(normalize_text)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_id,
        torch_dtype=_get_model_torch_dtype(args),
        low_cpu_mem_usage=True,
    )

    maybe_set_generation_prompt(model, processor, args.language, args.task)

    if args.freeze_encoder and hasattr(model, "freeze_encoder"):
        model.freeze_encoder()
    elif args.freeze_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    model.config.use_cache = False

    input_dtype = torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None

    train_ds = SpeechSeq2SeqDataset(train_df, args.audio_root, path_col, text_col)
    dev_ds = SpeechSeq2SeqDataset(dev_df, args.audio_root, path_col, text_col)
    test_ds = SpeechSeq2SeqDataset(test_df, args.audio_root, path_col, text_col)

    collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        input_dtype=input_dtype,
    )

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
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
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        dataloader_pin_memory=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics_factory(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
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

    test_metrics = trainer.evaluate(test_ds)
    (out_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )

    print("Done. Test metrics:", test_metrics)


if __name__ == "__main__":
    main()