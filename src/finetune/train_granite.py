from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Trainer, TrainingArguments

from src.data import schema
from src.evaluation.metrics import cer, prepare_asr_text
from src.finetune.train_seq2seq import SpeechSeq2SeqDataset
from src.finetune.asr_config import write_experiment_summary
from src.utils import io
from src.utils.text_policy import write_text_policy


@dataclass
class GraniteSpeechCollator:
    processor: Any
    prompt: str

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        features = [item for item in features if item is not None]
        audio = [item["audio"] for item in features]
        tokenizer = self.processor.tokenizer
        prefix_chats = [[{"role": "user", "content": self.prompt}]] * len(features)
        full_chats = [
            [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": item["text"]},
            ]
            for item in features
        ]
        prefixes = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            for chat in prefix_chats
        ]
        full = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in full_chats
        ]
        batch = self.processor(full, audio, device="cpu", return_tensors="pt", padding=True)
        prefix_batch = self.processor(prefixes, audio, device="cpu", return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        for row, length in enumerate(prefix_batch["attention_mask"].sum(dim=1).tolist()):
            labels[row, : int(length)] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)


def _load_config(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _predict(model, processor, dataset, prompt: str, max_tokens: int, remove_spaces: bool) -> tuple[list[dict], dict]:
    predictions = []
    values = []
    device = next(model.parameters()).device
    model.eval()
    for index in range(len(dataset)):
        item = dataset[index]
        if item is None:
            continue
        chat = [{"role": "user", "content": prompt}]
        formatted = processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(formatted, item["audio"], device=str(device), return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, num_beams=1)
        new_tokens = output[:, inputs["input_ids"].shape[-1] :]
        hypothesis = prepare_asr_text(processor.tokenizer.batch_decode(
            new_tokens, skip_special_tokens=True, add_special_tokens=False
        )[0], remove_spaces)
        score = cer(item["text"], hypothesis, strip_punct=True)
        if score is not None:
            values.append(score)
        predictions.append({"path": item["path"], "reference": item["text"], "prediction": hypothesis, "cer": score})
    return predictions, {"cer": float(np.mean(values)) if values else None, "examples": len(predictions)}


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Granite Speech ASR")
    parser.add_argument("--config", required=True)
    args = _load_config(parser.parse_args().config)

    for key in ("audio_root", "train_csv", "dev_csv", "test_csv", "out_dir"):
        args[key] = io.expand_path(args[key])
    frames = {name: pd.read_csv(args[f"{name}_csv"]) for name in ("train", "dev", "test")}
    path_col = schema.pick_col(frames["train"], schema.PATH_COL_CANDIDATES)
    text_col = schema.pick_col(frames["train"], schema.TEXT_COL_CANDIDATES)
    remove_spaces = bool(args.get("remove_spaces", True))
    for frame in frames.values():
        frame[text_col] = frame[text_col].astype(str).map(
            lambda value: prepare_asr_text(value, remove_spaces)
        )
    if args.get("max_train_samples"):
        frames["train"] = frames["train"].head(int(args["max_train_samples"]))
    if args.get("max_eval_samples"):
        for name in ("dev", "test"):
            frames[name] = frames[name].head(int(args["max_eval_samples"]))

    run_name = f"{args.get('run_prefix', 'granite')}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args["out_dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "run_config.json").write_text(json.dumps(args, indent=2), encoding="utf-8")

    seed = int(args.get("train_seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(args["model_id"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args["model_id"], torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("Granite LoRA training requires the 'peft' package") from exc
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(args["lora_r"]),
            lora_alpha=int(args["lora_alpha"]),
            lora_dropout=float(args["lora_dropout"]),
            target_modules=list(args["lora_target_modules"]),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    datasets = {
        name: SpeechSeq2SeqDataset(frame, args["audio_root"], path_col, text_col)
        for name, frame in frames.items()
    }
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=int(args.get("batch_size", 1)),
            per_device_eval_batch_size=int(args.get("eval_batch_size", 1)),
            gradient_accumulation_steps=int(args.get("grad_accum_steps", 8)),
            num_train_epochs=float(args.get("epochs", 10)),
            learning_rate=float(args.get("lr", 2e-4)),
            warmup_steps=int(args.get("warmup_steps", 20)),
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            bf16=True,
            gradient_checkpointing=True,
            remove_unused_columns=False,
            report_to="none",
            seed=seed,
        ),
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        data_collator=GraniteSpeechCollator(processor, args["prompt"]),
    )
    trainer.train()
    best = out_dir / "best"
    trainer.save_model(str(best))
    processor.save_pretrained(str(best))
    write_text_policy(best, remove_spaces)
    predictions, metrics = _predict(
        trainer.model, processor, datasets["test"], args["prompt"],
        int(args.get("generation_max_length", 256)), remove_spaces
    )
    pd.DataFrame(predictions).to_csv(out_dir / "test_predictions.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_experiment_summary(
        "granite", run_name,
        {"model_id": args["model_id"], "output_dir": str(out_dir), "test": metrics},
    )


if __name__ == "__main__":
    main()
