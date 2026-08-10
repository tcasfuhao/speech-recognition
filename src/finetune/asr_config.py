from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data import schema
from src.utils.io import expand_path, resolve_audio_path


SUPPORTED_BACKENDS = {"ctc", "whisper", "granite", "allosaurus"}
KNOWN_MODELS = {
    "facebook/mms-1b-all": "ctc",
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft": "ctc",
    "neurlang/ipa-whisper-base": "whisper",
    "openai/whisper-large-v3": "whisper",
    "ibm-granite/granite-4.0-1b-speech": "granite",
}
EXPECTED_MODEL_TYPES = {"ctc": {"wav2vec2"}, "whisper": {"whisper"}, "granite": {"granite_speech"}}
REJECTED_MODEL_HINTS = {
    "byt5": "text-only",
    "ministral": "text-only",
    "pronounceai": "G2P, not ASR",
    "xphonebert": "text-only",
    "vibevoice": "outside the selected ASR scope",
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError("Training config must be a YAML mapping")
    config["_config_path"] = str(config_path)
    return config


def _expanded(config: dict[str, Any], key: str) -> str | None:
    value = config.get(key)
    return expand_path(str(value)) if value is not None else None


def validate_config(config: dict[str, Any], *, check_audio: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    architecture: dict[str, Any] | None = None
    backend = config.get("backend")
    if backend not in SUPPORTED_BACKENDS:
        errors.append(f"backend must be one of {sorted(SUPPORTED_BACKENDS)}")
    if not isinstance(config.get("remove_spaces", True), bool):
        errors.append("remove_spaces must be true or false")

    model_id = str(config.get("model_id", ""))
    lowered = model_id.lower()
    for marker, reason in REJECTED_MODEL_HINTS.items():
        if marker in lowered:
            errors.append(f"model_id {model_id!r} is incompatible: {reason}")

    if backend == "allosaurus":
        if config.get("remove_spaces", True) is False:
            errors.append(
                "Allosaurus cannot preserve word-boundary spaces in its phone-label targets; "
                "set remove_spaces: true"
            )
        if config.get("pretrained_model") != "uni2005":
            errors.append("Allosaurus pretrained_model must be pinned to 'uni2005'")
        root = Path(_expanded(config, "allosaurus_root") or "")
        model_root = root / "allosaurus" / "pretrained" / "uni2005"
        if not (model_root / "phone.txt").is_file():
            errors.append(f"uni2005 is not installed beneath allosaurus_root: {root}")
        expected_sha = config.get("model_sha256")
        model_file = model_root / "model.pt"
        if expected_sha and model_file.is_file():
            actual_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()
            if actual_sha != expected_sha:
                errors.append(f"uni2005 model.pt checksum mismatch: {actual_sha}")
    else:
        expected = KNOWN_MODELS.get(model_id)
        if expected is None:
            errors.append(
                f"unsupported model_id {model_id!r}; use one of the explicit supported model configs"
            )
        elif expected != backend:
            errors.append(
                f"model_id {model_id!r} has backend {expected!r}, not {backend!r}"
            )
        else:
            try:
                from transformers import AutoConfig

                hf_config = AutoConfig.from_pretrained(model_id)
                architecture = {
                    "model_type": hf_config.model_type,
                    "architectures": list(getattr(hf_config, "architectures", []) or []),
                }
                if hf_config.model_type not in EXPECTED_MODEL_TYPES[backend]:
                    errors.append(
                        f"model architecture {hf_config.model_type!r} is incompatible with backend {backend!r}"
                    )
            except Exception as exc:
                errors.append(f"could not inspect model architecture for {model_id!r}: {exc}")

    out_dir = _expanded(config, "out_dir")
    if not out_dir or f"/processed/asr/{backend}/" not in out_dir.rstrip("/") + "/":
        errors.append(f"out_dir must be beneath <data-root>/processed/asr/{backend}/")

    split_frames: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "dev", "test"):
        key = f"{split_name}_csv"
        path = _expanded(config, key)
        if not path or not Path(path).is_file():
            errors.append(f"{key} does not exist: {path}")
            continue
        frame = pd.read_csv(path)
        try:
            path_col = schema.pick_col(frame, schema.PATH_COL_CANDIDATES)
            schema.pick_col(frame, schema.TEXT_COL_CANDIDATES)
        except ValueError as exc:
            errors.append(f"{key}: {exc}")
            continue
        split_frames[split_name] = frame
        if check_audio:
            missing = [
                resolve_audio_path(str(value), _expanded(config, "audio_root"))
                for value in frame[path_col]
                if not Path(resolve_audio_path(str(value), _expanded(config, "audio_root"))).is_file()
            ]
            if missing:
                errors.append(f"{key}: {len(missing)} audio paths do not resolve (first: {missing[0]})")

    if len(split_frames) == 3:
        path_sets = {}
        for name, frame in split_frames.items():
            col = schema.pick_col(frame, schema.PATH_COL_CANDIDATES)
            path_sets[name] = set(frame[col].astype(str))
        for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
            overlap = path_sets[left] & path_sets[right]
            if overlap:
                errors.append(f"{left}/{right} manifests overlap in {len(overlap)} audio paths")

    if backend in {"whisper", "granite"} and config.get("lora", False):
        for key in ("lora_r", "lora_alpha", "lora_dropout", "lora_target_modules"):
            if key not in config:
                errors.append(f"LoRA config is missing {key}")
    if backend == "granite" and "<|audio|>" not in str(config.get("prompt", "")):
        errors.append("Granite prompt must contain the <|audio|> placeholder")

    report = {
        "valid": not errors,
        "backend": backend,
        "model": config.get("pretrained_model") if backend == "allosaurus" else model_id,
        "architecture": architecture,
        "split_rows": {name: len(frame) for name, frame in split_frames.items()},
        "errors": errors,
    }
    return report


def write_validation_report(config: dict[str, Any], report: dict[str, Any]) -> Path:
    config_path = Path(config["_config_path"])
    logs_dir = Path(config.get("validation_dir", "logs/validation")).expanduser()
    logs_dir.mkdir(parents=True, exist_ok=True)
    target = logs_dir / f"{config.get('backend', 'unknown')}__{config_path.stem}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def write_experiment_summary(backend: str, run_name: str, payload: dict[str, Any]) -> Path:
    target = Path("logs/experiments") / backend / f"{run_name}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
