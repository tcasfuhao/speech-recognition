from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from src.finetune.asr_config import load_config, validate_config, write_validation_report


MODULES = {
    "ctc": "src.finetune.train_ctc",
    "whisper": "src.finetune.train_seq2seq",
    "granite": "src.finetune.train_granite",
}


def _smoke_config(config: dict, config_path: Path) -> Path:
    smoke = {key: value for key, value in config.items() if not key.startswith("_")}
    smoke["epochs"] = 1
    smoke["max_train_samples"] = min(int(smoke.get("max_train_samples", 8)), 8)
    smoke["max_eval_samples"] = min(int(smoke.get("max_eval_samples", 4)), 4)
    smoke["run_prefix"] = "smoke"
    target = Path("logs/smoke/configs") / f"{config_path.stem}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(smoke, sort_keys=False), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and dispatch an explicit ASR backend")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    report = validate_config(config)
    report_path = write_validation_report(config, report)
    if not report["valid"]:
        raise SystemExit("Configuration validation failed before run creation:\n- " + "\n- ".join(report["errors"]))
    print(f"Validation passed; report: {report_path}", flush=True)
    if args.validate_only:
        return

    dispatch_config = _smoke_config(config, config_path) if args.smoke else config_path
    completed = subprocess.run(
        [sys.executable, "-m", MODULES[config["backend"]], "--config", str(dispatch_config)],
        check=False,
    )
    if completed.returncode:
        raise SystemExit(f"{config['backend']} backend exited with status {completed.returncode}")


if __name__ == "__main__":
    main()
