from __future__ import annotations

import argparse
import json
import shutil
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from legacy.src.finetune.allosaurus_backend import edit_distance, prepare_manifests, run_feature_preparation
from src.finetune.asr_config import write_experiment_summary
from src.utils.text_policy import write_text_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare, adapt, and evaluate pinned Allosaurus uni2005")
    parser.add_argument("--config", required=True)
    config = yaml.safe_load(Path(parser.parse_args().config).read_text(encoding="utf-8"))
    run_name = f"{config.get('run_prefix', 'uni2005')}_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        local_root, work_root, records = prepare_manifests(config, run_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    run_feature_preparation(config, work_root)

    source_root = Path(config["allosaurus_root"]).expanduser().resolve()
    pretrained_dir = source_root / "allosaurus" / "pretrained"
    output = Path(config["out_dir"]).expanduser() / run_name
    model_dir = output / "model"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(pretrained_dir / "uni2005", model_dir)
    write_text_policy(model_dir, True)

    # Native Allosaurus resolves adaptation names inside its source model store.
    # A temporary symlink lets its unmodified trainer write checkpoints externally.
    link = pretrained_dir / run_name
    link.symlink_to(model_dir.resolve(), target_is_directory=True)
    try:
        from allosaurus.am.factory import transfer_am
        from allosaurus.am.loader import read_loader
        from allosaurus.am.trainer import Trainer

        train_config = Namespace(
            pretrained_model="uni2005", new_model=run_name, path=str(work_root),
            lang=str((local_root / "target_phone_inventory.txt").resolve()),
            device_id=int(config.get("device_id", 0)), batch_frame_size=int(config.get("batch_frame_size", 6000)),
            criterion="ctc", optimizer="sgd", lr=float(config.get("lr", 0.01)), grad_clip=5.0,
            epoch=int(config.get("epochs", 10)), log=str(local_root / "adaptation.log"), verbose=True,
            report_per_batch=int(config.get("report_per_batch", 10)),
        )
        train_loader = read_loader(work_root / "train", train_config)
        validate_loader = read_loader(work_root / "validate", train_config)
        trainer = Trainer(transfer_am(train_config), train_config)
        trainer.train(train_loader, validate_loader)
        train_loader.close()
        validate_loader.close()
    finally:
        link.unlink(missing_ok=True)

    from allosaurus.app import read_recognizer
    recognizer = read_recognizer("model", alt_model_path=output)
    predictions = []
    total_errors = total_phones = 0
    for item in records["test"]:
        try:
            hypothesis = recognizer.recognize(item["audio"], lang_id=str(local_root / "target_phone_inventory.txt"))
            hyp_tokens = hypothesis.split()
            errors = edit_distance(item["phones"], hyp_tokens)
            total_errors += errors
            total_phones += len(item["phones"])
            predictions.append({**item, "prediction": "".join(hyp_tokens), "errors": errors, "failure": ""})
        except Exception as exc:
            predictions.append({**item, "prediction": "", "errors": "", "failure": str(exc)})
    pd.DataFrame(predictions).to_csv(local_root / "test_predictions.csv", index=False)
    metrics = {"phone_error_rate": total_errors / total_phones if total_phones else None, "reference_phones": total_phones, "failures": sum(bool(x["failure"]) for x in predictions)}
    (local_root / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_experiment_summary(
        "allosaurus", run_name,
        {"model_id": "uni2005", "output_dir": str(output), "test": metrics},
    )


if __name__ == "__main__":
    main()
