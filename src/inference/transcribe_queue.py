from __future__ import annotations

"""Run the comparison checkpoints on their original train/dev/test manifests."""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.evaluation.metrics import cer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "evaluation" / "comparison_inference"
SPLITS = ("train", "dev", "test")
SUMMARY_FIELDS = (
    "name", "language", "edition", "model", "source", "baseline_test_cer",
    "train_cer", "dev_cer", "test_cer", "heldout_cer", "all_cer",
    "heldout_minus_train_pp", "test_minus_baseline_pp",
    "train_scored", "dev_scored", "test_scored", "train_rows", "dev_rows", "test_rows",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_queue(config_path: str | Path) -> dict:
    queue_path = Path(config_path).expanduser().resolve()
    raw = yaml.safe_load(queue_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("jobs"), list) or not raw["jobs"]:
        raise ValueError("Queue config must contain a nonempty jobs list")
    jobs = []
    seen = set()
    for relative in raw["jobs"]:
        if not isinstance(relative, str):
            raise ValueError("Each queue job must be a config filename")
        job_path = (queue_path.parent / relative).resolve()
        job = yaml.safe_load(job_path.read_text(encoding="utf-8"))
        if not isinstance(job, dict):
            raise ValueError(f"Invalid job config: {job_path}")
        required = {"name", "language", "edition", "model", "backend", "source", "audio_root", "train_csv", "dev_csv", "test_csv", "baseline_test_cer", "remove_spaces"}
        missing = required - job.keys()
        if missing:
            raise ValueError(f"{job_path}: missing {sorted(missing)}")
        name = job["name"]
        if not isinstance(name, str) or not name or name in seen or not all(c.isalnum() or c in "-_" for c in name):
            raise ValueError(f"Duplicate or invalid job name: {name!r}")
        seen.add(name)
        if job["backend"] not in {"ctc", "whisper"}:
            raise ValueError(f"{name}: unsupported backend")
        source = job["source"]
        if not isinstance(source, dict) or source.get("type") not in {"hub", "local"}:
            raise ValueError(f"{name}: source must be hub or local")
        if source["type"] == "hub" and not all(isinstance(source.get(key), str) and source[key] for key in ("repo_id", "revision")):
            raise ValueError(f"{name}: Hub source needs repo_id and revision")
        if source["type"] == "local" and not isinstance(source.get("path"), str):
            raise ValueError(f"{name}: local source needs path")
        if not isinstance(job["remove_spaces"], bool):
            raise ValueError(f"{name}: remove_spaces must be boolean")
        job["baseline_test_cer"] = float(job["baseline_test_cer"])
        for field in ("audio_root", "train_csv", "dev_csv", "test_csv"):
            job[field] = str(_path(job[field]))
        if "test_metrics" in job:
            job["test_metrics"] = str(_path(job["test_metrics"]))
        if source["type"] == "local":
            source["path"] = str(_path(source["path"]))
        job["config_path"] = str(job_path)
        jobs.append(job)
    return {"queue_config": str(queue_path), "jobs": jobs}


def _manifest_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"segment_path", "text"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: needs segment_path and text columns")
        return list(reader)


def validate_queue(queue: dict, *, check_hub: bool = False) -> None:
    errors = []
    hub_api = None
    if check_hub and any(job["source"]["type"] == "hub" for job in queue["jobs"]):
        from huggingface_hub import HfApi
        hub_api = HfApi()
    for job in queue["jobs"]:
        name = job["name"]
        root = Path(job["audio_root"])
        if not root.is_dir():
            errors.append(f"{name}: missing audio root {root}")
        seen_paths = set()
        for split in SPLITS:
            manifest = Path(job[f"{split}_csv"])
            if not manifest.is_file():
                errors.append(f"{name}/{split}: missing manifest {manifest}")
                continue
            try:
                rows = _manifest_rows(manifest)
                if not rows:
                    errors.append(f"{name}/{split}: empty manifest")
                for row in rows:
                    audio_path = row["segment_path"]
                    if not audio_path or audio_path in seen_paths:
                        errors.append(f"{name}/{split}: duplicate or empty segment path {audio_path!r}")
                        break
                    seen_paths.add(audio_path)
                    resolved = Path(audio_path) if Path(audio_path).is_absolute() else root / audio_path
                    if not resolved.is_file():
                        errors.append(f"{name}/{split}: missing audio {resolved}")
                        break
            except (OSError, ValueError) as exc:
                errors.append(f"{name}/{split}: {exc}")
        source = job["source"]
        if job.get("test_metrics"):
            try:
                saved_cer = float(json.loads(Path(job["test_metrics"]).read_text(encoding="utf-8"))["eval_cer"])
                if abs(saved_cer - job["baseline_test_cer"]) > 1e-12:
                    errors.append(f"{name}: baseline test CER differs from {job['test_metrics']}")
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{name}: cannot read saved test CER: {exc}")
        if source["type"] == "local":
            if not (Path(source["path"]) / "config.json").is_file():
                errors.append(f"{name}: missing local checkpoint {source['path']}")
        elif hub_api is not None:
            try:
                hub_api.model_info(source["repo_id"], revision=source["revision"], token=True)
            except Exception as exc:
                errors.append(f"{name}: Hub access failed for {source['repo_id']}@{source['revision']}: {exc}")
    if errors:
        raise ValueError("Comparison inference validation failed:\n- " + "\n- ".join(errors))


def _materialize_model(source: dict) -> str:
    if source["type"] == "local":
        return source["path"]
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=source["repo_id"], revision=source["revision"], token=True)


def _run_transcribe(job: dict, split: str, model_path: str, attempt_dir: Path, limit: int) -> Path:
    attempt_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, "-m", "src.inference.transcribe",
        "--model_id_or_path", model_path,
        "--metadata", job[f"{split}_csv"],
        "--utt_root", job["audio_root"],
        "--out_dir", str(attempt_dir),
        "--remove_spaces" if job["remove_spaces"] else "--no-remove_spaces",
    ]
    if limit:
        command.extend(("--limit", str(limit)))
    with (attempt_dir / "transcribe.log").open("w", encoding="utf-8") as log:
        log.write("Command: " + " ".join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"transcribe exited {result.returncode}; see {attempt_dir / 'transcribe.log'}")
    files = list(attempt_dir.glob("*/preds-*.csv"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one predictions CSV under {attempt_dir}, found {len(files)}")
    return files[0]


def score_predictions(predictions: Path, manifest: Path, limit: int, output: Path) -> dict:
    expected = _manifest_rows(manifest)
    if limit:
        expected = expected[:limit]
    with predictions.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"segment_path", "ref_text", "pred_text"}.issubset(reader.fieldnames):
            raise ValueError(f"{predictions}: missing prediction columns")
        actual = list(reader)
    if len(actual) != len(expected):
        raise ValueError(f"{predictions}: got {len(actual)} predictions, expected {len(expected)}; check transcription_errors TSV")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    scored = 0
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("segment_path", "ref_text", "pred_text", "cer"))
        writer.writeheader()
        for source, prediction in zip(expected, actual):
            if source["segment_path"] != prediction["segment_path"]:
                raise ValueError(f"{predictions}: prediction order differs from manifest")
            if source["text"] != prediction["ref_text"]:
                raise ValueError(f"{predictions}: reference differs from manifest for {source['segment_path']}")
            value = cer(source["text"], prediction["pred_text"], strip_punct=True, empty_ref_policy="skip")
            writer.writerow({"segment_path": source["segment_path"], "ref_text": source["text"], "pred_text": prediction["pred_text"], "cer": "" if value is None else repr(value)})
            if value is not None:
                scored += 1
                total += value
    return {"rows": len(expected), "scored": scored, "cer_sum": total, "cer": total / scored if scored else None, "predictions": str(predictions), "scored_predictions": str(output)}


def _mean(parts: list[dict]) -> float | None:
    count = sum(part["scored"] for part in parts)
    return sum(part["cer_sum"] for part in parts) / count if count else None


def write_summary(state: dict, run_dir: Path) -> Path:
    target = run_dir / ("partial_summary.csv" if state["limit"] else "summary.csv")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for entry in state["jobs"]:
            if any(entry["splits"][split]["status"] != "succeeded" for split in SPLITS):
                continue
            job = entry["config"]
            metrics = {split: entry["splits"][split]["metrics"] for split in SPLITS}
            heldout = _mean([metrics["dev"], metrics["test"]])
            all_cer = _mean(list(metrics.values()))
            train = metrics["train"]["cer"]
            test = metrics["test"]["cer"]
            writer.writerow({
                "name": job["name"], "language": job["language"], "edition": job["edition"], "model": job["model"],
                "source": (job["source"]["repo_id"] + "@" + job["source"]["revision"])
                if job["source"]["type"] == "hub" else job["source"]["path"],
                "baseline_test_cer": job["baseline_test_cer"],
                "train_cer": train, "dev_cer": metrics["dev"]["cer"], "test_cer": test,
                "heldout_cer": heldout, "all_cer": all_cer,
                "heldout_minus_train_pp": (heldout - train) * 100 if heldout is not None and train is not None else "",
                "test_minus_baseline_pp": (test - job["baseline_test_cer"]) * 100 if test is not None else "",
                "train_scored": metrics["train"]["scored"], "dev_scored": metrics["dev"]["scored"], "test_scored": metrics["test"]["scored"],
                "train_rows": metrics["train"]["rows"], "dev_rows": metrics["dev"]["rows"], "test_rows": metrics["test"]["rows"],
            })
    os.replace(temporary, target)
    return target


def _new_state(queue: dict, limit: int, output_root: Path) -> tuple[dict, Path]:
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": 1, "queue_config": queue["queue_config"], "limit": limit,
        "created_at": _now(), "status": "pending",
        "jobs": [{"config": job, "splits": {split: {"status": "pending", "metrics": None, "error": None} for split in SPLITS}} for job in queue["jobs"]],
    }
    _atomic_json(run_dir / "queue_state.json", state)
    return state, run_dir


def run_state(state: dict, run_dir: Path) -> int:
    state_path = run_dir / "queue_state.json"
    state["status"] = "running"
    _atomic_json(state_path, state)
    failed = False
    for index, entry in enumerate(state["jobs"], 1):
        job = entry["config"]
        if all(entry["splits"][split]["status"] == "succeeded" for split in SPLITS):
            continue
        print(f"Model {index}/{len(state['jobs'])}: {job['name']}", flush=True)
        try:
            model_path = _materialize_model(job["source"])
        except KeyboardInterrupt:
            state["status"] = "interrupted"
            _atomic_json(state_path, state)
            raise
        except Exception as exc:
            entry["error"] = f"Model load/download failed: {exc}"
            failed = True
            _atomic_json(state_path, state)
            continue
        for split in SPLITS:
            part = entry["splits"][split]
            if part["status"] == "succeeded":
                continue
            part.update(status="running", error=None)
            _atomic_json(state_path, state)
            print(f"  {split}: transcribing", flush=True)
            try:
                attempt_dir = run_dir / "jobs" / job["name"] / split / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                predictions = _run_transcribe(job, split, model_path, attempt_dir, state["limit"])
                metrics = score_predictions(predictions, Path(job[f"{split}_csv"]), state["limit"], attempt_dir / "preds_scored.csv")
                part.update(status="succeeded", metrics=metrics, error=None)
                print(f"  {split}: CER={metrics['cer']} ({metrics['scored']}/{metrics['rows']} scored)", flush=True)
            except KeyboardInterrupt:
                part.update(status="interrupted", error="Interrupted")
                state["status"] = "interrupted"
                _atomic_json(state_path, state)
                write_summary(state, run_dir)
                raise
            except Exception as exc:
                part.update(status="failed", error=str(exc))
                failed = True
                print(f"  {split}: FAILED: {exc}", file=sys.stderr, flush=True)
            finally:
                _atomic_json(state_path, state)
                write_summary(state, run_dir)
    state["status"] = "failed" if failed else "succeeded"
    state["finished_at"] = _now()
    _atomic_json(state_path, state)
    write_summary(state, run_dir)
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Queue YAML to start")
    source.add_argument("--resume", help="Existing evaluation run directory")
    parser.add_argument("--validate-only", action="store_true", help="Check manifests, audio, checkpoints, and Hub access without inference")
    parser.add_argument("--limit", type=int, default=0, help="Transcribe at most N clips per split; writes a partial summary")
    parser.add_argument("--job", help="Run one named model from the queue (useful for a pilot)")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUTPUT_ROOT), help="Root for timestamped queue runs")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.resume:
        if args.validate_only or args.limit or args.job:
            parser.error("--resume cannot be combined with --validate-only, --limit, or --job")
        run_dir = Path(args.resume).expanduser().resolve()
        state = json.loads((run_dir / "queue_state.json").read_text(encoding="utf-8"))
        if state.get("schema_version") != 1:
            parser.error("Unsupported queue state")
        queue = {"jobs": [
            entry["config"] for entry in state["jobs"]
            if any(entry["splits"][split]["status"] != "succeeded" for split in SPLITS)
        ]}
    else:
        queue = load_queue(args.config)
        if args.job:
            queue["jobs"] = [job for job in queue["jobs"] if job["name"] == args.job]
            if not queue["jobs"]:
                parser.error(f"job {args.job!r} was not found in the queue")
    full_run = (state["limit"] if args.resume else args.limit) == 0
    if not args.validate_only and full_run:
        import torch
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            parser.error("Full comparison inference requires an accessible CUDA or MPS GPU")
    validate_queue(queue, check_hub=True)
    print(f"Validated {len(queue['jobs'])} pending models and their split manifests", flush=True)
    if args.validate_only:
        return
    if not args.resume:
        state, run_dir = _new_state(queue, args.limit, _path(args.out_dir))
    print(f"Queue run: {run_dir}", flush=True)
    try:
        raise SystemExit(run_state(state, run_dir))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
