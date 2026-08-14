from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.finetune.asr_config import load_config, validate_config, write_validation_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUEUE_LOG_ROOT = PROJECT_ROOT / "logs" / "queues"
EXPERIMENT_LOG_ROOT = PROJECT_ROOT / "logs" / "experiments"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("queue and job names must contain a letter, number, dot, dash, or underscore")
    return cleaned


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_queue(path: str | Path) -> dict[str, Any]:
    queue_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(queue_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Queue config must be a YAML mapping")

    queue_name = raw.get("queue_name")
    jobs = raw.get("jobs")
    if not isinstance(queue_name, str) or not queue_name.strip():
        raise ValueError("queue_name must be a non-empty string")
    if not isinstance(raw.get("stop_on_failure", True), bool):
        raise ValueError("stop_on_failure must be true or false")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a non-empty list")

    seen_names: set[str] = set()
    seen_configs: set[Path] = set()
    resolved_jobs: list[dict[str, str]] = []
    for index, job in enumerate(jobs, start=1):
        if not isinstance(job, dict):
            raise ValueError(f"job {index} must be a mapping")
        name = job.get("name")
        config_value = job.get("config")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"job {index} name must be a non-empty string")
        safe_name = _safe_name(name)
        if safe_name in seen_names:
            raise ValueError(f"duplicate job name: {name!r}")
        if not isinstance(config_value, str) or not config_value.strip():
            raise ValueError(f"job {name!r} config must be a non-empty path")
        config_path = (queue_path.parent / config_value).expanduser().resolve()
        if config_path in seen_configs:
            raise ValueError(f"duplicate training config: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"job {name!r} training config does not exist: {config_path}")
        seen_names.add(safe_name)
        seen_configs.add(config_path)
        resolved_jobs.append({"name": name.strip(), "safe_name": safe_name, "config": str(config_path)})

    return {
        "queue_name": queue_name.strip(),
        "safe_queue_name": _safe_name(queue_name),
        "stop_on_failure": raw.get("stop_on_failure", True),
        "queue_config": str(queue_path),
        "jobs": resolved_jobs,
    }


def validate_queue(queue: dict[str, Any]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    for job in queue["jobs"]:
        config = load_config(job["config"])
        report = validate_config(config)
        report_path = write_validation_report(config, report)
        item = {
            **job,
            "backend": config.get("backend"),
            "model_id": config.get("model_id"),
            "configured_output_root": str(Path(str(config.get("out_dir", ""))).expanduser()),
            "validation_report": str(report_path.resolve()),
        }
        validated.append(item)
        if not report["valid"]:
            errors.extend(f"{job['name']}: {message}" for message in report["errors"])
    if errors:
        raise ValueError("Queue validation failed before training:\n- " + "\n- ".join(errors))
    return validated


def validate_pending_jobs(state: dict[str, Any]) -> None:
    pending = [job for job in state["jobs"] if job["status"] != "succeeded"]
    if not pending:
        return
    validated = validate_queue({"jobs": pending})
    by_name = {job["name"]: job for job in validated}
    for job in pending:
        details = by_name[job["name"]]
        for key in ("backend", "model_id", "configured_output_root", "validation_report"):
            job[key] = details[key]


def _new_state(queue: dict[str, Any], jobs: list[dict[str, Any]], smoke: bool) -> tuple[dict[str, Any], Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = QUEUE_LOG_ROOT / queue["safe_queue_name"] / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "schema_version": 1,
        "queue_name": queue["queue_name"],
        "queue_config": queue["queue_config"],
        "queue_run_dir": str(run_dir),
        "stop_on_failure": queue["stop_on_failure"],
        "smoke": smoke,
        "status": "pending",
        "created_at": _now(),
        "started_at": None,
        "finished_at": None,
        "jobs": [
            {
                **job,
                "order": index,
                "status": "pending",
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "exit_code": None,
                "output_dir": None,
                "log": str(run_dir / f"{index:02d}_{job['safe_name']}.log"),
                "error": None,
            }
            for index, job in enumerate(jobs, start=1)
        ],
    }
    _atomic_json(run_dir / "queue_state.json", state)
    return state, run_dir


def _load_resume(run_dir_value: str | Path) -> tuple[dict[str, Any], Path]:
    run_dir = Path(run_dir_value).expanduser().resolve()
    state_path = run_dir / "queue_state.json"
    if not state_path.is_file():
        raise ValueError(f"queue_state.json was not found in {run_dir}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("schema_version") != 1 or not isinstance(state.get("jobs"), list):
        raise ValueError(f"unsupported or malformed queue state: {state_path}")
    return state, run_dir


def _experiment_snapshot(backend: str) -> dict[Path, int]:
    directory = EXPERIMENT_LOG_ROOT / backend
    if not directory.is_dir():
        return {}
    return {path.resolve(): path.stat().st_mtime_ns for path in directory.glob("*.json")}


def _completed_output(backend: str, before: dict[Path, int]) -> str:
    after = _experiment_snapshot(backend)
    changed = [path for path, modified in after.items() if path not in before or before[path] != modified]
    if len(changed) != 1:
        raise RuntimeError(f"expected one new {backend} experiment summary, found {len(changed)}")
    summary = json.loads(changed[0].read_text(encoding="utf-8"))
    value = summary.get("output_dir")
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"experiment summary has no output_dir: {changed[0]}")
    output_dir = Path(value).expanduser().resolve()
    if not output_dir.is_dir() or not (output_dir / "test_metrics.json").is_file():
        raise RuntimeError(f"training output is incomplete: {output_dir}")
    return str(output_dir)


def _run_child(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{_now()}] {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert child.stdout is not None
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return child.wait()
        except KeyboardInterrupt:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def run_state(state: dict[str, Any], run_dir: Path, *, continue_on_error: bool = False) -> int:
    state_path = run_dir / "queue_state.json"
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or _now()
    state["finished_at"] = None
    _atomic_json(state_path, state)

    failed = False
    for job in state["jobs"]:
        if job["status"] == "succeeded":
            print(f"Skipping completed job {job['order']}: {job['name']}", flush=True)
            continue
        if job["status"] == "skipped":
            job["status"] = "pending"
        job.update(
            status="running", started_at=_now(), finished_at=None,
            duration_seconds=None, exit_code=None, output_dir=None, error=None,
        )
        _atomic_json(state_path, state)
        print(f"Starting job {job['order']}/{len(state['jobs'])}: {job['name']}", flush=True)
        before = _experiment_snapshot(job["backend"])
        command = [sys.executable, "-m", "src.finetune.train_asr", "--config", job["config"]]
        if state.get("smoke", False):
            command.append("--smoke")
        started = datetime.now(timezone.utc)
        try:
            exit_code = _run_child(command, Path(job["log"]))
            job["exit_code"] = exit_code
            if exit_code:
                raise RuntimeError(f"training process exited with status {exit_code}")
            job["output_dir"] = _completed_output(job["backend"], before)
            job["status"] = "succeeded"
            print(f"Completed job {job['order']}: {job['output_dir']}", flush=True)
        except KeyboardInterrupt:
            job["status"] = "interrupted"
            job["exit_code"] = 130
            job["error"] = "interrupted by user"
            state["status"] = "interrupted"
            raise
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            failed = True
            print(f"Job {job['order']} failed: {exc}", file=sys.stderr, flush=True)
        finally:
            finished = datetime.now(timezone.utc)
            job["finished_at"] = finished.isoformat()
            job["duration_seconds"] = round((finished - started).total_seconds(), 3)
            _atomic_json(state_path, state)

        if failed and state.get("stop_on_failure", True) and not continue_on_error:
            for remaining in state["jobs"]:
                if remaining["status"] == "pending":
                    remaining["status"] = "skipped"
                    remaining["error"] = "not run because an earlier job failed"
            break

    state["status"] = "failed" if any(job["status"] == "failed" for job in state["jobs"]) else "succeeded"
    state["finished_at"] = _now()
    _atomic_json(state_path, state)
    return 1 if state["status"] == "failed" else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR training configurations sequentially")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="Queue YAML to start")
    source.add_argument("--resume", help="Existing queue run directory to resume")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.resume:
        if args.validate_only:
            parser.error("--validate-only cannot be combined with --resume")
        state, run_dir = _load_resume(args.resume)
        if args.smoke and not state.get("smoke", False):
            parser.error("--smoke cannot change the mode of an existing queue run")
        for job in state["jobs"]:
            if job["status"] in {"running", "failed", "interrupted", "skipped"}:
                job["status"] = "pending"
        validate_pending_jobs(state)
        print("Validation passed for all incomplete queued jobs.", flush=True)
        try:
            exit_code = run_state(state, run_dir, continue_on_error=args.continue_on_error)
        except KeyboardInterrupt:
            raise SystemExit(130)
        raise SystemExit(exit_code)

    queue = load_queue(args.config)
    jobs = validate_queue(queue)
    print(f"Validation passed for {len(jobs)} queued jobs.", flush=True)
    if args.validate_only:
        return
    state, run_dir = _new_state(queue, jobs, args.smoke)
    print(f"Queue run: {run_dir}", flush=True)
    try:
        exit_code = run_state(state, run_dir, continue_on_error=args.continue_on_error)
    except KeyboardInterrupt:
        raise SystemExit(130)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
