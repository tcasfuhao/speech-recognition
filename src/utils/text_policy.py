from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


TEXT_POLICY_FILENAME = "asr_text_config.json"


def write_text_policy(model_dir: str | Path, remove_spaces: bool) -> Path:
    """Store the ASR text policy beside a saved model."""
    target = Path(model_dir) / TEXT_POLICY_FILENAME
    target.write_text(
        json.dumps({"remove_spaces": bool(remove_spaces)}, indent=2),
        encoding="utf-8",
    )
    return target


def resolve_remove_spaces(
    model_id_or_path: str | Path,
    override: Optional[bool] = None,
) -> bool:
    """Resolve an inference override, saved model policy, or the safe default."""
    if override is not None:
        if not isinstance(override, bool):
            raise ValueError("remove_spaces must be true, false, or null")
        return override

    model_path = Path(model_id_or_path).expanduser()
    candidates = (
        model_path / TEXT_POLICY_FILENAME,
        model_path / "config.json",
        model_path.parent / TEXT_POLICY_FILENAME,
        model_path.parent / "run_config.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        value = payload.get("remove_spaces")
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ValueError(f"{candidate}: remove_spaces must be true or false")
        return value
    return True
