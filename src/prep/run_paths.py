from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


_RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?$")


def new_prep_run_id(now: datetime | None = None) -> str:
    """Return a sortable, collision-resistant preparation timestamp."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")


def validate_prep_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "prep_run_id must use YYYYMMDD_HHMMSS or "
            f"YYYYMMDD_HHMMSS_microseconds, got {run_id!r}"
        )
    return run_id


def timestamped_prep_dir(configured_dir: Path, run_id: str) -> Path:
    """Place a run timestamp at the batch level or on an individual run."""
    run_id = validate_prep_run_id(run_id)
    parts = configured_dir.parts
    comparison_indexes = [
        index for index, component in enumerate(parts) if component == "comparison"
    ]
    if comparison_indexes:
        index = comparison_indexes[-1]
        return Path(*parts[: index + 1]) / run_id / Path(*parts[index + 1 :])
    return configured_dir.with_name(f"{configured_dir.name}_{run_id}")
