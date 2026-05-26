# Standard libraries
import os

from pathlib import Path
from typing import Optional


def expand_path(p):
    return str(Path(p).expanduser()) if p else p


def resolve_audio_path(p: str, audio_root: Optional[str]) -> str:
    if os.path.isabs(p):
        return p
    if audio_root is not None:
        return str(Path(audio_root) / p)
    return p
