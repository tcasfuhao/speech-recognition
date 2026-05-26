from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import torch


@dataclass
class AlignmentResult:
    text: str
    units: list[AlignmentUnit]
    model_id: str
    unit_type: str = "char"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentUnit:
    label: str
    start_ms: int
    end_ms: int
    score: float | None = None


class ASRBackend(Protocol):
    model_id: str | Path
    device: str
    dtype: torch.dtype

    supports_forced_alignment: bool
    supports_transcription: bool
    supports_transcription_timestamps: bool

    def forced_align(
        self,
        wav_16k: torch.Tensor,
        text: str,
        unit_type: str = "char",
    ) -> AlignmentResult:
        ...

    def transcribe(self, wav_16k: torch.Tensor) -> TranscriptResult:
        ...

    def transcribe_with_timestamps(
        self,
        wav_16k: torch.Tensor,
        unit_type: str = "char",
    ) -> TranscriptResult:
        ...


@dataclass
class TranscriptResult:
    text: str
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
