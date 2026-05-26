from __future__ import annotations

# Standard libraries
from typing import Optional

# Third-party libraries
import torch

from src.inference.backends.base import ASRBackend
from src.inference.backends.ctc import CTCBackend
from src.inference.backends.seq2seq import Seq2SeqBackend


def load_asr_backend(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    lm_path: Optional[str] = None,
    lm_weight: float = 0.5,
    beam_width: int = 50,
) -> ASRBackend:
    try:
        return Seq2SeqBackend(
            model_id=model_id,
            device=device,
            dtype=dtype,
        )
    except Exception:
        return CTCBackend(
            model_id=model_id,
            device=device,
            dtype=dtype,
            lm_path=lm_path,
            lm_weight=lm_weight,
            beam_width=beam_width,
        )
