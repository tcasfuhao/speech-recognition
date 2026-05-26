from __future__ import annotations

# Standard libraries
from dataclasses import dataclass
from pathlib import Path

# Third-party libraries
import torch

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Local imports
from src.evaluation.metrics import normalize_text
from src.inference.backends.base import (
    AlignmentResult,
    TranscriptResult,
)


@dataclass
class Seq2SeqBackend:
    model_id: str | Path
    device: str
    dtype: torch.dtype
    max_new_tokens: int = 256

    supports_forced_alignment: bool = False
    supports_transcription: bool = True
    supports_transcription_timestamps: bool = False

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=True,
        )

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            token=True,
        )

        self.model.to(self.device)
        self.model.eval()

        print("[INFO] Using seq2seq model")

    def forced_align(
        self,
        wav_16k: torch.Tensor,
        text: str,
        unit_type: str = "char",
    ) -> AlignmentResult:
        raise NotImplementedError(
            "Forced alignment is not supported by this backend."
        )

    @torch.inference_mode()
    def transcribe(self, wav_16k: torch.Tensor) -> TranscriptResult:
        inputs = self.processor(
            wav_16k.cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )

        inputs = {
            k: (
                v.to(self.device, dtype=self.dtype)
                if torch.is_floating_point(v)
                else v.to(self.device)
            )
            for k, v in inputs.items()
        }

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            task="transcribe",
        )

        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0]

        return TranscriptResult(
            text=normalize_text(text, strip_punct=False),
            model_id=str(self.model_id),
            metadata={
                "backend_type": "seq2seq",
                "decoding": "generate",
                "sample_rate": 16000,
                "max_new_tokens": self.max_new_tokens,
            },
        )

    def transcribe_with_timestamps(
        self,
        wav_16k: torch.Tensor,
        unit_type: str = "char",
    ) -> TranscriptResult:
        raise NotImplementedError(
            "Recognition-time timestamps are not supported by this backend yet."
        )
