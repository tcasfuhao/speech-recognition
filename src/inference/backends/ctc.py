from __future__ import annotations

# Standard libraries
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Third-party libraries
import torch

from transformers import AutoModelForCTC, AutoProcessor

# Local imports
from src.evaluation.metrics import normalize_text
from src.inference.backends.base import (
    AlignmentResult,
    AlignmentUnit,
    TranscriptResult,
)


def get_trellis(
    emissions: torch.Tensor,
    tokens: list[int],
    blank_id: int,
) -> torch.Tensor:
    num_frames = emissions.size(0)
    num_tokens = len(tokens)

    trellis = torch.full(
        (num_frames + 1, num_tokens + 1),
        -float("inf"),
    )

    trellis[0, 0] = 0.0

    for t in range(num_frames):
        trellis[t + 1, 0] = trellis[t, 0] + emissions[t, blank_id]

    for t in range(num_frames):
        trellis[t + 1, 1:] = torch.maximum(
            trellis[t, 1:] + emissions[t, blank_id],
            trellis[t, :-1] + emissions[t, tokens],
        )

    return trellis


@dataclass
class CTCBackend:
    model_id: str | Path
    device: str
    dtype: torch.dtype
    lm_path: Optional[str] = None
    lm_weight: float = 0.5
    beam_width: int = 50

    supports_forced_alignment: bool = True
    supports_transcription: bool = True
    supports_transcription_timestamps: bool = False

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=True,
        )

        self.model = AutoModelForCTC.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            token=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.decoder = None

        if self.lm_path:
            from pyctcdecode.decoder import build_ctcdecoder

            vocab_list = self.get_labels().copy()
            vocab_list[self.get_blank_id()] = ""

            tokenizer = self.processor.tokenizer

            if getattr(tokenizer, "word_delimiter_token_id", None) is not None:
                vocab_list[tokenizer.word_delimiter_token_id] = " "

            print("[LM CHECK] vocab size:", len(vocab_list))
            print("[LM CHECK] first 20 labels:", vocab_list[:20])
            print("[LM CHECK] pad_token_id:", self.get_blank_id())
            print(
                "[LM CHECK] word_delimiter_token_id:",
                getattr(tokenizer, "word_delimiter_token_id", None),
            )

            self.decoder = build_ctcdecoder(
                labels=vocab_list,
                kenlm_model_path=self.lm_path,
                alpha=self.lm_weight,
            )

        if self.decoder is not None:
            print(f"[INFO] Using CTC + KenLM ({self.lm_path})")
        else:
            print("[INFO] Using CTC greedy decoding")

    def forced_align(
        self,
        wav_16k: torch.Tensor,
        text: str,
        unit_type: str = "char",
        alignment_mode: str = "partial",
    ) -> AlignmentResult:
        if unit_type not in {"char", "word"}:
            raise ValueError(f"Unsupported unit_type: {unit_type}")

        blank_id = self.get_blank_id()

        transcript = normalize_text(text, strip_punct=False)
        alignment_text = transcript.replace(" ", "")

        tokens = self.transcript_to_tokens(alignment_text)

        emissions = self.get_emissions(wav_16k)
        trellis = get_trellis(emissions, tokens, blank_id=blank_id)

        path = backtrack(
            trellis, emissions, tokens, blank_id=blank_id, mode=alignment_mode
        )
        char_segments = merge_repeats(path, alignment_text)

        if unit_type == "word":
            output_segments = derive_word_segments_from_text(
                char_segments=char_segments,
                transcript=transcript,
            )
        else:
            output_segments = char_segments

        units = segments_to_alignment_units(
            segments=output_segments,
            num_frames=emissions.size(0),
            num_samples=wav_16k.numel(),
            sample_rate=16000,
        )

        return AlignmentResult(
            text=transcript,
            units=units,
            model_id=str(self.model_id),
            unit_type=unit_type,
            metadata={
                "backend_type": "ctc",
                "alignment_type": "forced",
                "alignment_text": alignment_text,
                "spaces_aligned": False,
                "sample_rate": 16000,
                "num_frames": emissions.size(0),
                "num_samples": wav_16k.numel(),
            },
        )

    def get_blank_id(self) -> int:
        blank_id = self.processor.tokenizer.pad_token_id
        if blank_id is None:
            raise ValueError("Tokenizer does not define pad_token_id.")
        return blank_id

    @torch.inference_mode()
    def get_emissions(self, wav_16k: torch.Tensor) -> torch.Tensor:
        logits = self.get_logits(wav_16k)
        emissions = torch.nn.functional.log_softmax(logits, dim=-1)
        return emissions.cpu()

    def get_labels(self) -> list[str]:
        vocab = self.processor.tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        return [token for token, _ in sorted_vocab]

    @torch.inference_mode()
    def get_logits(self, wav_16k: torch.Tensor) -> torch.Tensor:
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

        logits = self.model(**inputs).logits
        return logits[0].cpu()

    @torch.inference_mode()
    def transcribe(self, wav_16k: torch.Tensor) -> TranscriptResult:
        logits = self.get_logits(wav_16k)

        if self.decoder is not None:
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            text = self.decoder.decode(
                log_probs.cpu().numpy(),
                beam_width=self.beam_width,
            )

            decoding = "ctc_beam_search_with_lm"
        else:
            pred_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(pred_ids.unsqueeze(0))[0]
            decoding = "ctc_greedy"

        return TranscriptResult(
            text=normalize_text(text, strip_punct=False),
            model_id=str(self.model_id),
            metadata={
                "backend_type": "ctc",
                "decoding": decoding,
                "sample_rate": 16000,
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

    def transcript_to_tokens(
        self,
        transcript: str,
    ) -> list[int]:
        labels = self.get_labels()
        dictionary = {label: i for i, label in enumerate(labels)}

        missing = sorted({c for c in transcript if c not in dictionary})
        if missing:
            raise ValueError(
                f"Transcript contains characters not in tokenizer vocabulary: {missing}"
            )

        return [dictionary[c] for c in transcript]


@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(
    trellis: torch.Tensor,
    emissions: torch.Tensor,
    tokens: list[int],
    blank_id: int,
    mode: str = "full",
) -> list[Point]:
    if mode not in {"full", "partial"}:
        raise ValueError(f"Unsupported backtracking mode: {mode}")

    j = trellis.size(1) - 1

    if mode == "full":
        t = trellis.size(0) - 1
    else:
        t = int(torch.argmax(trellis[1:, j]).item()) + 1

    path: list[Point] = []

    while t > 0:
        if j == 0:
            if mode == "partial":
                break

            prob = emissions[t - 1, blank_id].exp().item()
            path.append(Point(token_index=0, time_index=t - 1, score=prob))
            t -= 1
            continue

        p_stay = emissions[t - 1, blank_id]
        stayed = trellis[t - 1, j] + p_stay

        token_id = tokens[j - 1]
        p_change = emissions[t - 1, token_id]
        changed = trellis[t - 1, j - 1] + p_change

        if changed > stayed:
            prob = p_change.exp().item()
            path.append(Point(token_index=j, time_index=t - 1, score=prob))
            j -= 1
        else:
            prob = p_stay.exp().item()
            path.append(Point(token_index=j, time_index=t - 1, score=prob))

        t -= 1

    if j != 0:
        raise ValueError(
            "Backtracking failed: not all transcript tokens were consumed. "
            f"Remaining token index: {j}"
        )

    return path[::-1]


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start


def derive_word_segments_from_text(
    char_segments: list[Segment],
    transcript: str,
) -> list[Segment]:
    words: list[Segment] = []
    char_index = 0

    for word_text in transcript.split():
        word_len = len(word_text)
        word_chars = char_segments[char_index : char_index + word_len]

        if len(word_chars) != word_len:
            raise ValueError(
                "Not enough character segments to reconstruct word spans."
            )

        score = (
            sum(seg.score * seg.length for seg in word_chars)
            / sum(seg.length for seg in word_chars)
        )

        words.append(
            Segment(
                label=word_text,
                start=word_chars[0].start,
                end=word_chars[-1].end,
                score=score,
            )
        )

        char_index += word_len

    if char_index != len(char_segments):
        raise ValueError(
            "Unused character segments remain after reconstructing word spans."
        )

    return words


def merge_repeats(
    path: list[Point],
    transcript: str,
) -> list[Segment]:
    segments: list[Segment] = []

    i1 = 0
    while i1 < len(path):
        i2 = i1

        while (
            i2 < len(path)
            and path[i2].token_index == path[i1].token_index
        ):
            i2 += 1

        token_index = path[i1].token_index

        if token_index > 0:
            label = transcript[token_index - 1]
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)

            segments.append(
                Segment(
                    label=label,
                    start=path[i1].time_index,
                    end=path[i2 - 1].time_index + 1,
                    score=score,
                )
            )

        i1 = i2

    return segments


def segments_to_alignment_units(
    segments: list[Segment],
    num_frames: int,
    num_samples: int,
    sample_rate: int = 16000,
) -> list[AlignmentUnit]:
    audio_duration_ms = num_samples / sample_rate * 1000
    frame_duration_ms = audio_duration_ms / num_frames

    units: list[AlignmentUnit] = []

    for seg in segments:
        units.append(
            AlignmentUnit(
                label=seg.label,
                start_ms=round(seg.start * frame_duration_ms),
                end_ms=round(seg.end * frame_duration_ms),
                score=seg.score,
            )
        )

    return units
