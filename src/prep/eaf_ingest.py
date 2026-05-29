from __future__ import annotations

# Standard libraries
import re
import unicodedata

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Third-party libraries
import pandas as pd
import pympi

from pydub import AudioSegment


DEFAULT_AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".flac"}


@dataclass(frozen=True)
class IngestConfig:
    raw_dir: str
    out_dir: str
    include_tier_regex: Optional[str] = None
    exclude_tier_regex: Optional[str] = None
    min_dur_ms: int = 200
    max_dur_ms: int = 30000
    lowercase: bool = True
    strip_punct: bool = False
    keep_apostrophe: bool = True
    remove_bracketed: bool = True
    remove_diacritics: bool = False
    collapse_whitespace: bool = True
    session_id_from: str = "parent_dir"  # "parent_dir" | "recording_id"
    audio_exts: Tuple[str, ...] = tuple(DEFAULT_AUDIO_EXTS)
    normalize_quotes: bool = True
    remove_hash_and_question: bool = True
    map_numeric_speaker_codes: bool = False

_SINGLE_CHAR_TRANSLATION = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "“": "''",
        "”": "''",
    }
)


def normalize_text(
    s: str,
    *,
    lowercase: bool,
    strip_punct: bool,
    keep_apostrophe: bool,
    remove_bracketed: bool,
    remove_diacritics: bool,
    collapse_whitespace: bool,
    normalize_quotes: bool,
    remove_hash_and_question: bool,
    map_numeric_speaker_codes: bool,

) -> str:
    if s is None:
        return ""
    s = str(s).strip()

    if lowercase:
        s = s.lower()
    
    if normalize_quotes:
        s = s.translate(_SINGLE_CHAR_TRANSLATION)

    if remove_bracketed:
        s = re.sub(r"\[[^\]]*\]", " ", s)
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"[\[\]\(\)]", " ", s)

    if remove_hash_and_question:
        s = s.replace("#", "")
        s = s.replace("?", "")
    
    if map_numeric_speaker_codes:
        s = s.replace("33", "M").replace("35", "R").replace("55", "H").replace("53", "F")


    if remove_diacritics:
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")

    if strip_punct:
        if keep_apostrophe:
            s = re.sub(r"[^\w\s']", " ", s)
        else:
            s = re.sub(r"[^\w\s]", " ", s)

    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s).strip()

    return s


def slugify_tier(tier_name: str) -> str:
    s = tier_name.strip().replace(" ", "-")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "tier"


def find_audio_for_eaf(eaf_path: Path, audio_exts: Iterable[str]) -> Optional[Path]:
    base = eaf_path.with_suffix("")
    for ext in audio_exts:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate

    audio_files = [
        p for p in eaf_path.parent.iterdir()
        if p.is_file() and p.suffix.lower() in audio_exts
    ]
    if len(audio_files) == 1:
        return audio_files[0]
    return None


def detect_speech_tiers(
    eaf_obj: pympi.Elan.Eaf,
    *,
    include_regex: Optional[str] = None,
    exclude_regex: Optional[str] = None,
    min_chars: int = 1,
) -> List[str]:
    include_re = re.compile(include_regex, re.IGNORECASE) if include_regex else None
    exclude_re = re.compile(exclude_regex, re.IGNORECASE) if exclude_regex else None

    tiers = []
    for tier_name in eaf_obj.get_tier_names():
        if include_re and not include_re.search(tier_name):
            continue
        if exclude_re and exclude_re.search(tier_name):
            continue

        annots = eaf_obj.get_annotation_data_for_tier(tier_name)
        speech_like = 0
        for _, _, text in annots:
            if text is None:
                continue
            if len(re.findall(r"\w", str(text))) >= min_chars:
                speech_like += 1
                break

        if speech_like > 0:
            tiers.append(tier_name)

    return tiers


def _session_id_from_path(eaf_path: Path, mode: str) -> str:
    if mode == "parent_dir":
        return eaf_path.parent.name
    if mode == "recording_id":
        return eaf_path.stem
    raise ValueError(f"Unknown session_id_from: {mode}")


def ingest_eaf_directory(cfg: IngestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(cfg.raw_dir).expanduser()
    out_dir = Path(cfg.out_dir).expanduser()
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    seg_infos: List[dict] = []
    skip_infos: List[dict] = []

    eaf_files = list(raw_dir.rglob("*.eaf"))
    if not eaf_files:
        raise ValueError(f"No .eaf files found under {raw_dir}")

    for eaf_path in eaf_files:
        audio_path = find_audio_for_eaf(eaf_path, cfg.audio_exts)
        if audio_path is None:
            skip_infos.append(
                {
                    "eaf_path": str(eaf_path),
                    "recording_id": eaf_path.stem,
                    "tier_name": "",
                    "start_ms": "",
                    "end_ms": "",
                    "text": "",
                    "reason": "audio_not_found",
                }
            )
            continue

        eaf_obj = pympi.Elan.Eaf(str(eaf_path))
        tiers = detect_speech_tiers(
            eaf_obj,
            include_regex=cfg.include_tier_regex,
            exclude_regex=cfg.exclude_tier_regex,
        )

        if not tiers:
            skip_infos.append(
                {
                    "eaf_path": str(eaf_path),
                    "recording_id": eaf_path.stem,
                    "tier_name": "",
                    "start_ms": "",
                    "end_ms": "",
                    "text": "",
                    "reason": "no_speech_tiers_detected",
                }
            )
            continue

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        recording_id = eaf_path.stem
        session_id = _session_id_from_path(eaf_path, cfg.session_id_from)

        for tier_name in tiers:
            for start_ms, end_ms, text in eaf_obj.get_annotation_data_for_tier(tier_name):
                if start_ms is None or end_ms is None:
                    skip_infos.append(
                        {
                            "eaf_path": str(eaf_path),
                            "recording_id": recording_id,
                            "tier_name": tier_name,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "text": text,
                            "reason": "missing_timecodes",
                        }
                    )
                    continue

                dur_ms = int(end_ms) - int(start_ms)
                if dur_ms < cfg.min_dur_ms:
                    skip_infos.append(
                        {
                            "eaf_path": str(eaf_path),
                            "recording_id": recording_id,
                            "tier_name": tier_name,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "text": text,
                            "reason": "too_short",
                        }
                    )
                    continue
                if dur_ms > cfg.max_dur_ms:
                    skip_infos.append(
                        {
                            "eaf_path": str(eaf_path),
                            "recording_id": recording_id,
                            "tier_name": tier_name,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "text": text,
                            "reason": "too_long",
                        }
                    )
                    continue

                norm_text = normalize_text(
                    text,
                    lowercase=cfg.lowercase,
                    strip_punct=cfg.strip_punct,
                    keep_apostrophe=cfg.keep_apostrophe,
                    remove_bracketed=cfg.remove_bracketed,
                    remove_diacritics=cfg.remove_diacritics,
                    collapse_whitespace=cfg.collapse_whitespace,
                    normalize_quotes=cfg.normalize_quotes,
                    remove_hash_and_question=cfg.remove_hash_and_question,
                    map_numeric_speaker_codes=cfg.map_numeric_speaker_codes,
                )
                if not norm_text:
                    skip_infos.append(
                        {
                            "eaf_path": str(eaf_path),
                            "recording_id": recording_id,
                            "tier_name": tier_name,
                            "start_ms": start_ms,
                            "end_ms": end_ms,
                            "text": text,
                            "reason": "empty_text",
                        }
                    )
                    continue

                tier_slug = slugify_tier(tier_name)
                seg_name = f"{recording_id}_{tier_slug}_{int(start_ms):07d}_{int(end_ms):07d}.wav"
                seg_path = wav_dir / seg_name

                segment = audio[int(start_ms):int(end_ms)]
                segment.export(seg_path, format="wav")

                seg_infos.append(
                    {
                        "segment_path": str(seg_path.relative_to(out_dir)),
                        "text": norm_text,
                        "recording_id": recording_id,
                        "session_id": session_id,
                        "tier_name": tier_name,
                        "start_ms": int(start_ms),
                        "end_ms": int(end_ms),
                        "dur_ms": int(dur_ms),
                        "speaker_id": tier_name,
                    }
                )

    seg_df = pd.DataFrame(seg_infos)
    skip_df = pd.DataFrame(skip_infos)

    return seg_df, skip_df
