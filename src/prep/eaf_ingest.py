from __future__ import annotations

# Standard libraries
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# Third-party libraries
import pandas as pd
try:  # Keep selector/path helpers testable without the optional audio stack.
    import pympi
except ModuleNotFoundError:  # pragma: no cover - guarded again at ingestion.
    pympi = None

try:
    from pydub import AudioSegment
except ModuleNotFoundError:  # pragma: no cover - guarded again at ingestion.
    AudioSegment = None


DEFAULT_AUDIO_EXTS = {".wav", ".mp3", ".mp4", ".m4a", ".flac"}


@dataclass(frozen=True)
class IngestConfig:
    annotations_dir: str
    audio_root: str
    clips_dir: str
    include_tier_regex: Optional[str] = None
    exclude_tier_regex: Optional[str] = None
    exclude_annotation_path_regex: Optional[str] = None
    merge_same_time_annotations: bool = False
    annotation_joiner: str = ""
    min_dur_ms: int = 200
    max_dur_ms: int = 30000
    session_id_from: str = "parent_dir"  # "parent_dir" | "recording_id"
    audio_exts: Tuple[str, ...] = tuple(DEFAULT_AUDIO_EXTS)


def slugify_tier(tier_name: str) -> str:
    s = tier_name.strip().replace(" ", "-")
    s = re.sub(r"[^A-Za-z0-9_\-]+", "", s)
    return s or "tier"


def find_audio_for_eaf(
    eaf_path: Path,
    audio_root: Path,
    audio_exts: Iterable[str],
) -> Optional[Path]:
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

    # Normalised EAFs can live in a separate timestamped tree. Fall back to
    # the configured data root, where the original recording audio remains.
    allowed_exts = {ext.casefold() for ext in audio_exts}
    direct_matches = [
        path
        for path in audio_root.glob(f"{eaf_path.stem}.*")
        if path.is_file() and path.suffix.casefold() in allowed_exts
    ]
    if len(direct_matches) == 1:
        return direct_matches[0]

    matches = [
        path
        for path in audio_root.rglob(f"{eaf_path.stem}.*")
        if path.is_file()
        and path.suffix.casefold() in allowed_exts
        and not {"processed", "normalised", "undone"}.intersection(path.parts)
    ]
    matches = sorted({path.resolve() for path in matches if path.is_file()})
    if len(matches) == 1:
        return matches[0]
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
        for annotation in annots:
            # pympi returns three fields for alignable annotations and may
            # append a reference identifier for dependent tiers.
            _, _, text = annotation[:3]
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


def find_eaf_files(
    annotations_dir: Path, *, exclude_path_regex: Optional[str] = None
) -> List[Path]:
    """Find EAFs deterministically, optionally excluding copied annotation trees."""
    eaf_files = sorted(annotations_dir.rglob("*.eaf"))
    if exclude_path_regex:
        path_re = re.compile(exclude_path_regex, re.IGNORECASE)
        eaf_files = [path for path in eaf_files if not path_re.search(str(path))]
    return eaf_files


def has_annotation_text(text: object) -> bool:
    """Whether an edition retained a usable annotation after normalisation."""
    return bool("" if text is None else str(text).strip())


def tier_annotations(
    eaf_obj: pympi.Elan.Eaf,
    tier_name: str,
    *,
    merge_same_time: bool = False,
    joiner: str = "",
) -> List[Tuple[int, int, str]]:
    """Return alignable rows, optionally joining reference-tier words."""
    rows = [
        tuple(annotation[:3])
        for annotation in eaf_obj.get_annotation_data_for_tier(tier_name)
    ]
    if not merge_same_time:
        return rows

    grouped: dict[Tuple[int, int], List[str]] = {}
    for start_ms, end_ms, text in rows:
        key = (start_ms, end_ms)
        if has_annotation_text(text):
            grouped.setdefault(key, []).append(str(text).strip())
        else:
            grouped.setdefault(key, [])
    return [
        (start_ms, end_ms, joiner.join(parts))
        for (start_ms, end_ms), parts in grouped.items()
    ]


def ingest_eaf_directory(cfg: IngestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pympi is None or AudioSegment is None:
        raise ModuleNotFoundError(
            "EAF ingestion requires pympi-ling and pydub; install requirements.txt "
            "in the shared ASR environment."
        )
    annotations_dir = Path(cfg.annotations_dir).expanduser()
    audio_root = Path(cfg.audio_root).expanduser()
    clips_dir = Path(cfg.clips_dir).expanduser()
    wav_dir = clips_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    seg_infos: List[dict] = []
    skip_infos: List[dict] = []

    eaf_files = find_eaf_files(
        annotations_dir, exclude_path_regex=cfg.exclude_annotation_path_regex
    )
    if not eaf_files:
        raise ValueError(f"No .eaf files found under {annotations_dir}")

    for eaf_path in eaf_files:
        audio_path = find_audio_for_eaf(eaf_path, audio_root, cfg.audio_exts)
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

        recording_id = eaf_path.stem
        session_id = _session_id_from_path(eaf_path, cfg.session_id_from)

        for tier_name in tiers:
            for start_ms, end_ms, text in tier_annotations(
                eaf_obj,
                tier_name,
                merge_same_time=cfg.merge_same_time_annotations,
                joiner=cfg.annotation_joiner,
            ):
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

                source_text = "" if text is None else str(text)
                if not has_annotation_text(source_text):
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
                        "segment_path": str(seg_path.relative_to(clips_dir)),
                        "text": source_text,
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
