from __future__ import annotations

# Standard libraries
from typing import List, Optional

# Third-party libraries
import pandas as pd


DUR_COL_CANDIDATES = [
    "dur_ms", "dur_msec", "dur_sec",
    "duration_ms", "duration_msec", "duration_sec",
]

END_COL_CANDIDATES = [
    "end", "end_ms", "end_msec",
]

HYP_COL_CANDIDATES = [
    "hyp", "hypothesis", "pred", "prediction", "targ", "target",
    "hyp_text", "hypothesis_text", "pred_text", "prediction_text",
    "targ_text", "target_text",
]

LOCATION_COL_CANDIDATES = ["location", "village", "site"]

MODEL_COL_CANDIDATES = [
    "asr_model", "model",
    "asr_id", "model_id",
]

PATH_COL_CANDIDATES = [
    "audio", "file", "path", "rec", "recording", "seg", "segment", "source",
    "src", "utt", "utterance", "wav",
    "audio_path", "file_path", "rec_path", "recording_path", "seg_path",
    "segment_path", "source_path", "src_path", "utt_path", "utterance_path",
    "wav_path",
]

RECORDING_COL_CANDIDATES = [
    "audio_id", "file_id", "rec_id", "recording_id", "seg_id", "segment_id",
    "source_id", "src_id", "utt_id", "utterance_id", "wav_id",
]

SESSION_COL_CANDIDATES = [
    "rec_date", "recording_date", "sess", "session",
    "sess_id", "session_id",
]

SPEAKER_COL_CANDIDATES = [
    "speaker", "spk", "spkr",
    "speaker_id", "spk_id", "spkr_id",
]

START_COL_CANDIDATES = [
    "start", "start_ms", "start_msec",
]

TEXT_COL_CANDIDATES = [
    "ref", "reference", "sentence", "text", "transcript",
    "ref_text", "reference_text",
]


def pick_col(
    df: pd.DataFrame,
    candidates: List[str],
    *,
    required: bool = True,
) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}

    for cand in candidates:
        key = cand.lower()
        if key in cols:
            return cols[key]

    if required:
        raise ValueError(
            f"None of these columns exist: {candidates}. "
            f"Available: {list(df.columns)}"
        )

    return None


def session_id_from_recording_id(recording_id: str) -> str:
    if recording_id and len(recording_id) > 1:
        return recording_id[:-1]

    return recording_id
