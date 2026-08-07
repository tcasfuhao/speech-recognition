from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data import schema
from src.evaluation.metrics import strip_whitespace
from src.utils.io import resolve_audio_path


ASPIRATE_MAP = {
    "þ": "tʰ",
    "ƙ": "kʰ",
    "ƥ": "pʰ",
    "ƾ": "tsʰ",
    "ƪ": "tʂʰ",
    "ƫ": "tɕʰ",
}
TONE_MAP = {"M": ("˧",), "H": ("˥",), "R": ("˧", "˥"), "F": ("˥", "˧")}


def read_phone_ids(phone_file: str | Path) -> dict[str, int]:
    result = {}
    for line in Path(phone_file).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields:
            result[fields[0]] = int(fields[1]) if len(fields) > 1 else len(result) + 1
    return result


def tokenize_ipa(text: str, phones: Iterable[str]) -> tuple[list[str], list[dict]]:
    """Tokenize compact IPA with greedy longest-match and explicit source offsets."""
    phone_set = set(phones)
    candidates = sorted(phone_set, key=lambda item: (-len(item), item))
    source = unicodedata.normalize("NFC", str(text))
    tokens: list[str] = []
    unsupported: list[dict] = []
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char in ASPIRATE_MAP:
            value = ASPIRATE_MAP[char]
            if value in phone_set:
                tokens.append(value)
            else:
                unsupported.append({"token": value, "source": char, "offset": index})
            index += 1
            continue
        if char in TONE_MAP:
            for value in TONE_MAP[char]:
                if value in phone_set:
                    tokens.append(value)
                else:
                    unsupported.append({"token": value, "source": char, "offset": index})
            index += 1
            continue
        match = next((phone for phone in candidates if source.startswith(phone, index)), None)
        if match is not None:
            tokens.append(match)
            index += len(match)
            continue
        unsupported.append({"token": char, "source": char, "offset": index})
        index += 1
    return tokens, unsupported


def utterance_id(split_name: str, audio_path: str, row_number: int) -> str:
    digest = hashlib.sha1(f"{split_name}\0{audio_path}\0{row_number}".encode()).hexdigest()[:12]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(audio_path).stem)[:48]
    return f"{split_name}-{stem}-{digest}"


def prepare_manifests(config: dict, run_name: str) -> tuple[Path, Path, dict[str, list[dict]]]:
    if config.get("remove_spaces", True) is False:
        raise ValueError(
            "Allosaurus cannot preserve word-boundary spaces in phone-label targets"
        )
    root = Path(config["allosaurus_root"]).expanduser().resolve()
    model_path = root / "allosaurus" / "pretrained" / "uni2005"
    phone_ids = read_phone_ids(model_path / "phone.txt")
    local_root = Path(config.get("manifest_dir", "logs/allosaurus/manifests")) / run_name
    work_root = Path(config["work_dir"]).expanduser() / run_name
    unsupported_rows: list[dict] = []
    records: dict[str, list[dict]] = {}
    observed: set[str] = set()
    all_ids: set[str] = set()
    all_paths: dict[str, set[str]] = {}

    for csv_name, native_name in (("train", "train"), ("dev", "validate"), ("test", "test")):
        frame = pd.read_csv(Path(config[f"{csv_name}_csv"]).expanduser())
        if csv_name == "train" and config.get("max_train_samples"):
            frame = frame.head(int(config["max_train_samples"]))
        if csv_name != "train" and config.get("max_eval_samples"):
            frame = frame.head(int(config["max_eval_samples"]))
        path_col = schema.pick_col(frame, schema.PATH_COL_CANDIDATES)
        text_col = schema.pick_col(frame, schema.TEXT_COL_CANDIDATES)
        split_records = []
        split_paths = set()
        for row_number, row in frame.iterrows():
            audio = str(Path(resolve_audio_path(str(row[path_col]), config.get("audio_root"))).resolve())
            uid = utterance_id(csv_name, audio, int(row_number))
            tokens, unsupported = tokenize_ipa(str(row[text_col]), phone_ids)
            if uid in all_ids:
                raise ValueError(f"duplicate generated utterance ID: {uid}")
            all_ids.add(uid)
            split_paths.add(audio)
            observed.update(tokens)
            for issue in unsupported:
                unsupported_rows.append(
                    {"split": csv_name, "utterance_id": uid, "text": row[text_col], **issue}
                )
            split_records.append({
                "utterance_id": uid,
                "audio": audio,
                "phones": tokens,
                "reference": strip_whitespace(row[text_col]),
            })
        all_paths[csv_name] = split_paths
        records[native_name] = split_records

        target = local_root / native_name
        target.mkdir(parents=True, exist_ok=True)
        (target / "wave").write_text(
            "".join(f"{item['utterance_id']} {item['audio']}\n" for item in split_records), encoding="utf-8"
        )
        (target / "text").write_text(
            "".join(f"{item['utterance_id']} {' '.join(item['phones'])}\n" for item in split_records), encoding="utf-8"
        )

    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = all_paths[left] & all_paths[right]
        if overlap:
            raise ValueError(f"Allosaurus {left}/{right} sets overlap: {len(overlap)} files")

    inventory = local_root / "target_phone_inventory.txt"
    inventory.write_text(
        "".join(f"{phone} {phone_ids[phone]}\n" for phone in sorted(observed, key=phone_ids.get)), encoding="utf-8"
    )
    unsupported_frame = pd.DataFrame(
        unsupported_rows,
        columns=["split", "utterance_id", "text", "token", "source", "offset"],
    )
    unsupported_frame.to_csv(local_root / "unsupported_tokens.csv", index=False)
    if not unsupported_frame.empty:
        unsupported_frame.groupby(["source", "token"], dropna=False).size().rename("occurrences").reset_index().sort_values(
            "occurrences", ascending=False
        ).to_csv(local_root / "unsupported_token_summary.csv", index=False)
    summary = {
        "rows": {key: len(value) for key, value in records.items()},
        "unique_utterance_ids": len(all_ids),
        "target_phones": len(observed),
        "unsupported_tokens": len(unsupported_rows),
    }
    (local_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if unsupported_rows:
        raise ValueError(
            f"{len(unsupported_rows)} labels cannot be represented by uni2005; "
            f"see {local_root / 'unsupported_tokens.csv'} and unsupported_token_summary.csv"
        )

    for native_name in ("train", "validate"):
        work_split = work_root / native_name
        work_split.mkdir(parents=True, exist_ok=True)
        for filename in ("wave", "text"):
            link = work_split / filename
            if not link.exists():
                link.symlink_to((local_root / native_name / filename).resolve())
    return local_root, work_root, records


def run_feature_preparation(config: dict, work_root: Path) -> None:
    root = Path(config["allosaurus_root"]).expanduser().resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    inventory = Path(config.get("manifest_dir", "logs/allosaurus/manifests")) / work_root.name / "target_phone_inventory.txt"
    for split_name in ("train", "validate"):
        split_path = work_root / split_name
        subprocess.run([sys.executable, "-m", "allosaurus.bin.prep_feat", "--model", "uni2005", "--path", str(split_path)], cwd=root, env=env, check=True)
        subprocess.run([sys.executable, "-m", "allosaurus.bin.prep_token", "--model", "uni2005", "--lang", str(inventory.resolve()), "--path", str(split_path)], cwd=root, env=env, check=True)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, 1):
        current = [row]
        for col, hyp in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[col] + 1, previous[col - 1] + (ref != hyp)))
        previous = current
    return previous[-1]
