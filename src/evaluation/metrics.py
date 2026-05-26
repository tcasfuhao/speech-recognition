from __future__ import annotations

# Standard libraries
import re

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")


def normalize_text(
    s: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = False,
    keep_apostrophe: bool = True,
    collapse_whitespace: bool = True,
) -> str:
    if s is None:
        return ""
    s = s.strip()
    if lowercase:
        s = s.lower()
    if strip_punct:
        if keep_apostrophe:
            s = re.sub(r"[^\w\s']", "", s)
        else:
            s = re.sub(r"[^\w\s]", "", s)
    if collapse_whitespace:
        s = _WS_RE.sub(" ", s).strip()
    return s


def to_char_sequence(s: str, *, remove_whitespace: bool = True) -> str:
    if remove_whitespace:
        return re.sub(r"\s+", "", s)
    return s


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def cer(
    ref: str,
    hyp: str,
    *,
    lowercase: bool = True,
    strip_punct: bool = False,
    remove_whitespace: bool = True,
    empty_ref_policy: str = "skip",
) -> Optional[float]:
    ref_n = normalize_text(ref, lowercase=lowercase, strip_punct=strip_punct)
    hyp_n = normalize_text(hyp, lowercase=lowercase, strip_punct=strip_punct)

    ref_c = to_char_sequence(ref_n, remove_whitespace=remove_whitespace)
    hyp_c = to_char_sequence(hyp_n, remove_whitespace=remove_whitespace)

    if len(ref_c) == 0:
        if empty_ref_policy == "skip":
            return None
        if empty_ref_policy == "zero":
            return 0.0 if len(hyp_c) == 0 else 1.0
        raise ValueError("Empty reference after preprocessing.")

    dist = levenshtein_distance(ref_c, hyp_c)
    return dist / len(ref_c)


@dataclass(frozen=True)
class ScoredItem:
    ref: str
    hyp: str
    meta: Dict[str, str]


def score_items_cer(
    items: Iterable[ScoredItem],
    *,
    lowercase: bool = True,
    strip_punct: bool = False,
    remove_whitespace: bool = True,
    empty_ref_policy: str = "skip",
) -> List[Tuple[ScoredItem, Optional[float]]]:
    out: List[Tuple[ScoredItem, Optional[float]]] = []
    for it in items:
        v = cer(
            it.ref,
            it.hyp,
            lowercase=lowercase,
            strip_punct=strip_punct,
            remove_whitespace=remove_whitespace,
            empty_ref_policy=empty_ref_policy,
        )
        out.append((it, v))
    return out


def aggregate_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def group_mean(
    scored: Iterable[Tuple[ScoredItem, Optional[float]]],
    group_key: str,
) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[Optional[float]]] = {}
    for item, v in scored:
        k = item.meta.get(group_key, "UNKNOWN")
        buckets.setdefault(k, []).append(v)
    return {k: aggregate_mean(vs) for k, vs in buckets.items()}
