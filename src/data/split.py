from __future__ import annotations

# Standard libraries
import json
import random

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Third-party libraries
import pandas as pd


DURATION_SPLIT_SEED_OFFSETS = {
    "train": 0,
    "dev": 1,
    "test": 2,
}


@dataclass(frozen=True)
class SplitRatios:
    train: float
    dev: float
    test: float

    @property
    def total(self) -> float:
        return self.train + self.dev + self.test

    def normalized(self) -> "SplitRatios":
        if self.total <= 0:
            raise ValueError("Split ratios must sum to > 0.")
        return SplitRatios(
            train=self.train / self.total,
            dev=self.dev / self.total,
            test=self.test / self.total,
        )


def _count_splits(n: int, ratios: SplitRatios) -> Tuple[int, int, int]:
    ratios = ratios.normalized()
    train_n = int(n * ratios.train)
    dev_n = int(n * ratios.dev)
    test_n = n - train_n - dev_n
    return train_n, dev_n, test_n


def split_rows(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    ratios: SplitRatios,
    min_per_split: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    train_n, dev_n, test_n = _count_splits(len(df), ratios)

    if min(train_n, dev_n, test_n) < min_per_split:
        raise ValueError(
            f"Split sizes too small for ratios {ratios} on {len(df)} rows. "
            "Either add more data or adjust ratios."
        )

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df = df.iloc[:train_n].copy()
    dev_df = df.iloc[train_n : train_n + dev_n].copy()
    test_df = df.iloc[train_n + dev_n :].copy()

    return train_df, dev_df, test_df


def split_by_group(
    df: pd.DataFrame,
    *,
    split_col: str,
    seed: int = 42,
    ratios: SplitRatios,
    min_per_split: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if split_col not in df.columns:
        raise ValueError(
            f"split_col '{split_col}' not found in columns: {list(df.columns)}"
        )

    groups = sorted(df[split_col].dropna().unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(groups)

    train_n, dev_n, test_n = _count_splits(len(groups), ratios)
    if min(train_n, dev_n, test_n) < min_per_split:
        raise ValueError(
            f"Not enough unique '{split_col}' values ({len(groups)}) "
            f"for ratios {ratios}. Use row-level splitting or adjust ratios."
        )

    train_set = set(groups[:train_n])
    dev_set = set(groups[train_n : train_n + dev_n])
    test_set = set(groups[train_n + dev_n :])

    train_df = df[df[split_col].isin(train_set)].copy()
    dev_df = df[df[split_col].isin(dev_set)].copy()
    test_df = df[df[split_col].isin(test_set)].copy()

    return train_df, dev_df, test_df


def duration_seconds(df: pd.DataFrame, dur_col: str) -> pd.Series:
    """Return a manifest duration column converted to seconds."""
    if dur_col not in df.columns:
        raise ValueError(
            f"Duration column '{dur_col}' not found in columns: {list(df.columns)}"
        )

    durations = pd.to_numeric(df[dur_col], errors="raise").astype(float)
    if durations.isna().any():
        raise ValueError(f"Duration column '{dur_col}' contains missing values.")
    if (durations < 0).any():
        raise ValueError(f"Duration column '{dur_col}' contains negative values.")

    if dur_col.lower().endswith(("ms", "msec")):
        durations = durations / 1000.0
    return durations


def select_duration_budget(
    df: pd.DataFrame,
    *,
    dur_col: str,
    target_duration_sec: Optional[float],
    seed: int,
) -> pd.DataFrame:
    """Deterministically select shuffled rows through the target duration.

    The row that reaches or crosses the target is retained. A null target, or a
    target at least as large as the available audio, returns the split unchanged.
    """
    if target_duration_sec is None:
        return df.copy()
    if target_duration_sec <= 0:
        raise ValueError("target_duration_sec must be greater than zero or null.")

    durations_sec = duration_seconds(df, dur_col)
    if durations_sec.sum() <= target_duration_sec:
        return df.copy()

    shuffled = df.sample(frac=1, random_state=seed)
    shuffled_durations = duration_seconds(shuffled, dur_col)
    crossing_position = int(
        (shuffled_durations.cumsum() >= target_duration_sec).to_numpy().argmax()
    )
    return shuffled.iloc[: crossing_position + 1].reset_index(drop=True).copy()


def apply_duration_budgets(
    *,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dur_col: str,
    split_seed: int,
    target_train_duration_sec: Optional[float] = None,
    target_dev_duration_sec: Optional[float] = None,
    target_test_duration_sec: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply independent deterministic duration targets to existing splits."""
    return tuple(
        select_duration_budget(
            split_df,
            dur_col=dur_col,
            target_duration_sec=target,
            seed=split_seed + DURATION_SPLIT_SEED_OFFSETS[split_name],
        )
        for split_name, split_df, target in (
            ("train", train_df, target_train_duration_sec),
            ("dev", dev_df, target_dev_duration_sec),
            ("test", test_df, target_test_duration_sec),
        )
    )


def build_split_summary(
    *,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_col: Optional[str] = None,
    dur_col: Optional[str] = None,
) -> Dict[str, object]:
    summary = {
        "n_train_rows": int(len(train_df)),
        "n_dev_rows": int(len(dev_df)),
        "n_test_rows": int(len(test_df)),
    }

    if split_col:
        summary["train_split_values"] = sorted(
            train_df[split_col].dropna().unique().tolist()
        )
        summary["dev_split_values"] = sorted(
            dev_df[split_col].dropna().unique().tolist()
        )
        summary["test_split_values"] = sorted(
            test_df[split_col].dropna().unique().tolist()
        )

    if dur_col and dur_col in train_df.columns:
        def _sum_sec(df: pd.DataFrame) -> float:
            return float(duration_seconds(df, dur_col).sum())

        summary["train_total_sec"] = _sum_sec(train_df)
        summary["dev_total_sec"] = _sum_sec(dev_df)
        summary["test_total_sec"] = _sum_sec(test_df)

    return summary


def save_split_summary(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
