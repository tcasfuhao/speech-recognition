from __future__ import annotations

# Standard libraries
import json
import random

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Third-party libraries
import pandas as pd


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
            if dur_col.lower().endswith(("ms", "msec")):
                return float(df[dur_col].sum() / 1000.0)
            return float(df[dur_col].sum())

        summary["train_total_sec"] = _sum_sec(train_df)
        summary["dev_total_sec"] = _sum_sec(dev_df)
        summary["test_total_sec"] = _sum_sec(test_df)

    return summary


def save_split_summary(path: str, summary: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
