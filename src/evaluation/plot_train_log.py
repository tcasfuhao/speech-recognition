from __future__ import annotations

# Standard libraries
import argparse
from math import ceil
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import yaml


PLOT_STYLE = {
    "figure_size": (10, 6),
    "linewidth": 2.5,
    "grid_linestyle": "--",
    "grid_linewidth": 1.5,
    "dpi": 300,
}

METRIC_STYLE = {
    "loss": {
        "color": sns.color_palette()[0],
        "linestyle": "-",
        "ylabel": "training loss",
    },
    "grad_norm": {
        "color": sns.color_palette()[1],
        "linestyle": (0, (5, 1)),
        "ylabel": "gradient norm",
    },
    "learning_rate": {
        "color": sns.color_palette()[2],
        "linestyle": (0, (3, 1, 1, 1)),
        "ylabel": "learning rate",
    },
    "eval_loss": {
        "color": sns.color_palette()[3],
        "linestyle": (0, (1, 1)),
        "ylabel": "development loss",
    },
    "eval_cer": {
        "color": sns.color_palette()[4],
        "linestyle": (0, (3, 1, 1, 1, 1, 1)),
        "ylabel": "development CER",
    },
}


def pretty_label(name: str) -> str:
    return name.replace("_", " ").strip()


def round_up(value: float, base: float) -> float:
    if pd.isna(value):
        return value

    return ceil(value / base) * base


def metric_frame(log_df: pd.DataFrame, y_col: str) -> pd.DataFrame:
    if y_col not in log_df.columns:
        raise ValueError(f"Column {y_col!r} not found in log file.")

    df = log_df.loc[log_df[y_col].notna()].copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    return df.loc[np.isfinite(df[y_col])].copy()


def finite_max(log_df: pd.DataFrame, y_col: str) -> float | None:
    if y_col not in log_df.columns:
        return None

    series = pd.to_numeric(log_df[y_col], errors="coerce")
    series = series[np.isfinite(series)]
    if series.empty:
        return None

    return series.max()


def plot_metric(
    log_df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    title: str | None = None,
    x_lim: float | None = None,
    y_lim: float | None = None,
    out_path: str | Path | None = None,
    show: bool = True,
) -> None:
    df = metric_frame(log_df, y_col)

    style = METRIC_STYLE.get(
        y_col,
        {
            "color": sns.color_palette()[0],
            "linestyle": "-",
            "ylabel": pretty_label(y_col),
        },
    )

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figure_size"])

    for location in ["left", "right", "top", "bottom"]:
        ax.spines[location].set_linewidth(1.5)

    sns.lineplot(
        data=df,
        x=x_col,
        y=y_col,
        ax=ax,
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=PLOT_STYLE["linewidth"],
        legend=False,
    )

    if x_lim is not None:
        ax.set_xlim(0, x_lim)
    if y_lim is not None:
        ax.set_ylim(0, y_lim)

    ax.tick_params(axis="both", which="major", labelsize=20, width=1.5)

    ax.set_xlabel(pretty_label(x_col), fontsize=20)
    ax.set_ylabel(style["ylabel"], fontsize=20)
    ax.set_title(
        title or f"{style['ylabel']} vs {pretty_label(x_col)}", fontsize=20
    )
    ax.grid(
        True, which="both",
        ls=PLOT_STYLE["grid_linestyle"], lw=PLOT_STYLE["grid_linewidth"]
    )

    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=PLOT_STYLE["dpi"], bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_training_suite(
    log_df: pd.DataFrame,
    *,
    x_col: str = "step",
    model_id: str | None = None,
    out_dir: str | Path | None = None,
    show: bool = True,
) -> None:
    out_dir = Path(out_dir) if out_dir is not None else None

    metrics = [
        "loss",
        "grad_norm",
        "learning_rate",
        "eval_loss",
        "eval_cer",
    ]

    step_max = round_up(log_df[x_col].max(), 1000)

    y_lims = {
        "loss": round_up(finite_max(log_df, "loss"), 5)
        if "loss" in log_df else None,
        "grad_norm": round_up(finite_max(log_df, "grad_norm"), 5)
        if "grad_norm" in log_df else None,
        "learning_rate":
            round_up(finite_max(log_df, "learning_rate"), 5e-5)
            if "learning_rate" in log_df else None,
        "eval_loss": round_up(finite_max(log_df, "eval_loss"), 2)
        if "eval_loss" in log_df else None,
        "eval_cer": 0.8,
    }

    for metric in metrics:
        if metric not in log_df.columns:
            continue

        fig_label = METRIC_STYLE.get(metric, {"ylabel": metric})["ylabel"]
        fig_label = "_".join(fig_label.lower().split())
        if model_id is not None:
            fig_label = f"{model_id}_{fig_label}"

        out_path = None
        if out_dir is not None:
            out_path = out_dir / f"{fig_label}_vs_{x_col}.png"

        plot_metric(
            log_df,
            x_col=x_col,
            y_col=metric,
            x_lim=step_max,
            y_lim=y_lims.get(metric),
            out_path=out_path,
            show=show,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, help="Path to YAML config file")

    ap.add_argument("--log_path", type=str, help="Path to train_log.tsv")
    ap.add_argument("--model_id", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--x_col", type=str, default="step")
    ap.add_argument("--show", action="store_true")

    args = ap.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        for k, v in config.items():
            setattr(args, k, v)

        print(f"Loaded config from {args.config}:")
        print(yaml.dump(config, sort_keys=False))

    if args.log_path is None:
        raise ValueError("log_path must be provided via CLI or config file")

    log_path = Path(args.log_path).expanduser()
    if args.model_id is None:
        args.model_id = log_path.parent.name

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser()
    else:
        out_dir = log_path.parent / "plots"

    log_df = pd.read_csv(log_path, sep="\t")
    plot_training_suite(
        log_df,
        x_col=args.x_col,
        model_id=args.model_id,
        out_dir=out_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
