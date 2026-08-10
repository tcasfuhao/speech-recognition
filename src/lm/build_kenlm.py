from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data import schema
from src.evaluation.metrics import strip_whitespace


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _character_tokens(text: str) -> str:
    return " ".join(strip_whitespace(text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a KenLM model from the saved training split.")
    parser.add_argument("--config", type=str)
    parser.add_argument("--train_csv", type=str)
    parser.add_argument("--kenlm_path", type=str)
    parser.add_argument("--lm_order", type=int, default=5)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    if args.config:
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        for key, value in config.items():
            setattr(args, key, value)

    for key in ("train_csv", "kenlm_path", "out_dir"):
        if not getattr(args, key, None):
            raise ValueError(f"{key} must be provided via CLI or config file")

    train_csv = _expand(args.train_csv)
    kenlm_path = _expand(args.kenlm_path)
    out_root = _expand(args.out_dir)
    lmplz = kenlm_path / "lmplz"
    build_binary = kenlm_path / "build_binary"

    for executable in (lmplz, build_binary):
        if not executable.is_file():
            raise FileNotFoundError(f"KenLM executable not found: {executable}")

    frame = pd.read_csv(train_csv)
    text_col = schema.pick_col(frame, schema.TEXT_COL_CANDIDATES)
    texts = frame[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 0].map(_character_tokens)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"lm_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    corpus_path = run_dir / "lm.txt"
    arpa_path = run_dir / "lm.arpa"
    binary_path = run_dir / "lm.binary"
    texts.to_csv(corpus_path, index=False, header=False)

    with corpus_path.open("r", encoding="utf-8") as corpus, arpa_path.open(
        "w", encoding="utf-8"
    ) as arpa:
        subprocess.run(
            [str(lmplz), "-o", str(args.lm_order), "--discount_fallback"],
            stdin=corpus,
            stdout=arpa,
            check=True,
        )
    subprocess.run([str(build_binary), str(arpa_path), str(binary_path)], check=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(vars(args), indent=2), encoding="utf-8"
    )
    print(f"KenLM corpus: {corpus_path}")
    print(f"KenLM ARPA: {arpa_path}")
    print(f"KenLM binary: {binary_path}")


if __name__ == "__main__":
    main()
