#!/usr/bin/env python3
"""Audit exact train/evaluation instruction overlap for LIBERO-Para."""

import argparse
import csv
import json
from pathlib import Path
import re


def normalize_instruction(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-paraphrases",
        type=Path,
        default=Path("examples/libero_para/train_paraphrases_v1.json"),
    )
    parser.add_argument("--libero-para-root", type=Path, required=True)
    args = parser.parse_args()

    with args.train_paraphrases.open(encoding="utf-8") as f:
        train_data = json.load(f)
    train_phrases = {
        normalize_instruction(phrase): (canonical, phrase)
        for canonical, phrases in train_data["paraphrases"].items()
        for phrase in phrases
    }

    metadata_path = args.libero_para_root / "metrics" / "libero_para_metadata.csv"
    with metadata_path.open(encoding="utf-8", newline="") as f:
        evaluation_phrases = {
            normalize_instruction(row["new_instruction"]): row["new_instruction"] for row in csv.DictReader(f)
        }

    overlaps = sorted(set(train_phrases).intersection(evaluation_phrases))
    print(f"Training paraphrases: {len(train_phrases)}")
    print(f"LIBERO-Para evaluation instructions: {len(evaluation_phrases)}")
    print(f"Normalized exact overlaps: {len(overlaps)}")
    for normalized in overlaps:
        canonical, train_phrase = train_phrases[normalized]
        print(f"  task={canonical!r} train={train_phrase!r} eval={evaluation_phrases[normalized]!r}")

    if overlaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
