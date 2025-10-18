#!/usr/bin/env python3
"""Populate the feedback database with demo entries for workshop scenarios."""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from typing import List

from app import CANONICAL_MASSNAHMEN, ensure_feedback_db, load_feedback_from_db, save_feedback_to_db


COMMENTS: List[str] = [
    "Bei diesem Lieferanten immer nachfassen",
    "Betrag war zu hoch für einfache Prüfung",
    "Gutschrift wäre richtig gewesen",
    "Telefonische Bestätigung empfohlen",
    "Bitte Dokumente nachreichen",
]

USERS = ["Maria Müller", "Workshop Gast", "S-Factoring Team", "Analyst", "Gast"]


def generate_demo_feedback(count: int, seed: int = 42) -> None:
    ensure_feedback_db()
    rng = random.Random(seed)
    now = datetime.utcnow()
    existing = load_feedback_from_db()
    base_index = int(existing["beleg_index"].max()) + 1 if not existing.empty else 0

    for idx in range(count):
        beleg_index = base_index + idx
        timestamp = now - timedelta(days=rng.randint(0, 27), hours=rng.randint(0, 23))
        predicted = rng.choice(CANONICAL_MASSNAHMEN)
        if rng.random() < 0.85:
            actual = predicted
            flag = "correct"
        else:
            alternatives = [label for label in CANONICAL_MASSNAHMEN if label != predicted]
            actual = rng.choice(alternatives)
            flag = "incorrect"

        entry = {
            "beleg_index": beleg_index,
            "beleg_id": f"DEMO-{beleg_index:05d}",
            "timestamp": timestamp.isoformat(),
            "user": rng.choice(USERS),
            "score": rng.uniform(30, 98),
            "prediction": None,
            "feedback": flag,
            "comment": rng.choice(COMMENTS),
            "actual_label": actual,
            "predicted_label": predicted,
            "is_correct": flag == "correct",
        }
        save_feedback_to_db(entry)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load demo feedback entries for workshops")
    parser.add_argument("--count", type=int, default=100, help="Number of feedback entries to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_demo_feedback(max(0, args.count), seed=args.seed)
    print(f"✅ {max(0, args.count)} Demo-Feedbacks geladen.")


if __name__ == "__main__":
    main()
