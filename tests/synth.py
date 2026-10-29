"""Synthetic corpora with known structure, for geometry and change tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

A = [f"alpha{i}" for i in range(8)]
B = [f"beta{i}" for i in range(8)]
FILL = ["the", "a", "of", "and", "to", "in"]


def sentence(rng: np.random.Generator, group: list[str], extra: list[str] | None = None) -> str:
    words = list(rng.choice(group, 6)) + list(rng.choice(FILL, 3))
    if extra:
        words += extra
    rng.shuffle(words)
    return " ".join(words) + "."


def write_clusters(path: Path, n_docs: int = 40, seed: int = 0) -> Path:
    """Two clusters; 'pivot' sits with A in era 1800 and with B in era 1900."""
    rng = np.random.default_rng(seed)
    with path.open("w", encoding="utf-8") as f:
        for d in range(n_docs):
            year = 1800 if d < n_docs // 2 else 1900
            sents = []
            for _ in range(30):
                g = A if rng.random() < 0.5 else B
                extra = ["pivot"] if (g is A) == (year == 1800) and rng.random() < 0.3 else None
                sents.append(sentence(rng, g, extra))
            f.write(json.dumps({"id": f"d{d}", "year": year, "text": " ".join(sents)}) + "\n")
    return path


CLUSTERS = [[f"c{c}w{i}" for i in range(15)] for c in range(6)]


def write_change(path: Path, n_docs: int = 60, seed: int = 0, shift: bool = True) -> Path:
    """Six topic clusters; 'pivot' co-occurs with cluster 0 in 1800 and cluster 3 in 1900."""
    rng = np.random.default_rng(seed)
    with path.open("w", encoding="utf-8") as f:
        for d in range(n_docs):
            year = 1800 if d % 2 == 0 else 1900
            home = 3 if (shift and year == 1900) else 0
            sents = []
            for _ in range(40):
                c = int(rng.integers(0, 6))
                extra = ["pivot"] if c == home and rng.random() < 0.5 else None
                sents.append(sentence(rng, CLUSTERS[c], extra))
            f.write(json.dumps({"id": f"d{d}", "year": year, "text": " ".join(sents)}) + "\n")
    return path
