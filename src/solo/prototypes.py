from __future__ import annotations

import numpy as np


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms < 1e-12):
        raise ValueError("Non-finite or zero embedding")
    return values / norms


def make_prototypes(embeddings: np.ndarray, labels: list[str]):
    if len(embeddings) != len(labels) or not labels:
        raise ValueError("Support labels and embeddings must be nonempty and aligned")
    values = normalize(embeddings)
    classes = sorted(set(labels))
    prototypes = normalize(
        np.stack(
            [
                values[[i for i, label in enumerate(labels) if label == name]].mean(axis=0)
                for name in classes
            ]
        )
    )
    return classes, prototypes
