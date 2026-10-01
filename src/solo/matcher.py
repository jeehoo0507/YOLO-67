from __future__ import annotations

import numpy as np

from .prototypes import normalize


def match_prototypes(embeddings, classes, prototypes, threshold=-1.0, min_margin=0.0):
    """Cosine matching; thresholds need support/validation calibration, not query labels."""
    if not -1 <= threshold <= 1 or not 0 <= min_margin <= 2:
        raise ValueError("Invalid matching threshold/margin")
    if len(classes) != len(prototypes) or not classes:
        raise ValueError("Invalid prototypes")
    if not len(embeddings):
        return []
    scores = np.clip(normalize(embeddings) @ normalize(prototypes).T, -1, 1)
    results = []
    for row in scores:
        order = np.argsort(-row, kind="stable")
        best = int(order[0])
        margin = float(row[best] - row[order[1]]) if len(order) > 1 else 2.0
        results.append(
            {
                "label": classes[best]
                if row[best] >= threshold and margin >= min_margin
                else "unknown",
                "top1_label": classes[best],
                "cosine": float(row[best]),
                "margin": margin,
                "scores": {name: float(value) for name, value in zip(classes, row, strict=True)},
            }
        )
    return results
