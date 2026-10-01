"""Mean dense features inside normalized xywh boxes; no crop backbone calls."""

from __future__ import annotations

import numpy as np


def roi_pool(features: np.ndarray, grid: tuple[int, int], boxes) -> np.ndarray:
    rows, cols = grid
    if features.ndim != 2 or len(features) != rows * cols or not np.isfinite(features).all():
        raise ValueError("Invalid dense features/grid")
    outputs = []
    for x, y, w, h in boxes:
        if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
            raise ValueError("Invalid ROI box")
        x0, x1 = max(0.0, x - w / 2) * cols, min(1.0, x + w / 2) * cols
        y0, y1 = max(0.0, y - h / 2) * rows, min(1.0, y + h / 2) * rows
        # Fractional cell coverage avoids discarding small boxes between patch centers.
        wx = np.maximum(0, np.minimum(np.arange(cols) + 1, x1) - np.maximum(np.arange(cols), x0))
        wy = np.maximum(0, np.minimum(np.arange(rows) + 1, y1) - np.maximum(np.arange(rows), y0))
        weight = (wy[:, None] * wx[None, :]).ravel()
        if weight.sum() <= 0:
            raise ValueError("ROI does not intersect the image")
        outputs.append((features * weight[:, None]).sum(axis=0) / weight.sum())
    return np.asarray(outputs, dtype=np.float32).reshape(-1, features.shape[1])
