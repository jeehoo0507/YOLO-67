"""CPU MaskCut adapted from facebookresearch/CutLER (CC BY-NC-SA 4.0).

See THIRD_PARTY.md and licenses/CutLER.txt. No CRF; boxes use patch masks directly.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.linalg import eigh

from .config import MaskCutConfig


def discover_masks(
    features: np.ndarray, grid: tuple[int, int], config: MaskCutConfig
) -> list[np.ndarray]:
    """Last-attention keys [patches, channels] -> nonoverlapping boolean patch masks."""
    rows, cols = grid
    if features.ndim != 2 or features.shape[0] != rows * cols:
        raise ValueError("Feature count does not match patch grid")
    if not np.isfinite(features).all():
        raise ValueError("DINO features contain NaN/Inf")
    # CPU graph/eigensolver use float64; GPU features are transferred as float32.
    feats = np.asarray(features, dtype=np.float64).copy()
    painting = np.zeros(rows * cols, dtype=bool)
    masks = []
    previous = np.zeros((rows, cols), dtype=bool)
    emitted = previous.copy()
    for index in range(config.max_objects):
        if index:
            painting |= previous.ravel()
            feats[painting] = 0
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        normalized = feats / np.maximum(norms, 1e-12)
        # macOS BLAS can leave spurious FP flags set (numpy/numpy#29820).
        # Inspect values and the cosine diagonal rather than trusting those flags.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            similarity = normalized @ normalized.T
        if not np.isfinite(similarity).all():
            raise ValueError("Non-finite CPU affinity matrix")
        active = norms[:, 0] > 1e-12
        if active.any() and not np.allclose(np.diag(similarity)[active], 1.0, atol=1e-6):
            raise ValueError("CPU BLAS produced an invalid cosine diagonal")
        affinity = np.where(similarity > config.tau, 1.0, config.epsilon)
        degree = affinity.sum(axis=1)
        laplacian = np.diag(degree) - affinity
        _, vectors = eigh(
            laplacian, np.diag(degree), subset_by_index=[1, 1], check_finite=False, driver="gvx"
        )
        vector = vectors[:, 0]
        partition = vector > vector.mean()
        seed = int(np.argmax(np.abs(vector)))
        image_partition = partition.reshape(rows, cols)
        corners = sum(
            bool(x)
            for x in (
                image_partition[0, 0],
                image_partition[0, -1],
                image_partition[-1, 0],
                image_partition[-1, -1],
            )
        )
        if corners >= 3 or not partition[seed]:
            partition = ~partition
            vector = -vector
        seed = int(np.argmax(vector))
        components, _ = ndimage.label(partition.reshape(rows, cols))  # 4-connectivity
        component_id = components.ravel()[seed]
        mask = (components == component_id) if component_id else np.zeros_like(previous)
        if index:
            intersection = np.logical_and(previous, mask).sum()
            union = np.logical_or(previous, mask).sum()
            if (union and intersection / union > config.max_mask_iou) or (
                mask.mean() <= config.min_mask_area
            ):
                mask = np.zeros_like(previous)
        previous = mask
        new_mask = mask & ~emitted
        emitted |= new_mask
        if new_mask.any():
            masks.append(new_mask)
    return masks
