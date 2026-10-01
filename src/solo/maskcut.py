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
        if config.all_components:
            # Remove old masks before components: subtraction can disconnect a region.
            fresh = partition.reshape(rows, cols) & ~emitted
            minimum = max(1, int(np.ceil(config.min_mask_area * rows * cols)))
            candidates = connected_masks(fresh, minimum)
            if not candidates:
                break
            previous = np.logical_or.reduce(candidates)
            emitted |= previous
            masks.extend(candidates[: config.max_objects - len(masks)])
            if len(masks) >= config.max_objects:
                break
            continue
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
    if config.split_depth:
        return refine_instances(masks, features, config)
    return masks


def connected_masks(mask: np.ndarray, minimum: int) -> list[np.ndarray]:
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    ids = sorted(range(1, count + 1), key=lambda i: (-sizes[i], i))
    return [labels == i for i in ids if sizes[i] >= minimum]


def split_mask(mask: np.ndarray, features: np.ndarray, config: MaskCutConfig) -> list[np.ndarray]:
    """Experimental within-mask normalized cut; no extra backbone forward.

    Soft cosine edges expose differences hidden by the coarse foreground threshold.
    A low cut cost, feature contrast, size, and connected children are all required.
    These are conservative heuristics, not a guarantee of semantic instance boundaries.
    """
    indices = np.flatnonzero(mask)
    minimum = max(config.split_min_patches, int(np.ceil(config.min_mask_area * mask.size)))
    if len(indices) < 2 * minimum:
        return [mask]
    vectors = np.asarray(features[indices], dtype=np.float64)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cosine = np.clip(vectors @ vectors.T, -1, 1)
    if not np.isfinite(cosine).all():
        raise ValueError("Non-finite instance affinity")
    if 1 - cosine.min() < config.split_min_cosine_distance:
        return [mask]  # Uniform appearance must not be divided arbitrarily.
    affinity = np.maximum(np.exp((cosine - 1) / config.split_temperature), config.epsilon)
    np.fill_diagonal(affinity, 0)
    degree = affinity.sum(axis=1)
    _, eigenvectors = eigh(
        np.diag(degree) - affinity, np.diag(degree), subset_by_index=[1, 1],
        check_finite=False, driver="gvx",
    )
    vector = eigenvectors[:, 0]
    best = None
    best_cost = config.split_max_ncut
    # Deterministic threshold sweep, rather than forcing equally sized instances.
    for threshold in np.unique(np.quantile(vector, np.linspace(0.1, 0.9, 9))):
        side = vector > threshold
        if min(side.sum(), (~side).sum()) < minimum:
            continue
        cut = affinity[np.ix_(side, ~side)].sum()
        cost = cut / degree[side].sum() + cut / degree[~side].sum()
        if cost >= best_cost:
            continue
        means = np.stack([vectors[side].mean(axis=0), vectors[~side].mean(axis=0)])
        means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-12)
        if 1 - np.dot(means[0], means[1]) < config.split_min_cosine_distance:
            continue
        children = []
        for selected in (side, ~side):
            child = np.zeros(mask.size, dtype=bool)
            child[indices[selected]] = True
            parts = connected_masks(child.reshape(mask.shape), minimum)
            # Reject fragmented cuts, rather than joining body fragments in one bbox.
            if len(parts) != 1 or parts[0].sum() != selected.sum():
                break
            children.extend(parts)
        if len(children) == 2:
            best, best_cost = children, cost
    return best if best is not None else [mask]


def refine_instances(
    masks: list[np.ndarray], features: np.ndarray, config: MaskCutConfig,
) -> list[np.ndarray]:
    pending = [(mask, 0) for mask in masks]
    result = []
    while pending:
        mask, depth = pending.pop(0)
        if depth >= config.split_depth or len(result) + len(pending) + 1 >= config.max_instances:
            result.append(mask)
            continue
        children = split_mask(mask, features, config)
        if len(children) == 1:
            result.append(mask)
        else:
            pending.extend((child, depth + 1) for child in children)
    return result
