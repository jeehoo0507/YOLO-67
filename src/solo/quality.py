from __future__ import annotations

from collections import Counter

import numpy as np

from .pseudo_labels import Box

AREA_BINS = [0.0, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.000001]
ASPECT_BINS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 1000000.0]


class QualityStats:
    """Bounded-memory quality aggregation, including recovered valid labels."""

    def __init__(self):
        self.box_counts = Counter()
        self.areas = np.zeros(len(AREA_BINS) - 1, dtype=np.int64)
        self.aspects = np.zeros(len(ASPECT_BINS) - 1, dtype=np.int64)

    def add(self, boxes: list[Box], image_size: tuple[int, int]) -> None:
        self.box_counts[len(boxes)] += 1
        width, height = image_size
        self.areas += np.histogram([b[2] * b[3] for b in boxes], AREA_BINS)[0]
        self.aspects += np.histogram([b[2] * width / (b[3] * height) for b in boxes], ASPECT_BINS)[
            0
        ]

    def report(self) -> dict:
        count = sum(self.box_counts.values())
        total_boxes = sum(k * v for k, v in self.box_counts.items())

        def order_stat(index):
            cumulative = 0
            for boxes, frequency in sorted(self.box_counts.items()):
                cumulative += frequency
                if cumulative > index:
                    return boxes
            return 0

        median = (order_stat((count - 1) // 2) + order_stat(count // 2)) / 2 if count else 0
        return {
            "valid_images": count,
            "total_boxes": total_boxes,
            "mean_boxes_per_image": total_boxes / count if count else 0,
            "median_boxes_per_image": median,
            "zero_box_image_ratio": self.box_counts[0] / count if count else 0,
            "boxes_per_image_counts": dict(sorted(self.box_counts.items())),
            "bbox_area_distribution": {"bin_edges": AREA_BINS, "counts": self.areas.tolist()},
            "bbox_aspect_ratio_distribution": {
                "units": "width/height in original image pixels",
                "bin_edges": ASPECT_BINS,
                "counts": self.aspects.tolist(),
            },
        }
