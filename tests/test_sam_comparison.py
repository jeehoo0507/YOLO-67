import numpy as np

from solo.config import FilterConfig
from solo.sam_comparison import filter_sam_masks


def test_sam_masks_use_original_coordinates_and_quality_order():
    first = np.zeros((20, 40), dtype=bool)
    first[2:10, 4:20] = True
    second = np.zeros_like(first)
    second[12:18, 25:35] = True
    full = np.ones_like(first)  # excluded by maximum box area
    annotations = [
        {"segmentation": first, "predicted_iou": .8},
        {"segmentation": full, "predicted_iou": .99},
        {"segmentation": second, "predicted_iou": .85},
        {"segmentation": first.copy(), "predicted_iou": .95},
    ]
    boxes, selected = filter_sam_masks(annotations, (40, 20), FilterConfig())
    np.testing.assert_allclose(boxes, [(.3, .3, .4, .4), (.75, .75, .25, .3)])
    assert [a["predicted_iou"] for a in selected] == [.95, .85]


def test_sam_does_not_force_a_box_count_or_remove_nested_parts():
    annotations = []
    for x in range(15):
        mask = np.zeros((20, 100), dtype=bool)
        mask[5:10, x*6:x*6+4] = True
        annotations.append({"segmentation": mask, "predicted_iou": .9})
    large = np.zeros((20, 100), dtype=bool)
    large[2:15, :94] = True
    annotations.append({"segmentation": large, "predicted_iou": .8})
    boxes, _ = filter_sam_masks(annotations, (100, 20), FilterConfig(min_box_area=.001))
    assert len(boxes) == 16
    assert filter_sam_masks([], (100, 20), FilterConfig()) == ([], [])
