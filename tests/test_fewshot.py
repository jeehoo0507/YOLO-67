import numpy as np
import pytest

from solo.matcher import match_prototypes
from solo.prototypes import make_prototypes
from solo.roi_pooling import roi_pool


def test_evaluation_never_counts_duplicate_boxes_as_two_leaves():
    from solo.crop_demo import pair_boxes

    a, b = (.2, .2, .2, .2), (.8, .8, .2, .2)
    pairs = pair_boxes([a, a, b], [a, b])
    assert len(pairs) == 2 and {r for _, r in pairs} == {0, 1}
    assert pair_boxes([], [a]) == []


def test_reference_coordinates_are_evaluation_only_voc(tmp_path, monkeypatch):
    from solo.crop_demo import read_reference

    (tmp_path / 'ref.xml').write_text(
        '<annotation><size><width>100</width><height>50</height></size>'
        '<object><name>Tomato leaf</name><bndbox><xmin>1</xmin><ymin>1</ymin>'
        '<xmax>50</xmax><ymax>25</ymax></bndbox></object></annotation>'
    )
    monkeypatch.setattr('solo.crop_demo.ROOT', tmp_path)
    boxes, labels = read_reference({'evaluation_xml': 'ref.xml'}, (200, 100))
    np.testing.assert_allclose(boxes, [[.25, .25, .5, .5]])
    assert labels == ['healthy']


def test_roi_pool_fractional_edges_and_empty_boxes():
    features = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    pooled = roi_pool(features, (2, 2), [(0.25, 0.25, 0.5, 0.5), (0.5, 0.25, 0.5, 0.5)])
    np.testing.assert_allclose(pooled, [[1, 0], [0.5, 0.5]])
    assert roi_pool(features, (2, 2), []).shape == (0, 2)
    with pytest.raises(ValueError):
        roi_pool(features, (2, 2), [(2, 2, 0.1, 0.1)])


def test_prototypes_normalize_each_support_before_averaging():
    labels, prototypes = make_prototypes(np.array([[100.0, 0], [0, 1], [-1, 0]]), ["a", "a", "b"])
    np.testing.assert_allclose(prototypes[0], [2**-0.5, 2**-0.5], rtol=1e-6)
    result = match_prototypes(np.array([[1.0, 1], [-1, 0]]), labels, prototypes)
    assert [r["label"] for r in result] == ["a", "b"]
    result = match_prototypes(np.array([[0.0, -1]]), labels, prototypes, threshold=0.8)
    assert result[0]["label"] == "unknown"
    with pytest.raises(ValueError):
        make_prototypes(np.zeros((1, 2)), ["a"])
