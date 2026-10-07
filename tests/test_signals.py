"""Checks for the label-free signal helpers and the confidence baselines (no model needed).

Run: python -m pytest tests
"""

import numpy as np

from ppe.estimator import Baselines, recall
from ppe.signals import aug_views, ks_stat, n_matched, temporal_views


def test_ks_identical_is_zero_and_disjoint_is_one():
    ref = np.sort(np.linspace(0.1, 0.9, 50))
    assert ks_stat(ref, ref.copy()) == 0.0
    assert ks_stat(ref, np.full(10, 0.95)) == 1.0
    assert ks_stat(ref, np.zeros(0)) == 1.0


def test_n_matched_is_per_class():
    a_cls, a_conf = np.array([0, 1]), np.array([0.9, 0.8])
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    # same boxes, but the second has another class: only one match
    assert n_matched(a_cls, a, a_conf, np.array([0, 2]), a.copy(), 0.5) == 1
    assert n_matched(a_cls, a, a_conf, a_cls.copy(), a.copy(), 0.5) == 2


def test_aug_views_map_boxes_back():
    img = np.zeros((100, 200, 3), np.uint8)
    views = aug_views(img, {"scale": 0.5, "brightness": 1.3})
    box = np.array([[10.0, 20.0, 30.0, 40.0]])
    flip_img, flip_back = views[0]
    # a box at x 10..30 appears at x 170..190 in the flipped view and maps back
    assert np.allclose(flip_back(np.array([[170.0, 20.0, 190.0, 40.0]])), box)
    small_img, small_back = views[1]
    assert small_img.shape[:2] == (50, 100)
    assert np.allclose(small_back(box / 2), box)


def test_temporal_views_are_seeded():
    img = np.full((64, 64, 3), 128, np.uint8)
    p = {"frames": 3, "shift_frac": 0.02, "noise_sigma": 0.01}
    a = temporal_views(img, p, np.random.default_rng(1))
    b = temporal_views(img, p, np.random.default_rng(1))
    assert len(a) == 3 and all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def _rec(confs, cls, tp, gt):
    return {"cand_conf": np.array(confs), "cand_cls": np.array(cls), "tp": np.array(tp, float),
            "gt": np.array(gt, float)}


def test_baselines_are_zero_on_their_source():
    src = [_rec([0.9, 0.5, 0.2], [0, 0, 1], [1, 0], [2, 1]), _rec([0.7, 0.3], [0, 1], [1, 1], [1, 1])]
    b = Baselines(src, [0, 1], True)
    assert abs(recall(src, range(2), [0, 1]) - 0.6) < 1e-9
    est = b(src, range(2))
    assert abs(est["ac"]) < 1e-9 and abs(est["atc"]) < 1e-9
    # a window whose confidences all fell estimates a positive drop with both baselines
    low = [_rec([0.15, 0.12], [0, 1], [0, 0], [1, 1])]
    est = b(low, range(1))
    assert est["ac"] > 0 and est["atc"] > 0
