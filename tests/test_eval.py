"""Checks for the matching and counting in ppe.eval (no model needed).

Run: python -m pytest tests
"""

import numpy as np

from ppe.eval import greedy_match, iou_matrix, match_per_class, person_level, prf_at, select_conf

NAMES = {0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles", 5: "none", 6: "person",
         7: "no_helmet", 8: "no_goggle", 9: "no_gloves", 10: "no_boots"}


def frame(pred, gt):
    """pred: list of (cls, conf, xyxy); gt: list of (cls, xyxy). Predictions get sorted by confidence."""
    pred = sorted(pred, key=lambda p: -p[1])
    return {
        "image": "x.jpg",
        "pred_cls": np.array([p[0] for p in pred], int),
        "pred_conf": np.array([p[1] for p in pred], float),
        "pred_xyxy": np.array([p[2] for p in pred], float).reshape(-1, 4),
        "gt_cls": np.array([g[0] for g in gt], int),
        "gt_xyxy": np.array([g[1] for g in gt], float).reshape(-1, 4),
    }


def test_iou_identical_and_disjoint():
    a = np.array([[0, 0, 10, 10]], float)
    assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0


def test_duplicate_prediction_is_a_false_positive():
    iou = np.array([[0.9], [0.8]])
    assert greedy_match(iou, 0.5) == {0: 0}


def test_prf_at_threshold():
    fr = frame([(7, 0.9, (0, 0, 10, 10)), (7, 0.2, (50, 50, 60, 60))],
               [(7, (0, 0, 10, 10)), (7, (100, 100, 110, 110))])
    per = match_per_class([fr], 11, 0.5)
    tp, fp, fn, p, r, _ = prf_at(per[7], 0.5)
    assert (tp, fp, fn, p, r) == (1, 0, 1, 1.0, 0.5)
    tp, fp, fn, p, r, _ = prf_at(per[7], 0.1)
    assert (tp, fp, fn) == (1, 1, 1)


def test_threshold_is_chosen_from_given_split_only():
    fr = frame([(0, 0.9, (0, 0, 10, 10)), (0, 0.3, (50, 50, 60, 60))], [(0, (0, 0, 10, 10))])
    t, f1 = select_conf(match_per_class([fr], 11, 0.5))
    assert 0.3 < t <= 0.9 and f1 == 1.0


def test_person_level_violation_and_false_alert():
    person_a, person_b = (0, 0, 100, 200), (200, 0, 300, 200)
    gt = [(6, person_a), (7, (40, 0, 60, 20)),          # A has no helmet
          (6, person_b), (0, (240, 0, 260, 20))]        # B wears a helmet
    pred = [(6, 0.9, person_a), (6, 0.9, person_b),
            (7, 0.8, (40, 0, 60, 20)),                  # correct no_helmet on A
            (9, 0.7, (210, 100, 230, 120))]             # wrong no_gloves on B
    pl = person_level([frame(pred, gt)], NAMES, 0.5, ["helmet", "gloves"], 0.5)
    assert pl["gt_viol"]["helmet"] == 1 and pl["hit"]["helmet"] == 1
    assert pl["alerts"]["gloves"] == 1 and pl["true_alerts"]["gloves"] == 0
    assert pl["nonviol_gt"] == 1 and pl["nonviol_flagged"] == 1


def test_missed_person_counts_as_missed_violation():
    gt = [(6, (0, 0, 100, 200)), (7, (40, 0, 60, 20))]
    pl = person_level([frame([], gt)], NAMES, 0.5, ["helmet"], 0.5)
    assert pl["gt_viol"]["helmet"] == 1 and pl["hit"]["helmet"] == 0 and pl["n_gt_unmatched"] == 1
