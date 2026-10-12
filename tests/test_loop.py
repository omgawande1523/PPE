"""Checks for the loop's pure parts: diagnosis rules, mining, poisoning, label error, the gate, rollback.

Run: python -m pytest tests
"""

import numpy as np
import pytest

from ppe import diagnose as dg
from ppe.improve import label_error, mine, poison, xyxy_to_yolo, yolo_to_xyxy
from ppe.verify import gate, promote, rollback

CFG = {"black_mean": 0.012, "covered_std": 0.01, "frozen_diff": 0.002, "blur_ratio": 0.5, "fault_frac": 0.5,
       "persist_windows": 3}
OK = {"mean": 0.4, "std": 0.2, "sharpness": 1.0, "diff": 0.1}


def test_frame_faults_one_per_frame_in_order():
    assert dg.frame_faults(OK, CFG, 1.0) == []
    assert dg.frame_faults({**OK, "mean": 0.005, "diff": 0.0}, CFG, 1.0) == ["black"]
    assert dg.frame_faults({**OK, "std": 0.005}, CFG, 1.0) == ["covered"]
    assert dg.frame_faults({**OK, "diff": 0.001}, CFG, 1.0) == ["frozen"]
    assert dg.frame_faults({**OK, "sharpness": 0.4}, CFG, 1.0) == ["blur"]
    assert dg.frame_faults({**OK, "diff": float("nan")}, CFG, 1.0) == []


def test_health_is_brightness_invariant_for_sharpness():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (120, 160, 3)).astype(np.uint8)
    a, _ = dg.health(img, None)
    b, _ = dg.health((img * 0.3).astype(np.uint8), None)
    assert abs(a["sharpness"] - b["sharpness"]) / a["sharpness"] < 0.1
    assert b["mean"] < a["mean"]


def test_diagnoser_fault_temporal_persistent():
    d = dg.Diagnoser(CFG, threshold=0.1, ref_sharp_q01=1.0)
    bad = [{**OK, "mean": 0.0}] * 10
    assert d.window(0, bad, est_drop=0.9)["finding"] == dg.CAMERA_FAULT   # fault wins over a high estimate
    assert d.window(1, [OK] * 10, 0.2)["finding"] == dg.TEMPORAL
    r = d.window(2, [OK] * 10, 0.0)
    assert r["finding"] == dg.HEALTHY and r["recovered_after"] == 1
    for k in range(3):
        r = d.window(3 + k, [OK] * 10, 0.2)
    assert r["finding"] == dg.PERSISTENT and r["action"] == "improve"


def test_mine_takes_top_scores_seeded():
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.0])
    assert set(mine(s, 2, 0)) == {1, 3}
    assert list(mine(s, 3, 0)) == list(mine(s, 3, 0))


def test_poison_swaps_violations_only():
    names = {0: "helmet", 7: "no_helmet", 6: "person"}
    labels = [(np.array([7, 6, 0]), np.zeros((3, 4)))]
    out, n = poison(labels, {"p": 1.0, "swap": {"no_helmet": "helmet"}}, names, 0)
    assert n == 1 and list(out[0][0]) == [0, 6, 0]
    assert list(labels[0][0]) == [7, 6, 0]          # input untouched


def test_label_error_counts_wrong_and_missed():
    box = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    truth = [(np.array([0, 7]), box)]
    assert label_error(truth, truth, 0.5)["error_rate"] == 0.0
    e = label_error([(np.array([0, 0]), box)], truth, 0.5)       # second box has the wrong class
    assert e["error_rate"] == 0.5 and e["miss_rate"] == 0.5
    assert label_error([(np.array([0, 0]), box)], truth, 0.5, [7])["miss_rate"] == 1.0


def test_yolo_xyxy_roundtrip():
    b = np.array([[10.0, 20.0, 50.0, 80.0]])
    assert np.allclose(yolo_to_xyxy(xyxy_to_yolo(b, 100, 100), 100, 100), b)


def _m(viol, map50, all_=0.5):
    names = ["no_helmet", "no_goggle", "no_gloves", "no_boots"]
    return {"recall": {**dict(zip(names, viol)), "all": all_}, "map50": map50,
            "tp": {n: 1 for n in names}, "gt": {n: 4 for n in names}}


def test_gate_rules():
    inc, inc_site = _m([0.2, 0.3, 0.2, 0.0], 0.57), _m([0, 0, 0, 0], 0, all_=0.4)
    ok, _ = gate(_m([0.2, 0.3, 0.25, 0.0], 0.56), inc, _m([0, 0, 0, 0], 0, all_=0.5), inc_site, 0.02)
    assert ok
    ok, lines = gate(_m([0.15, 0.3, 0.25, 0.0], 0.60), inc, _m([0, 0, 0, 0], 0, all_=0.5), inc_site, 0.02)
    assert not ok and lines[0].startswith("FAIL")                  # one no_* class fell
    ok, _ = gate(_m([0.2, 0.3, 0.2, 0.0], 0.54), inc, _m([0, 0, 0, 0], 0, all_=0.5), inc_site, 0.02)
    assert not ok                                                  # mAP fell by more than the tolerance
    ok, _ = gate(_m([0.2, 0.3, 0.2, 0.0], 0.57), inc, _m([0, 0, 0, 0], 0, all_=0.4), inc_site, 0.02)
    assert not ok                                                  # site recall did not improve


def test_promote_keeps_last_three_and_rolls_back(tmp_path, monkeypatch):
    import ppe.verify as vf
    monkeypatch.setattr(vf, "REPO_ROOT", tmp_path)
    (tmp_path / "w0.pt").write_bytes(b"0")
    state = {"current": "w0.pt"}
    for i in range(1, 5):
        c = tmp_path / f"cand{i}.pt"
        c.write_bytes(str(i).encode())
        promote(state, c, tmp_path / "promoted", 3, f"p{i}")
    assert state["current"] == "promoted/p4.pt"
    assert sorted(p.name for p in (tmp_path / "promoted").iterdir()) == ["p2.pt", "p3.pt", "p4.pt"]
    assert rollback(state) == "promoted/p3.pt"
    with pytest.raises(ValueError):
        rollback({"history": []})
