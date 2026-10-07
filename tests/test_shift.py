"""Checks for ppe.shift that need no model or dataset.

Run: python -m pytest tests
"""

import json

import numpy as np
import pytest

from ppe import shift
from ppe.shift import CONDITIONS, CORRUPT, bootstrap_delta, downscale, downscale_labels, image_rng

PARAMS = {
    "low_light": {"gamma": [1.5, 2.0, 2.5], "gain": [0.6, 0.4, 0.25], "noise_sigma": [0.01, 0.02, 0.03]},
    "haze": {"beta": [0.5, 1.0, 1.6], "airlight_bgr": [0.72, 0.80, 0.86]},
    "motion_blur": {"length_frac": [0.015, 0.03, 0.05]},
    "jpeg": {"quality": [25, 15, 7]},
    "rain": {"density": [0.0008, 0.0018, 0.0035], "length_frac": [0.03, 0.05, 0.07], "alpha": [0.5, 0.6, 0.7],
             "darken": [0.9, 0.8, 0.7]},
    "downscale": {"scale": [0.5, 0.35, 0.25]},
}


def img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)


@pytest.mark.parametrize("cond", CONDITIONS)
def test_corruptions_keep_shape_and_are_deterministic(cond):
    a = CORRUPT[cond](img(), PARAMS[cond], 1, image_rng(0, cond, 1, "x.jpg"))
    b = CORRUPT[cond](img(), PARAMS[cond], 1, image_rng(0, cond, 1, "x.jpg"))
    assert a.shape == img().shape and a.dtype == np.uint8
    assert np.array_equal(a, b)


def test_severity_increases_damage():
    for cond in ["low_light", "haze", "motion_blur", "jpeg", "rain"]:
        err = [np.abs(CORRUPT[cond](img(), PARAMS[cond], s, image_rng(0, cond, s, "x")).astype(float) - img()).mean()
               for s in range(3)]
        assert err[0] < err[2], cond


def test_downscale_labels_follow_the_content():
    h, w = 64, 96
    im = np.zeros((h, w, 3), np.uint8)
    im[16:48, 24:72] = 255                                  # box centred at (48, 32), 48 x 32
    rows = [["0", f"{48 / w}", f"{32 / h}", f"{48 / w}", f"{32 / h}"]]
    out = downscale(im, PARAMS["downscale"], 0, None)
    (c, cx, cy, bw, bh), = downscale_labels(rows, h, w, PARAMS["downscale"], 0)
    ys, xs = np.nonzero(out[..., 0] == 255)
    assert abs(float(cx) * w - (xs.min() + xs.max() + 1) / 2) <= 1
    assert abs(float(cy) * h - (ys.min() + ys.max() + 1) / 2) <= 1
    assert abs(float(bw) * w - (xs.max() - xs.min() + 1)) <= 1
    assert abs(float(bh) * h - (ys.max() - ys.min() + 1)) <= 1


def counts(tp, gt):
    tp, gt = np.array(tp, float), np.array(gt, float)
    return {"tp": tp, "gt": gt, "hit": np.zeros(len(tp)), "viol": np.zeros(len(tp)),
            "name": [str(i) for i in range(len(tp))]}


def test_paired_bootstrap_of_identical_sets_is_zero():
    a = counts([[1, 0], [2, 1], [0, 1]], [[2, 1], [2, 2], [1, 1]])
    ci = bootstrap_delta(a, a, [0, 1], [1], 200, 0, paired=True)
    assert ci["all"] == (0.0, 0.0) and ci["c0"] == (0.0, 0.0)


def test_paired_bootstrap_brackets_the_drop():
    ref = counts([[2]] * 20, [[2]] * 20)                    # recall 1.0
    cond = counts([[1]] * 20, [[2]] * 20)                   # recall 0.5 on every image
    lo, hi = bootstrap_delta(ref, cond, [0], [], 200, 0, paired=True)["c0"]
    assert lo == pytest.approx(-0.5) and hi == pytest.approx(-0.5)


def test_convert_sh17_maps_and_drops(tmp_path, monkeypatch):
    from PIL import Image

    src = tmp_path / "sh17"
    (src / "images").mkdir(parents=True)
    (src / "labels").mkdir()
    for stem in ["a", "b"]:
        Image.new("RGB", (100, 50)).save(src / "images" / f"{stem}.jpg")
    # person, glasses, safety-vest, shoes, head
    (src / "labels" / "a.txt").write_text("0 .5 .5 .2 .2\n8 .5 .5 .1 .1\n16 .5 .5 .3 .3\n14 .1 .1 .1 .1\n12 .2 .2 .1 .1\n")
    (src / "labels" / "b.txt").write_text("10 .5 .5 .2 .2\n")
    (src / "val_files.txt").write_text("a.jpg\n")
    monkeypatch.setattr(shift, "REPO_ROOT", tmp_path)
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "construction_ppe.yaml").write_text(
        "names: {0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none, 6: Person, 7: no_helmet, "
        "8: no_goggle, 9: no_gloves, 10: no_boots}\n")
    shift.convert("sh17", src, None, None, False)
    out = tmp_path / "datasets" / "sh17"
    assert sorted(p.name for p in (out / "images" / "test").iterdir()) == ["a.jpg"]      # val list only
    lines = (out / "labels" / "test" / "a.txt").read_text().split("\n")
    assert [ln.split()[0] for ln in lines if ln] == ["6", "2", "3"]                     # person, vest, boots
    summ = json.loads((out / "conversion.json").read_text())
    assert summ["dropped_instances"] == {"glasses": 1, "head": 1}


def test_convert_chv_voc(tmp_path, monkeypatch):
    from PIL import Image

    src = tmp_path / "chv"
    src.mkdir()
    Image.new("RGB", (200, 100)).save(src / "x.jpg")
    (src / "x.xml").write_text(
        "<annotation><size><width>200</width><height>100</height></size>"
        "<object><name>yellow</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>30</xmax><ymax>30</ymax></bndbox></object>"
        "<object><name>person</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object>"
        "</annotation>")
    monkeypatch.setattr(shift, "REPO_ROOT", tmp_path)
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "construction_ppe.yaml").write_text(
        "names: {0: helmet, 1: gloves, 2: vest, 3: boots, 4: goggles, 5: none, 6: Person, 7: no_helmet, "
        "8: no_goggle, 9: no_gloves, 10: no_boots}\n")
    shift.convert("chv", src, None, None, False)
    rows = [ln.split() for ln in (tmp_path / "datasets" / "chv" / "labels" / "test" / "x.txt").read_text().splitlines()]
    assert rows[0] == ["0", "0.100000", "0.200000", "0.100000", "0.200000"]
    assert rows[1][0] == "6"
