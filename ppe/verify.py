"""Stage 4 of the loop: the promotion gate, the promotion log and rollback.

The candidate replaces the incumbent only if ALL of these hold (brief, Stage 4):

  1. Golden set: recall on every no_* class is not lower than the incumbent's.
  2. Golden set: overall mAP@0.5 does not fall by more than `map50_tolerance`,
     fixed in the experiment config before the run.
  3. New-site sample: micro recall over all classes improves (strictly higher).

Both models are measured in the same run with the same code: per-class recall at
the operating confidence (Phase 1's 0.28, never re-chosen) through the deployed
predict path, and mAP@0.5 through Ultralytics validation (conf 0.001, NMS IoU 0.7),
exactly as ppe.eval does. The golden manifest is verified (sha256) before use.
The golden set is only evaluated here; nothing is fitted or tuned on it.

Every decision is appended to results/promotions.csv. Promoted weights are copied
to the run's promoted/ folder, which keeps the last `keep_last` promotions;
rollback() restores the previous one.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from ppe.data import REPO_ROOT, sha256_path
from ppe.eval import VIOLATION_CLASSES, label_for, load_labels
from ppe.monitor import list_dir, predict_frames, recall_of, tp_gt

PROMOTIONS_CSV = REPO_ROOT / "results" / "promotions.csv"
PROMOTION_FIELDS = ["date", "run_id", "scenario", "candidate", "incumbent", "candidate_recall", "incumbent_recall",
                    "candidate_map50", "incumbent_map50", "candidate_site_recall", "incumbent_site_recall",
                    "map50_tolerance", "pseudo_label_error_rate", "pseudo_label_error_rate_human", "human_minutes",
                    "decision", "reason", "command"]


def measure(weights: Path, images_dir: Path, cfg: dict, work: Path, tag: str) -> dict:
    """Per-class recall at the operating confidence, and Ultralytics mAP@0.5, of one weights file on one folder."""
    from ultralytics import YOLO

    from ppe.infer import set_seed
    from ppe.shift import runtime_yaml

    seed = int(cfg["seed"])
    model = YOLO(str(weights), task="detect")
    names = {int(k): v for k, v in model.names.items()}
    nc = len(names)
    images = list_dir(images_dir)
    # Ultralytics expects images/<split> and labels/<split> side by side; runtime_yaml points all splits at
    # <root>/images/test, so link the folder there when it is named differently.
    root = images_dir.parent.parent
    if images_dir.name != "test":
        root = work / f"link_{tag}"
        (root / "images").mkdir(parents=True, exist_ok=True)
        for kind in ("images", "labels"):
            dst = root / kind / "test"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to((images_dir.parent.parent / kind / images_dir.name).resolve())
    data_yaml = runtime_yaml(None, root, work / f"data_{tag}", names)
    set_seed(seed)
    m = model.val(data=str(data_yaml), split="test", imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]), conf=0.001,
                  iou=float(cfg["nms_iou"]), device=cfg["device"], plots=False, verbose=False,
                  project=str(work), name=f"ul_{tag}", exist_ok=True)
    set_seed(seed)
    dets = predict_frames(model, images, cfg, conf=float(cfg["operating_conf"]))
    tp, gt = [], []
    for img, d in zip(images, dets):
        h, w = d["hw"]
        a, b = tp_gt(d, *load_labels(label_for(img), w, h), nc, float(cfg["operating_conf"]), float(cfg["match_iou"]))
        tp.append(a)
        gt.append(b)
    tp, gt = np.stack(tp), np.stack(gt)
    rec = {names[c]: (float(tp[:, c].sum() / gt[:, c].sum()) if gt[:, c].sum() else float("nan")) for c in range(nc)}
    rec["all"] = recall_of(tp, gt)
    viol = [c for c in range(nc) if names[c] in VIOLATION_CLASSES]
    rec["violation_pooled"] = recall_of(tp, gt, viol)
    return {"recall": rec, "tp": {names[c]: int(tp[:, c].sum()) for c in range(nc)},
            "gt": {names[c]: int(gt[:, c].sum()) for c in range(nc)}, "map50": float(m.box.map50),
            "map50_95": float(m.box.map), "n_images": len(images), "weights_sha": sha256_path(weights)[:12]}


def gate(cand: dict, inc: dict, cand_site: dict, inc_site: dict, tol: float) -> tuple[bool, list[str]]:
    """The three promotion rules. Returns (promote, one line per rule with its numbers)."""
    lines, ok = [], True
    for c in VIOLATION_CLASSES:
        a, b = cand["recall"][c], inc["recall"][c]
        good = bool(a >= b - 1e-12)
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'} golden recall {c}: {a:.4f} vs {b:.4f} "
                     f"({cand['tp'][c]}/{cand['gt'][c]} vs {inc['tp'][c]}/{inc['gt'][c]})")
    drop = inc["map50"] - cand["map50"]
    good = bool(drop <= tol + 1e-12)
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'} golden mAP50 {cand['map50']:.4f} vs {inc['map50']:.4f} "
                 f"(drop {drop:+.4f}, tolerance {tol})")
    a, b = cand_site["recall"]["all"], inc_site["recall"]["all"]
    good = bool(a > b)
    ok &= good
    lines.append(f"{'PASS' if good else 'FAIL'} new-site recall {a:.4f} vs {b:.4f} must improve")
    return ok, lines


def log_decision(row: dict) -> None:
    PROMOTIONS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if PROMOTIONS_CSV.exists() and PROMOTIONS_CSV.stat().st_size:
        with PROMOTIONS_CSV.open(encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            if rd.fieldnames != PROMOTION_FIELDS:   # older header (Phase 0 placeholder): rewrite with the new one
                rows = [{k: r.get(k, "") for k in PROMOTION_FIELDS} for r in rd]
                with PROMOTIONS_CSV.open("w", newline="", encoding="utf-8") as out:
                    w = csv.DictWriter(out, fieldnames=PROMOTION_FIELDS)
                    w.writeheader()
                    w.writerows(rows)
    new = not PROMOTIONS_CSV.exists() or PROMOTIONS_CSV.stat().st_size == 0
    with PROMOTIONS_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PROMOTION_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in PROMOTION_FIELDS})


def recall_str(m: dict) -> str:
    """Per-class recall as 'class=value(tp/gt)' pairs, for one promotions.csv cell."""
    return ";".join(f"{c}={m['recall'][c]:.4f}({m['tp'][c]}/{m['gt'][c]})" for c in m["tp"]) + \
        f";all={m['recall']['all']:.4f}"


# ----------------------------------------------------------------------------- weights history and rollback

def promote(state: dict, candidate: Path, promoted_dir: Path, keep_last: int, tag: str) -> Path:
    """Copy the candidate into promoted/, make it current, keep the last `keep_last` promotions."""
    promoted_dir.mkdir(parents=True, exist_ok=True)
    dst = promoted_dir / f"{tag}.pt"
    shutil.copy2(candidate, dst)
    hist = state.setdefault("history", [])
    hist.append({"weights": str(dst.relative_to(REPO_ROOT)), "sha256": sha256_path(dst)[:12], "tag": tag,
                 "previous": state["current"]})
    state["current"] = str(dst.relative_to(REPO_ROOT))
    while len(hist) > keep_last:
        old = hist.pop(0)
        p = REPO_ROOT / old["weights"]
        if p.parent == promoted_dir and p.exists():
            p.unlink()
    return dst


def rollback(state: dict) -> str:
    """Return to the weights in use before the last promotion."""
    hist = state.get("history") or []
    if not hist:
        raise ValueError("nothing to roll back: no promotion recorded")
    last = hist.pop()
    state["current"] = last["previous"]
    return state["current"]


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
