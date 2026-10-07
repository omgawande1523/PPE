"""Evaluate a weights file on Construction-PPE and append rows to results/results.csv.

One command evaluates the validation split and the requested split:

  1. Ultralytics validation (conf 0.001, NMS IoU 0.7): per-class mAP@0.5 and
     mAP@0.5:0.95, the protocol behind the training notebook's 0.639.
  2. The deployed pipeline (model.predict, as ppe.infer runs it) at an
     operating confidence chosen on the validation split only, by maximising
     the macro F1 over all classes. The test split never influences it.
     At that confidence: per-class precision, recall and F1 (IoU 0.5), a
     confusion matrix, and person-level violation recall and false-alert rate.

Per-class precision-recall curves (CSV and PNG) and the confusion matrix
(CSV and PNG) are written to results/figures/<run_id>/.

Run (from the repository root):
    python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml
    python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/x/weights/best.pt
    python -m ppe.eval --compare RUN_ID_A RUN_ID_B      # are two runs identical?
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.associate import ITEM_CLASSES, PERSON_CLASS, VIOLATION, Box, assign, evaluate_frame, item_states
from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, resolve_data_yaml, sha256_path
from ppe.infer import set_seed

RESULTS_CSV = REPO_ROOT / "results" / "results.csv"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
FIELDS = ["run_id", "date", "git_commit", "phase", "model", "weights", "precision_mode", "device", "dataset",
          "split", "condition", "class", "metric", "value", "n", "seed", "command", "notes"]
VIOLATION_CLASSES = ["no_helmet", "no_goggle", "no_gloves", "no_boots"]
CONF_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ----------------------------------------------------------------------------- geometry

def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def greedy_match(iou: np.ndarray, thr: float) -> dict[int, int]:
    """Rows are predictions already sorted by descending confidence.

    Each prediction takes the unmatched ground truth with the highest IoU >= thr.
    Processing in confidence order makes the matches of any confidence prefix
    identical to matching that prefix alone, so one pass serves every threshold.
    """
    matches: dict[int, int] = {}
    taken: set[int] = set()
    for i in range(iou.shape[0]):
        best, best_iou = -1, thr
        for j in range(iou.shape[1]):
            if j not in taken and iou[i, j] >= best_iou:
                best, best_iou = j, iou[i, j]
        if best >= 0:
            matches[i] = best
            taken.add(best)
    return matches


# ----------------------------------------------------------------------------- data

def load_labels(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO txt -> (class ids, xyxy pixels)."""
    if not label_path.exists():
        return np.zeros(0, int), np.zeros((0, 4))
    rows = [ln.split() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    arr = np.array(rows, dtype=float)
    cls = arr[:, 0].astype(int)
    cx, cy, bw, bh = arr[:, 1] * w, arr[:, 2] * h, arr[:, 3] * w, arr[:, 4] * h
    return cls, np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def split_images(data: dict, split: str) -> list[Path]:
    d = Path(data["path"]) / data[split]
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def label_for(img: Path) -> Path:
    parts = list(img.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


# ----------------------------------------------------------------------------- prediction

def predict_split(model, images: list[Path], cfg: dict) -> list[dict]:
    """Run the deployed predict path at a low confidence; keep everything for later thresholds."""
    out = []
    for i in range(0, len(images), int(cfg["batch"])):
        chunk = images[i:i + int(cfg["batch"])]
        results = model.predict([str(p) for p in chunk], conf=float(cfg["conf_floor"]), iou=float(cfg["nms_iou"]),
                                imgsz=int(cfg["imgsz"]), device=cfg["device"], max_det=300, verbose=False)
        for img, r in zip(chunk, results):
            h, w = r.orig_shape
            b = r.boxes
            order = np.argsort(-b.conf.cpu().numpy(), kind="stable")
            gt_cls, gt_xyxy = load_labels(label_for(img), w, h)
            out.append({
                "image": img.name,
                "pred_cls": b.cls.cpu().numpy().astype(int)[order],
                "pred_conf": b.conf.cpu().numpy()[order],
                "pred_xyxy": b.xyxy.cpu().numpy()[order],
                "gt_cls": gt_cls,
                "gt_xyxy": gt_xyxy,
            })
    return out


# ----------------------------------------------------------------------------- box-level metrics

def match_per_class(preds: list[dict], nc: int, iou_thr: float) -> dict[int, dict]:
    """Per class: confidence and TP flag of every prediction, and the GT count."""
    per = {c: {"conf": [], "tp": [], "n_gt": 0} for c in range(nc)}
    for fr in preds:
        for c in range(nc):
            pm = fr["pred_cls"] == c
            gm = fr["gt_cls"] == c
            per[c]["n_gt"] += int(gm.sum())
            if not pm.any():
                continue
            m = greedy_match(iou_matrix(fr["pred_xyxy"][pm], fr["gt_xyxy"][gm]), iou_thr)
            per[c]["conf"].extend(fr["pred_conf"][pm].tolist())
            per[c]["tp"].extend([i in m for i in range(int(pm.sum()))])
    for c in per:
        per[c]["conf"] = np.array(per[c]["conf"])
        per[c]["tp"] = np.array(per[c]["tp"], dtype=bool)
    return per


def prf_at(per_c: dict, t: float) -> tuple[int, int, int, float, float, float]:
    keep = per_c["conf"] >= t
    tp = int(per_c["tp"][keep].sum())
    fp = int(keep.sum()) - tp
    fn = per_c["n_gt"] - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / per_c["n_gt"] if per_c["n_gt"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return tp, fp, fn, p, r, f1


def select_conf(per: dict[int, dict]) -> tuple[float, float]:
    """Global confidence maximising macro F1 over classes with ground truth. Ties: lowest conf."""
    classes = [c for c in per if per[c]["n_gt"] > 0]
    best_t, best_f1 = float(CONF_GRID[0]), -1.0
    for t in CONF_GRID:
        f1 = float(np.mean([prf_at(per[c], t)[5] for c in classes]))
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def pr_curve(per_c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall at every prediction's confidence, from the matched list."""
    order = np.argsort(-per_c["conf"], kind="stable")
    conf = per_c["conf"][order]
    tpc = np.cumsum(per_c["tp"][order])
    fpc = np.cumsum(~per_c["tp"][order])
    recall = tpc / max(per_c["n_gt"], 1)
    precision = tpc / np.maximum(tpc + fpc, 1)
    return conf, precision, recall


def ap50_from_curve(per_c: dict) -> float:
    from ultralytics.utils.metrics import compute_ap

    if per_c["n_gt"] == 0 or len(per_c["conf"]) == 0:
        return 0.0
    _, p, r = pr_curve(per_c)
    return float(compute_ap(r, p)[0])


def confusion(preds: list[dict], nc: int, t: float, iou_thr: float) -> np.ndarray:
    """Rows = true class (last row = background), columns = predicted class (last = missed).

    Class-agnostic greedy matching at IoU >= iou_thr, predictions in confidence order.
    """
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for fr in preds:
        keep = fr["pred_conf"] >= t
        pc, px = fr["pred_cls"][keep], fr["pred_xyxy"][keep]
        m = greedy_match(iou_matrix(px, fr["gt_xyxy"]), iou_thr)
        matched_gt = set(m.values())
        for i, j in m.items():
            cm[fr["gt_cls"][j], pc[i]] += 1
        for i in range(len(pc)):
            if i not in m:
                cm[nc, pc[i]] += 1
        for j, g in enumerate(fr["gt_cls"]):
            if j not in matched_gt:
                cm[g, nc] += 1
    return cm


# ----------------------------------------------------------------------------- person-level metrics

def person_level(preds: list[dict], model_names: dict[int, str], t: float, required: list[str],
                 iou_thr: float) -> dict:
    """Violation recall and false-alert rate per person, the unit an alert is raised for.

    Ground truth: each labelled no_*/PPE box is assigned to the labelled person
    box containing its centre (the rule ppe.associate applies to predictions).
    A labelled person is a violator for an item if a no_* box is assigned to them.
    Prediction: ppe.associate.evaluate_frame on the boxes at confidence >= t.
    Predicted and labelled persons are matched greedily at IoU >= iou_thr.
    """
    # Ground-truth names use the model's spelling; class indices are identical.
    items = [i for i in required if ITEM_CLASSES.get(i)]
    gt_viol = Counter()          # labelled violators per item and "any"
    hit = Counter()              # of those, flagged for the same item by the pipeline
    alerts = Counter()           # predicted violation flags per item and "any"
    true_alerts = Counter()
    nonviol_gt = 0
    nonviol_flagged = 0
    n_gt_persons = 0
    n_gt_unmatched = 0
    n_gt_orphan_neg = 0
    for fr in preds:
        gt_boxes = [Box(model_names[int(c)], 1.0, tuple(x)) for c, x in zip(fr["gt_cls"], fr["gt_xyxy"])]
        gt_persons = [b for b in gt_boxes if b.name == PERSON_CLASS]
        gt_assigned, gt_orphans = assign(gt_persons, [b for b in gt_boxes if b.name != PERSON_CLASS])
        n_gt_orphan_neg += sum(b.name in VIOLATION_CLASSES for b in gt_orphans)
        gt_states = [item_states(own, items)[0] for own in gt_assigned]

        keep = fr["pred_conf"] >= t
        pboxes = [Box(model_names[int(c)], float(s), tuple(x))
                  for c, s, x in zip(fr["pred_cls"][keep], fr["pred_conf"][keep], fr["pred_xyxy"][keep])]
        p_results, _ = evaluate_frame(pboxes, items)
        # evaluate_frame keeps prediction order (descending confidence) for persons.
        m = greedy_match(iou_matrix(np.array([p.person.xyxy for p in p_results]).reshape(-1, 4),
                                    np.array([g.xyxy for g in gt_persons]).reshape(-1, 4)), iou_thr)
        inv = {j: i for i, j in m.items()}

        n_gt_persons += len(gt_persons)
        for j, st in enumerate(gt_states):
            pred_st = p_results[inv[j]].items if j in inv else {}
            if j not in inv:
                n_gt_unmatched += 1
            viol_items = [i for i in items if st.get(i) == VIOLATION]
            if viol_items:
                gt_viol["any"] += 1
                hit["any"] += int(any(pred_st.get(i) == VIOLATION for i in items))
            else:
                nonviol_gt += 1
                nonviol_flagged += int(any(pred_st.get(i) == VIOLATION for i in items))
            for i in viol_items:
                gt_viol[i] += 1
                hit[i] += int(pred_st.get(i) == VIOLATION)
        for pi, pr in enumerate(p_results):
            gst = gt_states[m[pi]] if pi in m else {}
            flagged = [i for i in items if pr.items.get(i) == VIOLATION]
            if flagged:
                alerts["any"] += 1
                true_alerts["any"] += int(any(gst.get(i) == VIOLATION for i in flagged))
            for i in flagged:
                alerts[i] += 1
                true_alerts[i] += int(gst.get(i) == VIOLATION)
    return {"items": items, "gt_viol": gt_viol, "hit": hit, "alerts": alerts, "true_alerts": true_alerts,
            "nonviol_gt": nonviol_gt, "nonviol_flagged": nonviol_flagged, "n_gt_persons": n_gt_persons,
            "n_gt_unmatched": n_gt_unmatched, "n_gt_orphan_neg": n_gt_orphan_neg}


def bootstrap(preds: list[dict], nc: int, names: dict[int, str], t: float, required: list[str], iou_thr: float,
              n_boot: int, seed: int) -> dict[tuple[str, str], tuple[float, float]]:
    """95% percentile intervals from resampling images with replacement.

    Images, not boxes, are the independent unit, so whole images are resampled.
    Returns {(class, metric): (low, high)} for macro AP@0.5 (predict path), each
    violation class's box recall at t, and person-level violation recall.
    """
    per_img = [match_per_class([fr], nc, iou_thr) for fr in preds]
    per_img_pl = [person_level([fr], names, t, required, iou_thr) for fr in preds]
    rng = np.random.default_rng(seed)
    samples: dict[tuple[str, str], list[float]] = {}
    for _ in range(n_boot):
        idx = rng.integers(0, len(preds), len(preds))
        agg = {c: {"conf": np.concatenate([per_img[i][c]["conf"] for i in idx]),
                   "tp": np.concatenate([per_img[i][c]["tp"] for i in idx]).astype(bool),
                   "n_gt": int(sum(per_img[i][c]["n_gt"] for i in idx))} for c in range(nc)}
        aps = [ap50_from_curve(agg[c]) for c in range(nc) if agg[c]["n_gt"] > 0]
        samples.setdefault(("all", "ap50_predict"), []).append(float(np.mean(aps)))
        for c in range(nc):
            if names[c] in VIOLATION_CLASSES and agg[c]["n_gt"] > 0:
                samples.setdefault((names[c], "recall"), []).append(prf_at(agg[c], t)[4])
        gv = sum(per_img_pl[i]["gt_viol"]["any"] for i in idx)
        if gv:
            hit = sum(per_img_pl[i]["hit"]["any"] for i in idx)
            samples.setdefault(("person_any_violation", "violation_recall"), []).append(hit / gv)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}


# ----------------------------------------------------------------------------- figures

def save_pr_curves(per: dict, names: dict[int, str], out: Path, t: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    with (out / "pr_curves.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "conf", "precision", "recall"])
        for c, pc in per.items():
            conf, p, r = pr_curve(pc)
            for a, b, d in zip(conf, p, r):
                w.writerow([names[c], f"{a:.6f}", f"{b:.6f}", f"{d:.6f}"])
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = plt.get_cmap("tab20").colors
    for c, pc in per.items():
        if pc["n_gt"] == 0:
            continue
        conf, p, r = pr_curve(pc)
        style = "-" if names[c] in VIOLATION_CLASSES else ":"
        line, = ax.plot(r, p, style, color=colours[(2 * c) % 20 + (c >= 10)],
                        lw=2 if names[c] in VIOLATION_CLASSES else 1,
                        label=f"{names[c]} (n={pc['n_gt']})")
        k = conf >= t
        if k.any():
            idx = int(np.nonzero(k)[0][-1])
            ax.plot(r[idx], p[idx], "o", color=line.get_color())
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"Per-class PR at IoU 0.5 (dots: operating conf {t:.2f})")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=120)
    plt.close(fig)


def save_confusion(cm: np.ndarray, names: dict[int, str], out: Path, t: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [names[i] for i in range(len(names))]
    with (out / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true\\pred"] + labels + ["missed"])
        for i, row in enumerate(cm):
            w.writerow([(labels + ["background"])[i]] + row.tolist())
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(np.log1p(cm), cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(cm.shape[1]), labels + ["missed"], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(cm.shape[0]), labels + ["background"], fontsize=8)
    ax.set(xlabel="Predicted", ylabel="True", title=f"Confusion matrix, conf >= {t:.2f}, IoU 0.5 (counts)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- bookkeeping

def git_commit() -> str:
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()

    sha = run("rev-parse", "--short=10", "HEAD") or "unknown"
    # results/ changes on every run, so it does not count as a dirty tree.
    dirty = run("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    return sha + ("+dirty" if dirty else "")


def append_rows(rows: list[dict]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists() or RESULTS_CSV.stat().st_size == 0
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def fmt(v) -> str:
    return f"{v:.6f}" if isinstance(v, float) else str(v)


def compare(run_a: str, run_b: str) -> int:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    key = lambda r: (r["split"], r["condition"], r["class"], r["metric"])  # noqa: E731
    a = {key(r): r for r in rows if r["run_id"] == run_a}
    b = {key(r): r for r in rows if r["run_id"] == run_b}
    if not a or not b:
        sys.exit(f"Run not found in {RESULTS_CSV.name}: {run_a if not a else run_b}")
    diffs = [(k, a[k]["value"], b.get(k, {}).get("value")) for k in a if a[k]["value"] != b.get(k, {}).get("value")]
    diffs += [(k, None, b[k]["value"]) for k in b if k not in a]
    print(f"{run_a}: {len(a)} rows, {run_b}: {len(b)} rows, differing: {len(diffs)}")
    for d in diffs[:20]:
        print("  ", d)
    return 0 if not diffs else 1


# ----------------------------------------------------------------------------- main

def evaluate(cfg: dict, command: str) -> list[dict]:
    from ultralytics import YOLO
    import ultralytics

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}. "
                 f"pip install -r requirements.txt")
    seed = int(cfg["seed"])
    set_seed(seed)
    weights = REPO_ROOT / cfg["weights"]
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")

    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    work = REPO_ROOT / "runs" / "eval" / run_id
    data_yaml, data = resolve_data_yaml(cfg["data"], work)

    model = YOLO(str(weights), task="detect")
    model_names = {int(k): v for k, v in model.names.items()}
    data_names = {int(k): v for k, v in data["names"].items()}
    if {k: v.lower() for k, v in model_names.items()} != {k: v.lower() for k, v in data_names.items()}:
        sys.exit(f"Model classes {model_names} do not match dataset classes {data_names}")
    nc = len(model_names)

    golden_note = ""
    if cfg.get("golden_manifest") and cfg["split"] == "test":
        from ppe.golden import verify

        verify(REPO_ROOT / cfg["golden_manifest"])
        golden_note = f"golden={Path(cfg['golden_manifest']).stem} verified"

    base = {
        "run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(),
        "phase": cfg["phase"], "model": cfg["model"], "weights": f"{cfg['weights']}@{sha256_path(weights)[:12]}",
        "precision_mode": cfg["precision_mode"], "device": cfg["device"], "dataset": cfg["dataset"],
        "condition": cfg.get("condition", "clean"), "seed": seed, "command": command,
    }
    rows: list[dict] = []

    def add(split, cls, metric, value, n, notes=""):
        rows.append({**base, "split": split, "class": cls, "metric": metric, "value": fmt(value), "n": n,
                     "notes": "; ".join(x for x in (notes, golden_note if split == "test" else "") if x)})

    splits = ["val"] if cfg["split"] == "val" else ["val", cfg["split"]]
    per_split_preds = {}
    for split in splits:
        images = split_images(data, split)
        print(f"[{split}] {len(images)} images: Ultralytics val ...", flush=True)
        set_seed(seed)
        m = model.val(data=str(data_yaml), split=split, imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                      conf=float(cfg["conf_floor"]), iou=float(cfg["nms_iou"]), device=cfg["device"],
                      half=cfg["precision_mode"] == "fp16", plots=False, verbose=False,
                      project=str(work), name=f"ul_{split}", exist_ok=True)
        n_inst = Counter()
        for img in images:
            for c in load_labels(label_for(img), 1, 1)[0]:
                n_inst[int(c)] += 1
        ul_note = ("Ultralytics val, conf 0.001, NMS IoU 0.7; P and R at the confidence maximising mean F1 "
                   "on this same split (Ultralytics convention), not at the operating confidence")
        mp, mr, map50, map5095 = m.box.mean_results()
        total = sum(n_inst.values())
        add(split, "all", "map50", float(map50), total, f"{len(images)} images; {ul_note}")
        add(split, "all", "map50_95", float(map5095), total, f"{len(images)} images; {ul_note}")
        add(split, "all", "precision_ul", float(mp), total, ul_note)
        add(split, "all", "recall_ul", float(mr), total, ul_note)
        for i, c in enumerate(m.box.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            name = model_names[int(c)]
            add(split, name, "map50", float(ap50), n_inst[int(c)], ul_note)
            add(split, name, "map50_95", float(ap), n_inst[int(c)], ul_note)
            add(split, name, "precision_ul", float(p), n_inst[int(c)], ul_note)
            add(split, name, "recall_ul", float(r), n_inst[int(c)], ul_note)

        print(f"[{split}] deployed predict path ...", flush=True)
        set_seed(seed)
        per_split_preds[split] = predict_split(model, images, cfg)

    # Operating confidence: chosen on val only.
    iou_thr = float(cfg["match_iou"])
    per_val = match_per_class(per_split_preds["val"], nc, iou_thr)
    t, f1_val = select_conf(per_val)
    add("val", "all", "operating_conf", t, sum(pc["n_gt"] for pc in per_val.values()),
        f"argmax over conf {CONF_GRID[0]:.2f}..{CONF_GRID[-1]:.2f} step 0.01 of macro F1 over classes "
        f"at IoU {iou_thr}; macro F1 at it = {f1_val:.4f}. Chosen on val only.")
    print(f"Operating confidence (chosen on val): {t:.2f}, val macro F1 {f1_val:.4f}")

    for split in splits:
        preds = per_split_preds[split]
        per = per_val if split == "val" else match_per_class(preds, nc, iou_thr)
        op = f"deployed predict path, conf >= {t:.2f} (chosen on val), match IoU {iou_thr}"
        f1s, tp_all, fp_all, fn_all = [], 0, 0, 0
        for c, pc in per.items():
            name = model_names[c]
            tp, fp, fn, p, r, f1 = prf_at(pc, t)
            tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
            f1s.append(f1)
            note = f"{op}; tp={tp} fp={fp} fn={fn}"
            add(split, name, "precision", p, pc["n_gt"], note)
            add(split, name, "recall", r, pc["n_gt"], note)
            add(split, name, "f1", f1, pc["n_gt"], note)
            add(split, name, "ap50_predict", ap50_from_curve(pc), pc["n_gt"],
                "AP@0.5 recomputed from the predict path (cross-check of Ultralytics map50)")
        n_all = tp_all + fn_all
        add(split, "all", "precision", tp_all / max(tp_all + fp_all, 1), n_all, f"{op}; micro over classes")
        add(split, "all", "recall", tp_all / max(n_all, 1), n_all, f"{op}; micro over classes")
        add(split, "all", "macro_f1", float(np.mean(f1s)), n_all, op)
        viol_r = [prf_at(per[c], t)[4] for c in per if model_names[c] in VIOLATION_CLASSES]
        add(split, "violation_classes", "macro_recall", float(np.mean(viol_r)),
            sum(per[c]["n_gt"] for c in per if model_names[c] in VIOLATION_CLASSES),
            f"{op}; unweighted mean of {', '.join(VIOLATION_CLASSES)} box recall")

        pl = person_level(preds, model_names, t, list(cfg["required_items"]), iou_thr)
        pnote = (f"{op}; per labelled person; {pl['n_gt_persons']} labelled persons, "
                 f"{pl['n_gt_unmatched']} not detected; {pl['n_gt_orphan_neg']} labelled no_* boxes "
                 f"outside every labelled person are not counted")
        for key in ["any"] + pl["items"]:
            cls = "person_any_violation" if key == "any" else f"person_{ITEM_CLASSES[key]}"
            n_v = pl["gt_viol"][key]
            add(split, cls, "violation_recall", pl["hit"][key] / n_v if n_v else float("nan"), n_v,
                f"{pnote}; flagged {pl['hit'][key]} of {n_v} labelled violators"
                + (" for any item (an alert for the person, not necessarily the right item)" if key == "any"
                   else " for this item; the no_* box itself needs no IoU match, only the person"))
            n_a = pl["alerts"][key]
            add(split, cls, "false_alert_rate", (n_a - pl["true_alerts"][key]) / n_a if n_a else float("nan"),
                n_a, f"{pnote}; share of {n_a} predicted violation flags that do not match a labelled violation "
                     f"of the same item (1 - alert precision)")
        add(split, "person_any_violation", "nonviolator_flag_rate",
            pl["nonviol_flagged"] / pl["nonviol_gt"] if pl["nonviol_gt"] else float("nan"), pl["nonviol_gt"],
            f"{pnote}; labelled persons with no violation who were flagged for any item")

        n_boot = int(cfg.get("bootstrap", 0))
        if n_boot:
            print(f"[{split}] bootstrap, {n_boot} resamples ...", flush=True)
            per_all = [per[c] for c in per if per[c]["n_gt"] > 0]
            add(split, "all", "ap50_predict", float(np.mean([ap50_from_curve(pc) for pc in per_all])),
                sum(pc["n_gt"] for pc in per_all), "macro AP@0.5 from the predict path; compare with map50")
            ci = bootstrap(preds, nc, model_names, t, pl["items"], iou_thr, n_boot, seed)
            bnote = (f"95% percentile interval, {n_boot} bootstrap resamples of the {len(preds)} images, "
                     f"seed {seed}")
            for (cls, metric), (lo, hi) in sorted(ci.items()):
                add(split, cls, f"{metric}_ci95_low", lo, len(preds), bnote)
                add(split, cls, f"{metric}_ci95_high", hi, len(preds), bnote)

        fig_dir = FIGURES_DIR / run_id / split
        save_pr_curves(per, model_names, fig_dir, t)
        save_confusion(confusion(preds, nc, t, iou_thr), model_names, fig_dir, t)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.eval", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="eval YAML, e.g. configs/eval/baseline_yolo11s_test.yaml")
    ap.add_argument("--weights", help="override the weights in the config (path relative to the repo root)")
    ap.add_argument("--run-name", help="override run_name (use with --weights)")
    ap.add_argument("--model", help="override the model label, e.g. yolo11n (use with --weights)")
    ap.add_argument("--compare", nargs=2, metavar="RUN_ID", help="check two runs in results.csv are identical")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if not args.config:
        ap.error("--config is required")
    cfg_path = Path(args.config)
    with (cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg["split"] == "train":
        sys.exit("Evaluating on train is not a result; use val or test.")
    if args.weights:
        cfg["weights"] = Path(args.weights).as_posix()
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.model:
        cfg["model"] = args.model
    command = "python -m ppe.eval " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    rows = evaluate(cfg, command)
    append_rows(rows)
    run_id = rows[0]["run_id"]
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {run_id}")
    show = {("all", "map50"), ("all", "recall"), ("all", "precision"), ("person_any_violation", "violation_recall"),
            ("person_any_violation", "false_alert_rate")} | {(c, "recall") for c in VIOLATION_CLASSES}
    for r in rows:
        if (r["class"], r["metric"]) in show:
            print(f"  {r['split']:5s} {r['class']:22s} {r['metric']:18s} {float(r['value']):.4f}  n={r['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
