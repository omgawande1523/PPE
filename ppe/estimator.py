"""Reliability estimator (Phase 4): label-free signals -> estimated recall drop, with its error.

One command computes the six signals (ppe.signals) on every Phase 2 condition,
fits the estimator, scores it and the two confidence baselines, measures the
lead time of the alarm on synthetic streams, and appends every number to
results/results.csv:

    python -m ppe.estimator --config configs/estimator/p4_estimator_yolo11s.yaml
    python -m ppe.estimator --table RUN_ID          # markdown tables of a run from results.csv

Protocol (all fixed in the config before any estimate was scored):
  * Cells: clean + 6 conditions x 3 severities, built with ppe.shift's corruption
    code and Phase 2 parameters, on the val split (fit) and on the test split
    (golden set v1; scoring only).
  * Windows: 20 seeded subsets of 40 images per split, the same subsets in every
    cell, plus the full cell. True recall drop of a window = recall of its images
    clean minus recall of the same images under the condition, at the operating
    confidence (positive = recall lost).
  * Estimator: ridge regression on the standardised signals.
  * Hold-out: leave-one-condition-out. For each condition, fit on val windows of
    clean + the other five conditions, score on test windows of the held-out
    condition. This holds out images and shift type; it is not a site hold-out.
    Converted external sites on disk are scored too (config: holdout_sites).
  * Baselines (Guillory et al. 2021; Garg et al. 2022), source = clean val:
      AC  (average confidence, as difference of confidences):
          est drop = mean candidate confidence on source - on the window.
      ATC (average thresholded confidence): threshold t such that the share of
          source candidates with conf >= t equals source recall;
          est drop = source recall - share of window candidates with conf >= t.
    Neither baseline is fitted on shifted data.
  * Lead time: streams of 8 clean windows then 10 windows with a rising share of
    images under a condition. A real drop starts at the first window whose true
    drop >= 0.05; the monitor alarms at the first window whose estimate >= 0.05.
    Delay = alarm window - drop window (positive = late, negative = early).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shlex
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_path
from ppe.eval import FIELDS, FIGURES_DIR, RESULTS_CSV, VIOLATION_CLASSES, append_rows, fmt, git_commit
from ppe.infer import set_seed
from ppe.shift import CONDITIONS, SEVERITIES, make_cell
from ppe.signals import SIGNAL_GROUPS, SIGNALS, Extractor, Reference, pool

WORK = REPO_ROOT / "runs" / "estimator"
METHODS = ["signals", "ac", "atc"]


# ============================================================================= helpers

def load_yaml(p: str | Path) -> dict:
    p = Path(p)
    with (p if p.is_absolute() else REPO_ROOT / p).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_dir(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []


def cells() -> list[str]:
    return ["clean"] + [f"{c}_s{s + 1}" for c in CONDITIONS for s in range(SEVERITIES)]


def cond_of(cell: str) -> str:
    return cell.rsplit("_s", 1)[0] if cell != "clean" else "clean"


def signal_key(cfg: dict, shift_cfg: dict) -> str:
    """Hash of every setting that changes a per-image record."""
    keys = ["weights", "imgsz", "batch", "seed", "nms_iou", "match_iou", "operating_conf", "cand_conf",
            "feature_layer", "augment", "temporal", "person_detector", "device", "data", "reference"]
    blob = {k: cfg[k] for k in keys}
    blob["weights_sha"] = sha256_path(REPO_ROOT / cfg["weights"])
    blob["corruptions"] = shift_cfg["corruptions"]
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


def recall(recs: list[dict], idx, cls: list[int]) -> float:
    tp = sum(recs[i]["tp"][cls].sum() for i in idx)
    gt = sum(recs[i]["gt"][cls].sum() for i in idx)
    return float(tp / gt) if gt else float("nan")


# ============================================================================= stage 1: records

def extract(cfg: dict, fresh: bool) -> tuple[dict, Reference, str]:
    """Per-image records of every cell of both splits (cached), plus the training-site reference."""
    shift_cfg = load_yaml(cfg["shift_config"])
    key = signal_key(cfg, shift_cfg)
    rec_dir = WORK / "records" / key
    rec_dir.mkdir(parents=True, exist_ok=True)
    data = load_yaml(cfg["data"])
    root = REPO_ROOT / data["path"]
    seed = int(cfg["seed"])
    ex = None

    def cached(name: str, fn):
        nonlocal ex
        f = rec_dir / f"{name}.pkl"
        if f.exists() and not fresh:
            return pickle.loads(f.read_bytes())
        if ex is None:
            set_seed(seed)
            ex = Extractor(cfg)
        t0 = time.time()
        out = fn()
        f.write_bytes(pickle.dumps(out))
        print(f"  {name}: {len(out)} images, {time.time() - t0:.0f} s", flush=True)
        return out

    split_dir = lambda s: root / data[s]
    if not split_dir(cfg["fit_split"]).is_dir():
        sys.exit(f"Dataset not found at {split_dir(cfg['fit_split'])}. Run: python -m ppe.data --download construction-ppe")
    if cfg.get("golden_manifest"):
        from ppe.golden import verify

        verify(REPO_ROOT / cfg["golden_manifest"])

    r = cfg["reference"]
    train = list_dir(split_dir(r["split"]))
    pick = sorted(np.random.default_rng(seed).choice(len(train), int(r["n_images"]), replace=False))
    ref_recs = cached(f"reference_{r['split']}_{r['n_images']}",
                      lambda: (ex.records([train[i] for i in pick], "reference", light=True)))
    ref = Reference(ref_recs)

    recs: dict[tuple[str, str], list[dict]] = {}
    for split in (cfg["fit_split"], cfg["eval_split"]):
        images = list_dir(split_dir(split))
        for cell in cells():
            def build(split=split, cell=cell, images=images):
                if cell == "clean":
                    return ex.records(images, f"{split}_clean")
                cond, s = cond_of(cell), int(cell.rsplit("_s", 1)[1]) - 1
                out_root = WORK / "cells" / key / f"{split}_{cell}"
                make_cell(images, cond, s, shift_cfg["corruptions"][cond], int(shift_cfg["seed"]), out_root)
                return ex.records(list_dir(out_root / "images" / "test"), f"{split}_{cell}")
            recs[(split, cell)] = cached(f"{split}__{cell}", build)

    for site, spec in (cfg.get("holdout_sites") or {}).items():
        sd = load_yaml(spec["data"])
        imgs = list_dir(REPO_ROOT / sd["path"] / sd["test"])
        if not imgs:
            print(f"  site {site}: not on disk (see results/missing.md); skipped")
            continue
        recs[("site", site)] = cached(f"site__{site}_{len(imgs)}", lambda imgs=imgs, site=site:
                                      ex.records(imgs, f"site_{site}"))

    for rs in recs.values():
        for rec in rs:
            rec["feat_dist"] = ref.feat_dist(rec["feat"])
    for split in (cfg["fit_split"], cfg["eval_split"]):
        stems = [Path(x["image"]).stem for x in recs[(split, "clean")]]
        for cell in cells():
            if [Path(x["image"]).stem for x in recs[(split, cell)]] != stems:
                sys.exit(f"{split} {cell}: images do not line up with the clean cell")
    return recs, ref, key


# ============================================================================= stage 2: windows

class Baselines:
    """AC (difference of confidences) and ATC with clean val as the source, per target class set."""

    def __init__(self, src: list[dict], cls: list[int], cls_all: bool):
        self.cls, self.cls_all = set(cls), cls_all
        conf = self._conf(src, range(len(src)))
        self.r_src = recall(src, range(len(src)), cls)
        ok = len(conf) > 0 and np.isfinite(self.r_src)
        self.ac_src = float(conf.mean()) if ok else float("nan")
        self.t = float(np.quantile(conf, 1 - self.r_src)) if ok else float("nan")

    def _conf(self, recs, idx) -> np.ndarray:
        parts = [r["cand_conf"] if self.cls_all else r["cand_conf"][np.isin(r["cand_cls"], list(self.cls))]
                 for r in (recs[i] for i in idx)]
        return np.concatenate(parts) if parts else np.zeros(0)

    def __call__(self, recs, idx) -> dict[str, float]:
        c = self._conf(recs, idx)
        return {"ac": self.ac_src - (float(c.mean()) if len(c) else 0.0),
                "atc": self.r_src - (float((c >= self.t).mean()) if len(c) else 0.0)}


def window_sets(n: int, cfg: dict, split: str) -> list[np.ndarray]:
    rng = np.random.default_rng([int(cfg["seed"]), zlib.crc32(split.encode())])
    w = min(int(cfg["window_size"]), n)
    return [np.sort(rng.choice(n, w, replace=False)) for _ in range(int(cfg["windows_per_cell"]))]


def ridge(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


# ============================================================================= stage 3: fit, score, stream

def run(cfg: dict, command: str, fresh: bool) -> list[dict]:
    import ultralytics

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}.")
    seed = int(cfg["seed"])
    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    print(f"Run {run_id}: computing per-image signals (cached in runs/estimator/records)")
    recs, ref, key = extract(cfg, fresh)

    from ultralytics import YOLO
    names = {int(k): v for k, v in YOLO(str(REPO_ROOT / cfg["weights"])).names.items()}
    by_name = {v: k for k, v in names.items()}
    nc = len(names)
    targets = {k: (list(range(nc)) if v == "all" else [by_name[c] for c in v]) for k, v in cfg["targets"].items()}
    FS, ES = cfg["fit_split"], cfg["eval_split"]
    alpha = float(cfg["ridge_alpha"])
    W = int(cfg["window_size"])

    base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": cfg["phase"],
            "model": cfg["model"], "weights": f"{cfg['weights']}@{sha256_path(REPO_ROOT / cfg['weights'])[:12]}",
            "precision_mode": cfg["precision_mode"], "device": cfg["device"], "seed": seed, "command": command}
    rows: list[dict] = []

    def add(dataset, split, cond, cls, metric, value, n, notes):
        rows.append({**base, "dataset": dataset, "split": split, "condition": cond, "class": cls, "metric": metric,
                     "value": fmt(value), "n": n, "notes": notes})

    # ---- window table: one entry per (split, cell, window)
    win = {s: window_sets(len(recs[(s, "clean")]), cfg, s) for s in (FS, ES)}
    base_fns = {t: Baselines(recs[(FS, "clean")], cls, cfg["targets"][t] == "all") for t, cls in targets.items()}
    entries = []
    for split in (FS, ES):
        n = len(recs[(split, "clean")])
        for cell in cells():
            for w, idx in [(i, ix) for i, ix in enumerate(win[split])] + [("full", np.arange(n))]:
                r_c, r_x = recs[(split, "clean")], recs[(split, cell)]
                e = {"split": split, "cell": cell, "cond": cond_of(cell), "w": w, "n": len(idx),
                     "x": np.array([pool(r_x, idx, ref)[k] for k in SIGNALS])}
                for t, cls in targets.items():
                    e[f"true_{t}"] = recall(r_c, idx, cls) - recall(r_x, idx, cls)
                    e[f"gt_{t}"] = int(sum(r_x[i]["gt"][cls].sum() for i in idx))
                    for m, v in base_fns[t](r_x, idx).items():
                        e[f"{m}_{t}"] = v
                entries.append(e)
    print(f"  {len(entries)} windows")

    feat_sets = {"signals": SIGNALS, **{f"only_{g}": f for g, f in SIGNAL_GROUPS.items()}}

    def fit(t: str, held: str | None, feats: list[str]):
        cols = [SIGNALS.index(f) for f in feats]
        tr = [e for e in entries if e["split"] == FS and e["w"] != "full" and e["cond"] != held
              and np.isfinite(e[f"true_{t}"])]
        m = ridge(alpha).fit(np.stack([e["x"][cols] for e in tr]), [e[f"true_{t}"] for e in tr])
        return m, cols

    # ---- leave-one-condition-out predictions on test windows (clean test cell: model fit on all conditions)
    models: dict = {}
    for t in targets:
        for fs_name, feats in feat_sets.items():
            for held in CONDITIONS + [None]:
                models[(t, fs_name, held)] = fit(t, held, feats)
    for e in entries:
        if e["split"] != ES:
            continue
        for t in targets:
            for fs_name in feat_sets:
                m, cols = models[(t, fs_name, None if e["cond"] == "clean" else e["cond"])]
                e[f"{fs_name}_{t}"] = float(m.predict(e["x"][cols][None])[0])
                m_all, _ = models[(t, fs_name, None)]
                e[f"{fs_name}_imgholdout_{t}"] = float(m_all.predict(e["x"][cols][None])[0])

    ds = "construction-ppe"
    proto = (f"leave-one-condition-out: ridge (alpha {alpha}) fitted on {FS} windows of clean + the other five "
             f"conditions, scored on {ES} windows of the held-out condition; NOT a site hold-out")
    win_note = f"{len(win[ES])} windows of {W} images per cell, same images in every cell"
    rng = np.random.default_rng(seed)

    # ---- summary error per method over the 18 shifted test cells
    shifted = [c for c in cells() if c != "clean"]
    method_cols = {"signals": "signals", "ac": "ac", "atc": "atc",
                   **{f"only_{g}": f"only_{g}" for g in SIGNAL_GROUPS}, "signals_imgholdout": "signals_imgholdout"}
    summary: dict = {}
    for t in targets:
        for mname, col in method_cols.items():
            ws = [e for e in entries if e["split"] == ES and e["cell"] in shifted and e["w"] != "full"
                  and np.isfinite(e[f"true_{t}"])]
            err = np.array([e[f"{col}_{t}"] - e[f"true_{t}"] for e in ws])
            cell_of = np.array([e["cell"] for e in ws])
            full = [e for e in entries if e["split"] == ES and e["cell"] in shifted and e["w"] == "full"]
            ferr = np.array([e[f"{col}_{t}"] - e[f"true_{t}"] for e in full])
            boots = []
            for _ in range(1000):   # cluster bootstrap over the 18 cells
                pick = rng.choice(shifted, len(shifted))
                boots.append(np.mean(np.concatenate([np.abs(err[cell_of == c]) for c in pick])))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            truth = np.array([e[f"true_{t}"] for e in full])
            est = np.array([e[f"{col}_{t}"] for e in full])
            from scipy.stats import spearmanr
            rho = float(spearmanr(truth, est).correlation)
            summary[(t, mname)] = (float(np.abs(err).mean()), lo, hi, float(err.mean()), float(np.abs(ferr).mean()), rho)
            what = {"signals": proto, "signals_imgholdout": f"ridge fitted on all {FS} windows (all conditions), "
                    f"scored on {ES}: image hold-out only, the condition was seen in the fit",
                    "ac": "AC baseline (difference of confidences), source clean val, not fitted on shift",
                    "atc": "ATC baseline, threshold from clean val, not fitted on shift"}.get(
                mname, f"ablation: {proto}, using only {mname[5:]}")
            n_w = len(ws)
            add(ds, ES, "shifted_18_cells", t, f"mae_window_{mname}", summary[(t, mname)][0], n_w,
                f"{what}; mean |est - true| recall drop over {n_w} windows ({win_note}); 18 cells")
            add(ds, ES, "shifted_18_cells", t, f"mae_window_{mname}_ci95_low", lo, n_w,
                "95% percentile interval, 1000 bootstrap resamples of the 18 cells (windows kept with their cell)")
            add(ds, ES, "shifted_18_cells", t, f"mae_window_{mname}_ci95_high", hi, n_w,
                "95% percentile interval, 1000 bootstrap resamples of the 18 cells (windows kept with their cell)")
            add(ds, ES, "shifted_18_cells", t, f"bias_window_{mname}", summary[(t, mname)][3], n_w,
                f"{what}; mean (est - true); positive = overestimates the drop")
            add(ds, ES, "shifted_18_cells", t, f"mae_cell_{mname}", summary[(t, mname)][4], len(full),
                f"{what}; mean |est - true| over the 18 full cells ({len(recs[(ES, 'clean')])} images each)")
            add(ds, ES, "shifted_18_cells", t, f"spearman_cell_{mname}", rho, len(full),
                f"{what}; rank correlation of estimated and true drop over the 18 full cells")

    # ---- per cell: true drop and estimates on the full cell, each with a 95% interval over images
    n_bt, n_be = int(cfg["bootstrap_true"]), int(cfg["bootstrap_est"])
    n_es = len(recs[(ES, "clean")])
    percell: dict = {}
    for cell in cells():
        r_c, r_x = recs[(ES, "clean")], recs[(ES, cell)]
        full = next(e for e in entries if e["split"] == ES and e["cell"] == cell and e["w"] == "full")
        held = None if cell == "clean" else cond_of(cell)
        bidx = [rng.integers(0, n_es, n_es) for _ in range(max(n_bt, n_be))]
        for t, cls in targets.items():
            tb = [recall(r_c, b, cls) - recall(r_x, b, cls) for b in bidx[:n_bt]]
            m, cols = models[(t, "signals", held)]
            eb = {"signals": [], "ac": [], "atc": []}
            for b in bidx[:n_be]:
                eb["signals"].append(float(m.predict(np.array([pool(r_x, b, ref)[k] for k in SIGNALS])[cols][None])[0]))
                for k, v in base_fns[t](r_x, b).items():
                    eb[k].append(v)
            ci = lambda v: tuple(np.nanpercentile(v, [2.5, 97.5]))
            percell[(t, cell)] = {"true": (full[f"true_{t}"], *ci(tb))}
            gt = full[f"gt_{t}"]
            ctx = f"{n_es} images; paired against the same images clean; positive = recall lost"
            add(ds, ES, cell, t, "true_recall_drop", full[f"true_{t}"], gt,
                f"{ctx}; recall at operating conf {cfg['operating_conf']}, match IoU {cfg['match_iou']}")
            add(ds, ES, cell, t, "true_recall_drop_ci95_low", percell[(t, cell)]["true"][1], gt,
                f"95% percentile interval, {n_bt} paired bootstrap resamples of images")
            add(ds, ES, cell, t, "true_recall_drop_ci95_high", percell[(t, cell)]["true"][2], gt,
                f"95% percentile interval, {n_bt} paired bootstrap resamples of images")
            for mname in METHODS:
                est = full[f"{mname}_{t}"]
                percell[(t, cell)][mname] = (est, *ci(eb[mname]))
                ws = [e for e in entries if e["split"] == ES and e["cell"] == cell and e["w"] != "full"]
                wm = float(np.mean([abs(e[f"{mname}_{t}"] - e[f"true_{t}"]) for e in ws]))
                what = {"signals": (proto if cell != "clean" else f"ridge fitted on all {FS} windows")}.get(
                    mname, f"{mname.upper()} baseline, source clean {FS}")
                add(ds, ES, cell, t, f"est_recall_drop_{mname}", est, gt, f"{what}; full cell, no labels used")
                add(ds, ES, cell, t, f"est_recall_drop_{mname}_ci95_low", percell[(t, cell)][mname][1], gt,
                    f"95% percentile interval, {n_be} bootstrap resamples of images (signals recomputed)")
                add(ds, ES, cell, t, f"est_recall_drop_{mname}_ci95_high", percell[(t, cell)][mname][2], gt,
                    f"95% percentile interval, {n_be} bootstrap resamples of images (signals recomputed)")
                add(ds, ES, cell, t, f"mae_window_{mname}", wm, len(ws), f"{what}; {win_note}")
        print(f"  cell {cell}: done", flush=True)

    # ---- signal values per cell (full test cell), for the paper's signal table
    for cell in cells():
        full = next(e for e in entries if e["split"] == ES and e["cell"] == cell and e["w"] == "full")
        for k, v in zip(SIGNALS, full["x"]):
            add(ds, ES, cell, "all", f"signal_{k}", v, full["n"],
                f"label-free signal on the full cell; reference = {ref.n} seeded {cfg['reference']['split']} images")

    # ---- lead time on streams
    sc = cfg["stream"]
    thr = float(sc["drop_threshold"])
    n_clean, n_ramp = int(sc["clean_windows"]), int(sc["ramp_windows"])
    stream_res: dict = {}
    for cond in CONDITIONS:
        for sev in sc["severities"]:
            cell = f"{cond}_s{sev}"
            for t, cls in targets.items():
                m, cols = models[(t, "signals", cond)]
                srng = np.random.default_rng([seed, zlib.crc32(cell.encode()), zlib.crc32(t.encode())])
                out = {mm: {"delay": [], "missed": 0, "false_alarm": 0} for mm in METHODS}
                n_drop = 0
                curves = {mm: np.zeros(n_clean + n_ramp) for mm in METHODS + ["true"]}
                for _ in range(int(sc["repeats"])):
                    true_s, est_s = [], {mm: [] for mm in METHODS}
                    for k in range(n_clean + n_ramp):
                        idx = srng.choice(n_es, W, replace=False)
                        share = 0.0 if k < n_clean else (k - n_clean + 1) / n_ramp
                        hit = np.zeros(W, bool)
                        hit[srng.choice(W, int(round(share * W)), replace=False)] = True
                        mixed = [recs[(ES, cell)][i] if h else recs[(ES, "clean")][i] for i, h in zip(idx, hit)]
                        clean = [recs[(ES, "clean")][i] for i in idx]
                        ar = np.arange(W)
                        true_s.append(recall(clean, ar, cls) - recall(mixed, ar, cls))
                        x = np.array([pool(mixed, ar, ref)[s] for s in SIGNALS])
                        est_s["signals"].append(float(m.predict(x[cols][None])[0]))
                        for kk, v in base_fns[t](mixed, ar).items():
                            est_s[kk].append(v)
                    curves["true"] += np.array(true_s) / int(sc["repeats"])
                    onset = next((k for k, v in enumerate(true_s) if k >= n_clean and v >= thr), None)
                    n_drop += onset is not None
                    for mm in METHODS:
                        curves[mm] += np.array(est_s[mm]) / int(sc["repeats"])
                        al = next((k for k, v in enumerate(est_s[mm]) if v >= thr), None)
                        if al is not None and al < n_clean:
                            out[mm]["false_alarm"] += 1
                            al = next((k for k, v in enumerate(est_s[mm]) if k >= n_clean and v >= thr), None)
                        if onset is None:
                            continue
                        if al is None:
                            out[mm]["missed"] += 1
                        else:
                            out[mm]["delay"].append(al - onset)
                stream_res[(t, cell)] = (out, n_drop, curves)
                reps = int(sc["repeats"])
                snote = (f"{reps} streams of {n_clean} clean then {n_ramp} windows with 10%..100% of images under "
                         f"{cell}; {W} test images per window; drop and alarm threshold {thr}")
                add(ds, ES, f"stream_{cell}", t, "streams_with_real_drop", n_drop, reps,
                    f"{snote}; streams whose true drop reached the threshold after the clean prefix")
                for mm in METHODS:
                    o = out[mm]
                    what = proto if mm == "signals" else f"{mm.upper()} baseline"
                    if o["delay"]:
                        d = np.array(o["delay"], float)
                        add(ds, ES, f"stream_{cell}", t, f"alarm_delay_windows_mean_{mm}", d.mean(), len(d),
                            f"{what}; {snote}; alarm window minus true-drop window, positive = late; "
                            f"one window = {W} frames")
                        add(ds, ES, f"stream_{cell}", t, f"alarm_delay_windows_median_{mm}", float(np.median(d)),
                            len(d), f"{what}; {snote}")
                    add(ds, ES, f"stream_{cell}", t, f"missed_drops_{mm}", o["missed"], n_drop,
                        f"{what}; {snote}; real drops never alarmed within the stream")
                    add(ds, ES, f"stream_{cell}", t, f"false_alarm_streams_{mm}", o["false_alarm"], reps,
                        f"{what}; {snote}; streams that alarmed during the clean prefix")

    # ---- held-out sites (only when converted data is on disk)
    for (kind, site), r_s in recs.items():
        if kind != "site":
            continue
        spec = cfg["holdout_sites"][site]
        cls = list(range(nc)) if spec["classes"] == "all" else [by_name[c] for c in spec["classes"]]
        tname = f"site_{site}_classes"
        tr = [e for e in entries if e["split"] == FS and e["w"] != "full"]
        y = [recall(recs[(FS, "clean")], win[FS][e["w"]], cls) - recall(recs[(FS, e["cell"])], win[FS][e["w"]], cls)
             for e in tr]
        keep = [i for i, v in enumerate(y) if np.isfinite(v)]
        mdl = ridge(alpha).fit(np.stack([tr[i]["x"] for i in keep]), [y[i] for i in keep])
        bl = Baselines(recs[(FS, "clean")], cls, spec["classes"] == "all")
        r_ref = recall(recs[(ES, "clean")], range(n_es), cls)
        sw = window_sets(len(r_s), cfg, f"site_{site}") + [np.arange(len(r_s))]
        errs = {mm: [] for mm in METHODS}
        for i, idx in enumerate(sw):
            true = r_ref - recall(r_s, idx, cls)
            est = {"signals": float(mdl.predict(np.array([pool(r_s, idx, ref)[k] for k in SIGNALS])[None])[0]),
                   **bl(r_s, idx)}
            if i == len(sw) - 1:
                n_gt = int(sum(r["gt"][cls].sum() for r in r_s))
                note = (f"held-out site {site}, {len(r_s)} images; classes {spec['classes']}; true drop = "
                        f"{ES} clean recall minus site recall (different images, unpaired)")
                add(site, "test", "site", tname, "true_recall_drop", true, n_gt, note)
                for mm in METHODS:
                    add(site, "test", "site", tname, f"est_recall_drop_{mm}", est[mm], n_gt,
                        f"{note}; estimator fitted on all {FS} conditions" if mm == "signals" else note)
            else:
                for mm in METHODS:
                    errs[mm].append(abs(est[mm] - true))
        for mm in METHODS:
            add(site, "test", "site", tname, f"mae_window_{mm}", float(np.mean(errs[mm])), len(errs[mm]),
                f"held-out site {site}; {len(errs[mm])} windows of {min(W, len(r_s))} images")

    figures(run_id, targets, percell, stream_res, n_clean, n_ramp, thr)
    print_summary(summary, targets)
    return rows


# ============================================================================= figures and tables

def figures(run_id, targets, percell, stream_res, n_clean, n_ramp, thr) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = FIGURES_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(CONDITIONS, plt.get_cmap("tab10").colors))
    marks = {1: "o", 2: "s", 3: "^"}
    titles = {"signals": "Six signals (ridge, condition held out)", "ac": "AC baseline", "atc": "ATC baseline"}
    for t in targets:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
        lo = min(min(v["true"][1], *(v[m][1] for m in METHODS)) for (tt, _), v in percell.items() if tt == t)
        hi = max(max(v["true"][2], *(v[m][2] for m in METHODS)) for (tt, _), v in percell.items() if tt == t)
        for ax, m in zip(axes, METHODS):
            ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--")
            for (tt, cell), v in percell.items():
                if tt != t:
                    continue
                c = "black" if cell == "clean" else colors[cond_of(cell)]
                mk = "*" if cell == "clean" else marks[int(cell[-1])]
                x, y = v["true"], v[m]
                ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]],
                            fmt=mk, color=c, ms=7, capsize=2, lw=0.8)
            ax.set_title(titles[m])
            ax.set_xlabel("true recall drop (test, paired)")
        axes[0].set_ylabel("estimated recall drop (no labels)")
        handles = [plt.Line2D([], [], color=colors[c], marker="o", ls="", label=c) for c in CONDITIONS]
        handles += [plt.Line2D([], [], color="grey", marker=marks[s], ls="", label=f"severity {s}") for s in marks]
        fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=8)
        fig.suptitle(f"Estimated vs true recall drop, target '{t}', 18 shifted test cells + clean; bars = 95% CI over images")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out / f"est_vs_true_{t}.png", dpi=130)
        plt.close(fig)

    for t in targets:
        fig, axes = plt.subplots(2, 6, figsize=(20, 6.5), sharey=True)
        for ax, (key, (o, nd, curves)) in zip(axes.flat, [(k, v) for k, v in stream_res.items() if k[0] == t]):
            xs = np.arange(n_clean + n_ramp)
            ax.plot(xs, curves["true"], color="black", lw=2, label="true")
            for m, c in zip(METHODS, ["tab:blue", "tab:orange", "tab:green"]):
                ax.plot(xs, curves[m], color=c, label=m)
            ax.axhline(thr, color="red", ls=":", lw=1)
            ax.axvline(n_clean - 0.5, color="grey", ls="--", lw=1)
            ax.set_title(key[1], fontsize=9)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Streams (mean of repeats), target '{t}': clean windows, then a rising share of shifted images")
        fig.tight_layout()
        fig.savefig(out / f"streams_{t}.png", dpi=110)
        plt.close(fig)
    print(f"  figures in {out.relative_to(REPO_ROOT)}")


def print_summary(summary, targets) -> None:
    for t in targets:
        print(f"\nTarget {t}: MAE of recall-drop estimate on 18 shifted test cells")
        for (tt, m), (mae, lo, hi, bias, fmae, rho) in summary.items():
            if tt == t:
                print(f"  {m:40s} window MAE {mae:.4f} [{lo:.4f}, {hi:.4f}]  bias {bias:+.4f}  "
                      f"cell MAE {fmae:.4f}  spearman {rho:.3f}")


def table(run_id: str) -> int:
    with RESULTS_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == run_id]
    if not rows:
        print(f"No rows for {run_id}")
        return 1
    v = {(r["dataset"], r["condition"], r["class"], r["metric"]): r["value"] for r in rows}
    targets = sorted({r["class"] for r in rows if r["condition"] == "shifted_18_cells"})
    meths = ["signals", "signals_imgholdout", "ac", "atc"] + [f"only_{g}" for g in SIGNAL_GROUPS]
    ds = "construction-ppe"
    for t in targets:
        print(f"\n### Estimation error, target `{t}` (18 shifted test cells)\n")
        print("| method | window MAE [95% CI] | bias | full-cell MAE | Spearman (cells) |")
        print("|---|---|---|---|---|")
        for m in meths:
            g = lambda k: v.get((ds, "shifted_18_cells", t, k), "")
            print(f"| {m} | {g(f'mae_window_{m}')} [{g(f'mae_window_{m}_ci95_low')}, {g(f'mae_window_{m}_ci95_high')}] "
                  f"| {g(f'bias_window_{m}')} | {g(f'mae_cell_{m}')} | {g(f'spearman_cell_{m}')} |")
        print(f"\n### Per cell, target `{t}`: true vs estimated recall drop (full test cell, 95% CI)\n")
        print("| cell | true | signals | AC | ATC |")
        print("|---|---|---|---|---|")
        for cell in cells():
            g = lambda k: v.get((ds, cell, t, k), "")
            f = lambda k: f"{g(k)} [{g(k + '_ci95_low')}, {g(k + '_ci95_high')}]"
            print(f"| {cell} | {f('true_recall_drop')} | {f('est_recall_drop_signals')} | "
                  f"{f('est_recall_drop_ac')} | {f('est_recall_drop_atc')} |")
        print(f"\n### Lead time, target `{t}` (alarm delay in windows; positive = late)\n")
        print("| stream | real drops | signals: mean / median delay, missed, false-alarm streams | "
              "AC | ATC |")
        print("|---|---|---|---|---|")
        for c in CONDITIONS:
            for s in (2, 3):
                cond = f"stream_{c}_s{s}"
                if (ds, cond, t, "streams_with_real_drop") not in v:
                    continue
                g = lambda k: v.get((ds, cond, t, k), "-")
                cellf = lambda m: (f"{g(f'alarm_delay_windows_mean_{m}')} / {g(f'alarm_delay_windows_median_{m}')}, "
                                   f"{g(f'missed_drops_{m}')}, {g(f'false_alarm_streams_{m}')}")
                print(f"| {c}_s{s} | {g('streams_with_real_drop')} | {cellf('signals')} | {cellf('ac')} | {cellf('atc')} |")
    sites = sorted({r["dataset"] for r in rows if r["condition"] == "site"})
    for s in sites:
        print(f"\n### Held-out site {s}\n")
        for r in rows:
            if r["dataset"] == s:
                print(f"- {r['class']} {r['metric']}: {r['value']} (n={r['n']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.estimator", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="estimator YAML, e.g. configs/estimator/p4_estimator_yolo11s.yaml")
    ap.add_argument("--fresh", action="store_true", help="recompute the per-image signals instead of using the cache")
    ap.add_argument("--extract-only", action="store_true", help="compute and cache the per-image signals, then stop")
    ap.add_argument("--table", metavar="RUN_ID", help="print the Phase 4 tables of a run from results.csv")
    args = ap.parse_args(argv)
    if args.table:
        return table(args.table)
    if not args.config:
        ap.error("--config is required")
    cfg = load_yaml(args.config)
    if args.extract_only:
        extract(cfg, args.fresh)
        return 0
    command = "python -m ppe.estimator " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    rows = run(cfg, command, args.fresh)
    assert all(set(r) == set(FIELDS) for r in rows)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {rows[0]['run_id']}")
    print(f"Tables: python -m ppe.estimator --table {rows[0]['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
