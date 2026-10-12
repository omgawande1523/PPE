"""Stage 1 of the loop: the label-free monitor, as the orchestrator runs it on a stream.

Phase 4 found that confidence drift alone estimates the recall drop as well as
all six signals (results/phase4_estimator.md), and it is the only signal that
needs nothing beyond the detector's own boxes. So the loop's monitor is the
Phase 4 confidence-drift estimator:

  * Reference: candidate confidences (conf >= cand_conf) of a seeded sample of
    training images, under the model currently deployed.
  * Features per window: KS statistic and mean shift of the window's candidate
    confidences against that reference (ppe.signals.ks_stat, as in Phase 4).
  * Estimator: ridge regression (Phase 4 settings) from the two features to the
    micro recall drop over all classes, fitted on VALIDATION windows (clean plus
    the Phase 2 conditions built from val images). Conditions named in
    `fit_exclude_conditions` are left out of the fit, so a scenario's shift is
    unseen by the monitor (the Phase 4 leave-one-condition-out protocol).
  * Alarm threshold: the larger of `min_threshold` and the `threshold_quantile`
    of the estimate over seeded clean-val windows. Fixed by the config; never
    looked at test images.

The test split (golden set v1) is never read here.

No command of its own; ppe.loop builds it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zlib
from pathlib import Path

import numpy as np

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_path
from ppe.eval import label_for, load_labels, match_per_class, prf_at
from ppe.signals import ks_stat

CACHE = REPO_ROOT / "runs" / "loop" / "monitor_cache"
FEATURES = ["conf_ks", "conf_mean_shift"]


def list_dir(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []


def predict_frames(model, frames: list, cfg: dict, conf: float | None = None) -> list[dict]:
    """Detections per frame (file path or BGR array), sorted by confidence."""
    out = []
    bs = int(cfg["batch"])
    kw = dict(conf=float(cfg["cand_conf"] if conf is None else conf), iou=float(cfg["nms_iou"]),
              imgsz=int(cfg["imgsz"]), device=cfg["device"], max_det=300, verbose=False)
    for i in range(0, len(frames), bs):
        chunk = [str(f) if isinstance(f, Path) else f for f in frames[i:i + bs]]
        for r in model.predict(chunk, **kw):
            b = r.boxes
            c = b.conf.cpu().numpy()
            o = np.argsort(-c, kind="stable")
            out.append({"cls": b.cls.cpu().numpy().astype(int)[o], "conf": c[o], "xyxy": b.xyxy.cpu().numpy()[o],
                        "hw": tuple(r.orig_shape)})
    return out


def tp_gt(det: dict, gt_cls: np.ndarray, gt_xyxy: np.ndarray, nc: int, t: float, iou_thr: float):
    """True positives and ground truth per class at confidence t (ppe.eval matching)."""
    per = match_per_class([{"pred_cls": det["cls"], "pred_conf": det["conf"], "pred_xyxy": det["xyxy"],
                            "gt_cls": gt_cls, "gt_xyxy": gt_xyxy}], nc, iou_thr)
    return (np.array([prf_at(per[c], t)[0] for c in range(nc)], float),
            np.array([per[c]["n_gt"] for c in range(nc)], float))


def recall_of(tp: np.ndarray, gt: np.ndarray, cls: list[int] | None = None) -> float:
    """Micro recall from per-frame (n, nc) TP and GT arrays."""
    cls = list(range(tp.shape[1])) if cls is None else cls
    g = gt[:, cls].sum()
    return float(tp[:, cls].sum() / g) if g else float("nan")


class Monitor:
    """Confidence-drift estimator of the recall drop, calibrated for one set of weights."""

    def __init__(self, model, weights: Path, cfg: dict, data: dict, shift_cfg: dict):
        self.model, self.cfg = model, cfg
        self.mcfg = cfg["monitor"]
        self.nc = len(model.names)
        key_blob = {"weights": sha256_path(weights), "monitor": self.mcfg, "corr": shift_cfg["corruptions"],
                    **{k: cfg[k] for k in ("imgsz", "nms_iou", "cand_conf", "operating_conf", "match_iou", "seed", "window_size")}}
        self.key = hashlib.sha256(json.dumps(key_blob, sort_keys=True).encode()).hexdigest()[:12]
        cache = CACHE / f"{self.key}.pkl"
        if cache.exists():
            st = pickle.loads(cache.read_bytes())
        else:
            st = self._calibrate(data, shift_cfg)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps(st))
        self.ref, self.model_fit, self.threshold = st["ref"], st["ridge"], st["threshold"]
        self.calib = st["calib"]

    # ------------------------------------------------------------------ calibration (train + val only)
    def _calibrate(self, data: dict, shift_cfg: dict) -> dict:
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        from ppe.shift import CONDITIONS, SEVERITIES, make_cell

        m, cfg = self.mcfg, self.cfg
        seed = int(cfg["seed"])
        root = REPO_ROOT / data["path"]
        train = list_dir(root / data["train"])
        pick = sorted(np.random.default_rng(seed).choice(len(train), int(m["reference_images"]), replace=False))
        ref = np.sort(np.concatenate([d["conf"] for d in predict_frames(self.model, [train[i] for i in pick], cfg)]))
        print(f"  monitor: reference = {len(pick)} train images, {len(ref)} candidate boxes", flush=True)

        val = list_dir(root / data["val"])
        t, iou = float(cfg["operating_conf"]), float(cfg["match_iou"])

        def recs(images):
            dets = predict_frames(self.model, images, cfg)
            out = []
            for img, d in zip(images, dets):
                h, w = d["hw"]
                tp, gt = tp_gt(d, *load_labels(label_for(img), w, h), self.nc, t, iou)
                out.append({"conf": d["conf"], "tp": tp, "gt": gt})
            return out

        cells = {"clean": recs(val)}
        excl = set(m.get("fit_exclude_conditions") or [])
        for cond in CONDITIONS:
            if cond in excl:
                continue
            for s in range(SEVERITIES):
                out_root = REPO_ROOT / "runs" / "loop" / "cells" / f"val_{cond}_s{s + 1}"
                make_cell(val, cond, s, shift_cfg["corruptions"][cond], int(shift_cfg["seed"]), out_root)
                cells[f"{cond}_s{s + 1}"] = recs(list_dir(out_root / "images" / "test"))
        print(f"  monitor: fit cells = {list(cells)} (excluded: {sorted(excl) or 'none'})", flush=True)

        rng = np.random.default_rng([seed, zlib.crc32(b"monitor_fit")])
        W = int(cfg["window_size"])
        wins = [np.sort(rng.choice(len(val), W, replace=False)) for _ in range(int(m["fit_windows_per_cell"]))]
        X, y = [], []
        clean = cells["clean"]
        for name, rs in cells.items():
            for idx in wins:
                X.append(self._feats_from([rs[i]["conf"] for i in idx], ref))
                tp_c = np.stack([clean[i]["tp"] for i in idx]); gt_c = np.stack([clean[i]["gt"] for i in idx])
                tp_x = np.stack([rs[i]["tp"] for i in idx]); gt_x = np.stack([rs[i]["gt"] for i in idx])
                y.append(recall_of(tp_c, gt_c) - recall_of(tp_x, gt_x))
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=float(m["ridge_alpha"]))).fit(np.array(X), np.array(y))

        # Alarm threshold from clean val windows only.
        rng = np.random.default_rng([seed, zlib.crc32(b"monitor_threshold")])
        est = [float(ridge.predict(np.array([self._feats_from(
            [clean[i]["conf"] for i in np.sort(rng.choice(len(val), W, replace=False))], ref)]))[0])
            for _ in range(int(m["threshold_windows"]))]
        q = float(np.quantile(est, float(m["threshold_quantile"])))
        thr = max(float(m["min_threshold"]), q)
        calib = {"fit_cells": list(cells), "fit_windows": len(X), "clean_val_quantile": q,
                 "clean_val_windows": len(est), "threshold": thr, "reference_boxes": int(len(ref)),
                 "clean_val_est_mean": float(np.mean(est))}
        print(f"  monitor: alarm threshold {thr:.4f} (clean-val q{m['threshold_quantile']} = {q:.4f})", flush=True)
        return {"ref": ref, "ridge": ridge, "threshold": thr, "calib": calib}

    # ------------------------------------------------------------------ run time (no labels)
    @staticmethod
    def _feats_from(confs: list[np.ndarray], ref: np.ndarray) -> list[float]:
        c = np.concatenate(confs) if confs else np.zeros(0)
        return [ks_stat(ref, c), float(ref.mean()) - (float(c.mean()) if len(c) else 0.0)]

    def estimate(self, dets: list[dict]) -> tuple[float, dict]:
        """Estimated recall drop of a window from its detections (labels never read)."""
        f = self._feats_from([d["conf"] for d in dets], self.ref)
        return float(self.model_fit.predict(np.array([f]))[0]), dict(zip(FEATURES, f))
