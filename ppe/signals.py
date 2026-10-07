"""The six label-free reliability signals (Phase 4), computed per image and pooled per window.

For every image the detector sees, this module records what a deployed
monitor could compute without labels:

  1. Confidence drift        candidate box confidences (conf >= cand_conf), compared per window
                             with the training-site reference distribution (KS statistic and
                             shift of the mean).
  2. Augmentation disagreement  boxes at the operating confidence that appear or vanish under a
                             horizontal flip, a 0.75x rescale and a 1.3x brightness change.
  3. Temporal inconsistency  boxes at the operating confidence that flicker across a short
                             pseudo-sequence of the same still (small camera shake plus fresh
                             sensor noise per frame). This is a STILL-IMAGE PROXY: the datasets
                             have no video. The real signal needs tracked video (results/missing.md).
  4. Person-detector disagreement  persons found by a COCO-pretrained YOLO11n that no PPE-model
                             person box (operating confidence) overlaps.
  5. Orphan PPE              PPE and no_* boxes whose centre lies in no person box
                             (the same association rule as ppe.associate).
  6. Feature distance        Mahalanobis distance of the pooled C2PSA backbone feature from the
                             training-site reference (Ledoit-Wolf covariance).

It also records, for the same images, the labelled quantities the estimator
is fitted and scored against (true positives and ground truth per class at the
operating confidence, with the same matching as ppe.eval). Labels are never
read by any signal.

This module has no command of its own; ppe.estimator calls it:
    python -m ppe.estimator --config configs/estimator/p4_estimator_yolo11s.yaml
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np

from ppe.associate import PERSON_CLASS, PPE_CLASSES, Box, assign
from ppe.data import REPO_ROOT, sha256_file
from ppe.eval import greedy_match, iou_matrix, label_for, load_labels, match_per_class, prf_at

SIGNALS = ["conf_ks", "conf_mean_shift", "aug_disagree", "temporal_flicker", "person_disagree", "orphan_ppe",
           "feature_distance"]
SIGNAL_GROUPS = {   # the brief's six signals; confidence drift is two numbers
    "confidence_drift": ["conf_ks", "conf_mean_shift"],
    "augmentation_disagreement": ["aug_disagree"],
    "temporal_inconsistency": ["temporal_flicker"],
    "person_detector_disagreement": ["person_disagree"],
    "orphan_ppe": ["orphan_ppe"],
    "feature_distance": ["feature_distance"],
}


# ----------------------------------------------------------------------------- COCO person detector

def coco_weights(spec: dict) -> Path:
    """Download the COCO-pretrained detector once and check its pinned sha256."""
    path = REPO_ROOT / spec["path"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {spec['url']}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.rename(path)
    got = sha256_file(path)
    if got != spec["sha256"]:
        sys.exit(f"{path} sha256 {got} != pinned {spec['sha256']}; delete it and re-run, or update the pin knowingly.")
    return path


# ----------------------------------------------------------------------------- views

def _seed_rng(seed: int, tag: str, cell: str, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, zlib.crc32(tag.encode()), zlib.crc32(cell.encode()),
                                  zlib.crc32(name.encode("utf-8"))])


def aug_views(img: np.ndarray, p: dict) -> list[tuple[np.ndarray, callable]]:
    """(view image, function mapping view boxes back to original coordinates)."""
    import cv2

    h, w = img.shape[:2]
    out = []
    flip = img[:, ::-1].copy()
    out.append((flip, lambda b: np.stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]], 1) if len(b) else b))
    s = float(p["scale"])
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    sx, sy = w / small.shape[1], h / small.shape[0]
    out.append((small, lambda b: b * np.array([sx, sy, sx, sy]) if len(b) else b))
    bright = np.clip(img.astype(np.float32) * float(p["brightness"]), 0, 255).astype(np.uint8)
    out.append((bright, lambda b: b))
    return out


def temporal_views(img: np.ndarray, p: dict, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pseudo-frames: (frame, (dx, dy) shift applied). Frame 0 is the original image itself."""
    import cv2

    h, w = img.shape[:2]
    frames = []
    for _ in range(int(p["frames"])):
        dx, dy = rng.uniform(-1, 1, 2) * float(p["shift_frac"]) * np.array([w, h])
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        f = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        f += rng.normal(0, float(p["noise_sigma"]) * 255, f.shape)
        frames.append((np.clip(f, 0, 255).astype(np.uint8), np.array([dx, dy, dx, dy])))
    return frames


def n_matched(a_cls, a_xyxy, a_conf, b_cls, b_xyxy, iou_thr: float) -> int:
    """Same-class greedy matches between two box sets (a in confidence order)."""
    n = 0
    for c in np.unique(a_cls):
        am, bm = a_cls == c, b_cls == c
        if not bm.any():
            continue
        order = np.argsort(-a_conf[am], kind="stable")
        n += len(greedy_match(iou_matrix(a_xyxy[am][order], b_xyxy[bm]), iou_thr))
    return n


# ----------------------------------------------------------------------------- per-image records

class Extractor:
    """Holds the PPE model (with a feature hook) and the COCO person detector."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.cfg = cfg
        self.model = YOLO(str(REPO_ROOT / cfg["weights"]))
        self.names = dict(self.model.names)
        self.nc = len(self.names)
        self.person_id = next(i for i, n in self.names.items() if n == PERSON_CLASS)
        self.ppe_ids = {i for i, n in self.names.items() if n in PPE_CLASSES}
        self.coco = YOLO(str(coco_weights(cfg["person_detector"])))
        self._feats: list[np.ndarray] = []
        self._hook_on = False
        layer = self.model.model.model[int(cfg["feature_layer"])]
        layer.register_forward_hook(self._hook)
        # Boxes below cand_conf are never used, and dropping them before NMS cannot change any box above it
        # (a lower-scoring box never suppresses a higher one), so TP counts at the operating confidence equal
        # ppe.eval's at conf 0.001 while NMS runs several times faster on CPU.
        self.kw = dict(conf=float(cfg["cand_conf"]), iou=float(cfg["nms_iou"]), imgsz=int(cfg["imgsz"]),
                       device=cfg["device"], max_det=300, verbose=False)

    def _hook(self, _m, _i, out):
        if self._hook_on:
            t = out[0] if isinstance(out, (list, tuple)) else out
            self._feats.append(t.detach().float().mean(dim=(2, 3)).cpu().numpy())

    def _boxes(self, r) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        b = r.boxes
        conf = b.conf.cpu().numpy()
        order = np.argsort(-conf, kind="stable")
        return b.cls.cpu().numpy().astype(int)[order], conf[order], b.xyxy.cpu().numpy()[order]

    def _predict(self, sources: list, model=None, **extra) -> list:
        m = model or self.model
        kw = dict(self.kw, **extra)
        out = []
        bs = int(self.cfg["batch"])
        for i in range(0, len(sources), bs):
            out.extend(m.predict(sources[i:i + bs], **kw))
        return out

    def records(self, images: list[Path], cell: str, labelled: bool = True, light: bool = False) -> list[dict]:
        """One record per image. light=True: only candidate confidences and features (reference set)."""
        import cv2

        cfg = self.cfg
        t = float(cfg["operating_conf"])
        cand = float(cfg["cand_conf"])
        iou_thr = float(cfg["match_iou"])
        seed = int(cfg["seed"])

        # Original view: file paths in batches, as ppe.eval.predict_split, with the feature hook on.
        self._feats = []
        self._hook_on = True
        res = self._predict([str(p) for p in images])
        self._hook_on = False
        feats = np.concatenate(self._feats, 0)
        if len(feats) != len(images):
            sys.exit(f"feature hook captured {len(feats)} rows for {len(images)} images")

        coco = [] if light else self._predict([str(p) for p in images], model=self.coco,
                                               conf=float(cfg["person_detector"]["conf"]), classes=[0])
        out = []
        for k, (img_path, r) in enumerate(zip(images, res)):
            cls, conf, xyxy = self._boxes(r)
            rec = {"image": img_path.name, "cand_conf": conf[conf >= cand], "cand_cls": cls[conf >= cand],
                   "feat": feats[k]}
            if light:
                out.append(rec)
                continue
            h, w = r.orig_shape
            if labelled:
                g_cls, g_xyxy = load_labels(label_for(img_path), w, h)
                per = match_per_class([{"pred_cls": cls, "pred_conf": conf, "pred_xyxy": xyxy,
                                        "gt_cls": g_cls, "gt_xyxy": g_xyxy}], self.nc, iou_thr)
                rec["tp"] = np.array([prf_at(per[c], t)[0] for c in range(self.nc)], dtype=float)
                rec["gt"] = np.array([per[c]["n_gt"] for c in range(self.nc)], dtype=float)
            keep = conf >= t
            o_cls, o_conf, o_xyxy = cls[keep], conf[keep], xyxy[keep]

            # 4. person-detector disagreement
            ccls, cconf, cxyxy = self._boxes(coco[k])
            persons = o_xyxy[o_cls == self.person_id]
            if len(cxyxy):
                best = iou_matrix(cxyxy, persons).max(1) if len(persons) else np.zeros(len(cxyxy))
                rec["coco_n"], rec["coco_unmatched"] = len(cxyxy), int((best < cfg["person_detector"]["iou"]).sum())
            else:
                rec["coco_n"], rec["coco_unmatched"] = 0, 0

            # 5. orphan PPE (same rule as ppe.associate.assign)
            pboxes = [Box(self.names[int(c)], float(s), tuple(map(float, b)), int(c))
                      for c, s, b in zip(o_cls, o_conf, o_xyxy)]
            ppl = [b for b in pboxes if b.name == PERSON_CLASS]
            ppe = [b for b in pboxes if b.name in PPE_CLASSES]
            _, orphans = assign(ppl, ppe)
            rec["ppe_n"], rec["ppe_orphan"] = len(ppe), len(orphans)

            # 2. augmentation disagreement and 3. temporal flicker (same views batch)
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            av = aug_views(img, cfg["augment"])
            tv = temporal_views(img, cfg["temporal"], _seed_rng(seed, "temporal", cell, img_path.name))
            vres = self.model.predict([v for v, _ in av] + [f for f, _ in tv], **dict(self.kw, conf=t))
            un = tot = 0
            for (v, back), vr in zip(av, vres[:len(av)]):
                v_cls, v_conf, v_xyxy = self._boxes(vr)
                m = n_matched(o_cls, o_xyxy, o_conf, v_cls, back(v_xyxy), iou_thr)
                un += (len(o_cls) - m) + (len(v_cls) - m)
                tot += len(o_cls) + len(v_cls)
            rec["aug_unmatched"], rec["aug_total"] = un, tot
            seq = [(o_cls, o_conf, o_xyxy)]
            for (f, shift), fr in zip(tv, vres[len(av):]):
                f_cls, f_conf, f_xyxy = self._boxes(fr)
                seq.append((f_cls, f_conf, f_xyxy - shift if len(f_xyxy) else f_xyxy))
            un = tot = 0
            for (a_cls, a_conf, a_xyxy), (b_cls, b_conf, b_xyxy) in zip(seq[:-1], seq[1:]):
                m = n_matched(a_cls, a_xyxy, a_conf, b_cls, b_xyxy, iou_thr)
                un += (len(a_cls) - m) + (len(b_cls) - m)
                tot += len(a_cls) + len(b_cls)
            rec["temp_unmatched"], rec["temp_total"] = un, tot
            out.append(rec)
        return out


# ----------------------------------------------------------------------------- reference and pooling

class Reference:
    """Training-site reference: candidate confidences and backbone features of seeded train images."""

    def __init__(self, recs: list[dict]):
        from sklearn.covariance import LedoitWolf

        self.conf = np.sort(np.concatenate([r["cand_conf"] for r in recs]))
        self.conf_mean = float(self.conf.mean())
        f = np.stack([r["feat"] for r in recs])
        self.mu = f.mean(0)
        self.prec = LedoitWolf().fit(f).precision_
        self.n = len(recs)

    def feat_dist(self, feat: np.ndarray) -> float:
        d = feat - self.mu
        return float(np.sqrt(max(d @ self.prec @ d, 0.0)))


def ks_stat(sorted_ref: np.ndarray, x: np.ndarray) -> float:
    if len(x) == 0:
        return 1.0
    x = np.sort(x)
    grid = np.concatenate([sorted_ref, x])
    a = np.searchsorted(sorted_ref, grid, side="right") / len(sorted_ref)
    b = np.searchsorted(x, grid, side="right") / len(x)
    return float(np.abs(a - b).max())


def _ratio(a, b):
    return a / b if b else 0.0


def pool(recs: list[dict], idx: np.ndarray, ref: Reference) -> dict[str, float]:
    """Window-level signals from per-image records (ratios of sums, so big images do not dominate twice)."""
    sel = [recs[i] for i in idx]
    conf = np.concatenate([r["cand_conf"] for r in sel]) if sel else np.zeros(0)
    s = lambda k: float(sum(r[k] for r in sel))
    return {
        "conf_ks": ks_stat(ref.conf, conf),
        "conf_mean_shift": ref.conf_mean - (float(conf.mean()) if len(conf) else 0.0),
        "aug_disagree": _ratio(s("aug_unmatched"), s("aug_total")),
        "temporal_flicker": _ratio(s("temp_unmatched"), s("temp_total")),
        "person_disagree": _ratio(s("coco_unmatched"), s("coco_n")),
        "orphan_ppe": _ratio(s("ppe_orphan"), s("ppe_n")),
        "feature_distance": float(np.mean([r["feat_dist"] for r in sel])),
    }


def file_digest(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.name.encode())
        h.update(sha256_file(p).encode())
    return h.hexdigest()[:12]
