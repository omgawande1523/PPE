"""Stage 3 of the loop: mine frames, pseudo-label them with a teacher, fine-tune with replay.

  1. Mine. Each buffered frame gets a disagreement score: the share of its boxes
     (operating confidence) that appear or vanish under a horizontal flip, a 0.75x
     rescale and a 1.3x brightness change (ppe.signals.aug_views, Phase 4's
     augmentation-disagreement signal). A frame where no view finds anything
     scores 1. The `n_mine` highest-scoring frames are kept (seeded tie-break).
  2. Pseudo-label with a teacher that is not the deployed model:
       source_labels  the label files shipped with the source images. In the
                      synthetic site stream every frame is a corrupted copy of a
                      labelled training image, so this stands in for a PERFECT
                      teacher (or a fully human-labelled batch). It is an upper
                      bound on what a real teacher can give, not a real teacher.
       yolo_world     an open-vocabulary detector (Ultralytics YOLO-World) prompted
                      with one text per class. A label is kept only if the teacher
                      finds it again on the horizontally flipped frame (same class,
                      IoU >= agree_iou): the still-image stand-in for "teacher and
                      tracker agree", since the stream has no video.
     A `poison` block, when present, corrupts the teacher's labels on purpose
     (the brief's poisoned-batch scenario).
  3. Human check. A seeded random sample of pseudo-labelled frames is written,
     with the boxes drawn, to <run>/human_check/ together with a CSV to fill in.
     Nobody has checked it until that CSV comes back; until then the human
     pseudo-label error rate is missing and human minutes are 0.
     Because the synthetic frames have labels, the pseudo-label error against
     those labels is also measured automatically. That number is NOT a human check.
  4. Fine-tune from the current weights on the mined frames plus a seeded replay
     sample of original training images with their original labels. No held-out
     split is used during training (Ultralytics' val points at the training set),
     and the last epoch is the candidate: no checkpoint selection.

The test split (golden set v1) is never read here.
"""

from __future__ import annotations

import csv
import os
import shutil
import zlib
from pathlib import Path

import numpy as np
import yaml

from ppe.data import REPO_ROOT
from ppe.eval import greedy_match, iou_matrix, label_for, load_labels
from ppe.signals import aug_views, n_matched


# ----------------------------------------------------------------------------- 1. mining

def disagreement(model, frame: np.ndarray, det: dict, cfg: dict) -> float:
    """Share of operating-confidence boxes that appear or vanish under the three views."""
    t, iou = float(cfg["operating_conf"]), float(cfg["match_iou"])
    keep = det["conf"] >= t
    o_cls, o_conf, o_xyxy = det["cls"][keep], det["conf"][keep], det["xyxy"][keep]
    views = aug_views(frame, cfg["improve"]["augment"])
    res = model.predict([v for v, _ in views], conf=t, iou=float(cfg["nms_iou"]), imgsz=int(cfg["imgsz"]),
                        device=cfg["device"], max_det=300, verbose=False)
    un = tot = 0
    for (_, back), r in zip(views, res):
        b = r.boxes
        v_cls, v_xyxy = b.cls.cpu().numpy().astype(int), b.xyxy.cpu().numpy()
        m = n_matched(o_cls, o_xyxy, o_conf, v_cls, back(v_xyxy), iou)
        un += (len(o_cls) - m) + (len(v_cls) - m)
        tot += len(o_cls) + len(v_cls)
    return un / tot if tot else 1.0


def mine(scores: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Indices of the n highest scores; ties broken by a seeded random key."""
    tie = np.random.default_rng([seed, zlib.crc32(b"mine")]).random(len(scores))
    order = np.lexsort((tie, -scores))
    return np.sort(order[:n])


# ----------------------------------------------------------------------------- 2. teachers

def yolo_to_xyxy(rows: np.ndarray, w: int, h: int) -> np.ndarray:
    cx, cy, bw, bh = rows[:, 0] * w, rows[:, 1] * h, rows[:, 2] * w, rows[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def xyxy_to_yolo(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = (np.clip(xyxy[:, 0], 0, w), np.clip(xyxy[:, 1], 0, h), np.clip(xyxy[:, 2], 0, w),
                      np.clip(xyxy[:, 3], 0, h))
    return np.stack([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 1)


class SourceLabels:
    """Perfect-teacher stand-in: the dataset labels of each frame's source image."""

    kind = "source_labels"

    def __init__(self, tcfg: dict, names: dict[int, str], cfg: dict):
        pass

    def __call__(self, frames: list[np.ndarray], label_paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for f, lp in zip(frames, label_paths):
            h, w = f.shape[:2]
            out.append(load_labels(lp, w, h))
        return out


class YoloWorld:
    """Open-vocabulary teacher, kept only where it agrees with itself on the flipped frame."""

    kind = "yolo_world"

    def __init__(self, tcfg: dict, names: dict[int, str], cfg: dict):
        from ultralytics import YOLOWorld

        from ppe.signals import coco_weights

        self.cfg, self.t = cfg, tcfg
        path = coco_weights(tcfg["weights"])
        self.model = YOLOWorld(str(path))
        by_name = {v: k for k, v in names.items()}
        self.prompts = [p for p in tcfg["prompts"].values()]
        self.ids = np.array([by_name[c] for c in tcfg["prompts"]])
        self.model.set_classes(self.prompts)

    def _pred(self, imgs):
        res = self.model.predict(imgs, conf=float(self.t["conf"]), iou=float(self.cfg["nms_iou"]),
                                 imgsz=int(self.cfg["imgsz"]), device=self.cfg["device"], max_det=300, verbose=False)
        return [(r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy(), r.boxes.xyxy.cpu().numpy())
                for r in res]

    def __call__(self, frames: list[np.ndarray], label_paths: list[Path]) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        thr = float(self.t["agree_iou"])
        for i in range(0, len(frames), 8):
            chunk = frames[i:i + 8]
            a = self._pred(chunk)
            b = self._pred([f[:, ::-1].copy() for f in chunk])
            for f, (ac, aconf, ax), (bc, _, bx) in zip(chunk, a, b):
                w = f.shape[1]
                bx = np.stack([w - bx[:, 2], bx[:, 1], w - bx[:, 0], bx[:, 3]], 1) if len(bx) else bx
                keep = np.zeros(len(ac), bool)
                for c in np.unique(ac):
                    am, bm = np.where(ac == c)[0], np.where(bc == c)[0]
                    if not len(bm):
                        continue
                    order = am[np.argsort(-aconf[am], kind="stable")]
                    for r in greedy_match(iou_matrix(ax[order], bx[bm]), thr):
                        keep[order[r]] = True
                out.append((self.ids[ac[keep]] if keep.any() else np.zeros(0, int), ax[keep]))
        return out


TEACHERS = {"source_labels": SourceLabels, "yolo_world": YoloWorld}


def poison(labels: list[tuple[np.ndarray, np.ndarray]], pcfg: dict, names: dict[int, str], seed: int):
    """Swap violation labels to their compliant counterparts (no_helmet -> helmet, ...) with probability p.

    This is the harmful poison for a compliance monitor: it teaches the model that
    violators are compliant. Returns the poisoned labels and the number of boxes changed.
    """
    by_name = {v: k for k, v in names.items()}
    swap = {by_name[a]: by_name[b] for a, b in pcfg["swap"].items()}
    rng = np.random.default_rng([seed, zlib.crc32(b"poison")])
    out, changed = [], 0
    for cls, xyxy in labels:
        cls = cls.copy()
        for i, c in enumerate(cls):
            if int(c) in swap and rng.random() < float(pcfg["p"]):
                cls[i] = swap[int(c)]
                changed += 1
        out.append((cls, xyxy))
    return out, changed


def label_error(pseudo: list[tuple[np.ndarray, np.ndarray]], truth: list[tuple[np.ndarray, np.ndarray]],
                iou_thr: float, cls_subset: list[int] | None = None) -> dict:
    """Pseudo-label error against known labels: wrong boxes / pseudo boxes, missed boxes / true boxes."""
    n_p = n_t = matched = 0
    for (pc, px), (tc, tx) in zip(pseudo, truth):
        if cls_subset is not None:
            pk, tk = np.isin(pc, cls_subset), np.isin(tc, cls_subset)
            pc, px, tc, tx = pc[pk], px[pk], tc[tk], tx[tk]
        n_p += len(pc)
        n_t += len(tc)
        for c in np.unique(pc):
            pm, tm = pc == c, tc == c
            if tm.any():
                matched += len(greedy_match(iou_matrix(px[pm], tx[tm]), iou_thr))
    return {"n_pseudo": n_p, "n_true": n_t, "matched": matched,
            "error_rate": (n_p - matched) / n_p if n_p else float("nan"),
            "miss_rate": (n_t - matched) / n_t if n_t else float("nan")}


# ----------------------------------------------------------------------------- 3. human check sample

def human_sample(out_dir: Path, frames: list[np.ndarray], labels, names: dict[int, str], ids: list[str],
                 n: int, seed: int) -> list[int]:
    """Write a seeded random sample of pseudo-labelled frames, boxes drawn, and the CSV a checker fills in."""
    import cv2

    rng = np.random.default_rng([seed, zlib.crc32(b"human_sample")])
    pick = sorted(rng.choice(len(frames), min(n, len(frames)), replace=False).tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in pick:
        img = frames[i].copy()
        cls, xyxy = labels[i]
        for c, b in zip(cls, xyxy):
            x1, y1, x2, y2 = map(int, b)
            col = (0, 0, 230) if names[int(c)].startswith("no_") else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            cv2.putText(img, names[int(c)], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(str(out_dir / f"{ids[i]}.jpg"), img)
        rows.append({"frame": ids[i], "pseudo_boxes": len(cls), "wrong_boxes": "", "missing_boxes": "",
                     "minutes": "", "checker": ""})
    with (out_dir / "human_check.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else ["frame"])
        w.writeheader()
        w.writerows(rows)
    (out_dir / "README.txt").write_text(
        "Human check of pseudo-labels. For each image: count boxes that are wrong (wrong class or no such\n"
        "object) and objects that have no box, write them in human_check.csv with the minutes you spent and\n"
        "your initials. Green = PPE or person, red = violation class. Then record it with:\n"
        "  python -m ppe.loop --human-check <this run folder>\n", encoding="utf-8")
    return pick


# ----------------------------------------------------------------------------- 4. fine-tune with replay

def build_dataset(root: Path, mined: list[tuple[str, np.ndarray, tuple[np.ndarray, np.ndarray]]],
                  replay: list[Path], names: dict[int, str]) -> Path:
    """images/train + labels/train: mined frames with pseudo-labels, replay images with original labels."""
    import cv2

    if root.exists():
        shutil.rmtree(root)
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    for stem, img, (cls, xyxy) in mined:
        h, w = img.shape[:2]
        cv2.imwrite(str(root / "images" / "train" / f"{stem}.png"), img)
        yl = xyxy_to_yolo(xyxy, w, h) if len(xyxy) else np.zeros((0, 4))
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            "".join(f"{int(c)} {a:.6f} {b:.6f} {cc:.6f} {d:.6f}\n" for c, (a, b, cc, d) in zip(cls, yl)),
            encoding="utf-8")
    for p in replay:
        shutil.copy2(p, root / "images" / "train" / f"replay_{p.name}")
        shutil.copy2(label_for(p), root / "labels" / "train" / f"replay_{p.stem}.txt")
    data = root / "data.yaml"
    with data.open("w", encoding="utf-8") as fh:
        # val points at the training images: no held-out split is touched while training
        yaml.safe_dump({"path": str(root.resolve()), "train": "images/train", "val": "images/train",
                        "names": {int(k): v for k, v in names.items()}}, fh, sort_keys=False)
    return data


def finetune(weights: Path, data_yaml: Path, fcfg: dict, seed: int, device: str, project: Path, name: str) -> Path:
    from ultralytics import YOLO

    from ppe.infer import set_seed

    os.environ.setdefault("WANDB_MODE", "disabled")
    set_seed(seed)
    model = YOLO(str(weights))
    model.train(data=str(data_yaml), epochs=int(fcfg["epochs"]), imgsz=int(fcfg["imgsz"]), batch=int(fcfg["batch"]),
                seed=seed, deterministic=True, device=device, workers=int(fcfg["workers"]),
                optimizer=fcfg["optimizer"], lr0=float(fcfg["lr0"]), lrf=float(fcfg["lrf"]),
                warmup_epochs=float(fcfg["warmup_epochs"]), freeze=int(fcfg["freeze"]),
                mosaic=float(fcfg["mosaic"]), close_mosaic=int(fcfg["close_mosaic"]), amp=False, plots=False,
                val=False, project=str(project), name=name, exist_ok=True, verbose=False)
    last = project / name / "weights" / "last.pt"
    if not last.exists():
        raise FileNotFoundError(last)
    return last
