"""Single inference pipeline with per-person PPE compliance.

Replaces the two disagreeing pipelines (main.py crop-and-rerun at conf 0.3,
app.py whole-frame at conf 0.5) with one whole-frame pass whose settings come
from one config file. Every frame is written to a JSON-lines run log.

Run (from the repository root):
    python -m ppe.infer                                   # webcam 0, configs/infer_default.yaml
    python -m ppe.infer --source path/to/images/          # folder of images
    python -m ppe.infer --source clip.mp4 --show          # video, with a preview window
    python -m ppe.infer --config configs/my_run.yaml      # another experiment config
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import yaml

from ppe.associate import (
    COMPLIANT,
    ITEM_CLASSES,
    PERSON_CLASS,
    VIOLATION,
    Box,
    PersonResult,
    evaluate_frame,
)
from ppe.logging import RunLogger

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = Path("configs") / "infer_default.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EXPECTED_CLASSES = {PERSON_CLASS, *ITEM_CLASSES, *(n for n in ITEM_CLASSES.values() if n)}


@dataclass
class FrameResult:
    frame_idx: int
    source: str
    image: np.ndarray
    boxes: list[Box]
    persons: list[PersonResult]
    orphans: list[Box]
    infer_ms: float

    @property
    def counts(self) -> Counter:
        return Counter(p.status for p in self.persons)


def load_config(path: str | Path) -> dict:
    path = _resolve(path)
    with path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    cfg["config_path"] = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
    return cfg


def _resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repository root."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class Pipeline:
    """Loads the detector once; turns frames or a source into FrameResults."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        self.cfg = cfg
        weights = _resolve(cfg["weights"])
        if not weights.exists():
            sys.exit(
                f"Weights not found: {weights}\n"
                f"Put best.pt in weights/ or set 'weights' in {cfg.get('config_path', 'the config')}."
            )
        set_seed(int(cfg.get("seed", 0)))
        self.model = YOLO(str(weights))
        self.names: dict[int, str] = dict(self.model.names)
        missing = EXPECTED_CLASSES - set(self.names.values())
        if missing:
            sys.exit(f"Model classes {sorted(self.names.values())} lack {sorted(missing)}; wrong weights?")
        self.required_items = list(cfg.get("required_items", list(ITEM_CLASSES)))
        self.predict_kwargs = dict(
            conf=float(cfg.get("conf", 0.25)),
            iou=float(cfg.get("iou", 0.7)),
            imgsz=int(cfg.get("imgsz", 640)),
            device=cfg.get("device") or None,
            verbose=False,
        )

    def _to_boxes(self, result) -> list[Box]:
        b = result.boxes
        if b is None or len(b) == 0:
            return []
        xyxy = b.xyxy.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        conf = b.conf.cpu().numpy()
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else [None] * len(cls)
        return [
            Box(
                name=self.names[int(c)],
                conf=float(s),
                xyxy=tuple(float(v) for v in box),
                cls_id=int(c),
                track_id=None if t is None else int(t),
            )
            for box, c, s, t in zip(xyxy, cls, conf, ids)
        ]

    def _wrap(self, result, frame_idx: int, infer_ms: float) -> FrameResult:
        boxes = self._to_boxes(result)
        persons, orphans = evaluate_frame(boxes, self.required_items)
        return FrameResult(frame_idx, str(result.path), result.orig_img, boxes, persons, orphans, infer_ms)

    def process(self, frame: np.ndarray, frame_idx: int = 0, track: bool = True) -> FrameResult:
        """Run one BGR frame (for the Flask app or any caller that owns the capture loop)."""
        t0 = time.perf_counter()
        if track:
            result = self.model.track(frame, persist=True, tracker=self.cfg.get("tracker", "bytetrack.yaml"),
                                      **self.predict_kwargs)[0]
        else:
            result = self.model.predict(frame, **self.predict_kwargs)[0]
        return self._wrap(result, frame_idx, (time.perf_counter() - t0) * 1000.0)

    def run(self, source: str, track: bool | None = None) -> Iterator[FrameResult]:
        """Stream results for a webcam index, video file, image file or image folder.

        Tracking is on for webcams and videos and off for still images, where
        track IDs would be meaningless across unrelated pictures.
        """
        src = _parse_source(source)
        if track is None:
            track = not _is_still(src)
        if track:
            stream = self.model.track(src, stream=True, persist=True,
                                      tracker=self.cfg.get("tracker", "bytetrack.yaml"), **self.predict_kwargs)
        else:
            stream = self.model.predict(src, stream=True, **self.predict_kwargs)
        for i, result in enumerate(stream):
            # Ultralytics' own per-stage timing, so capture/decoding time is excluded.
            infer_ms = sum(v for v in (result.speed or {}).values() if v is not None)
            yield self._wrap(result, i, infer_ms)


def _parse_source(source: str):
    if isinstance(source, str) and source.isdigit():
        return int(source)
    p = Path(source)
    if not p.is_absolute() and not p.exists() and (REPO_ROOT / p).exists():
        p = REPO_ROOT / p
    if not p.exists():
        sys.exit(f"Source not found: {source}")
    return str(p)


def _is_still(src) -> bool:
    if isinstance(src, int):
        return False
    p = Path(src)
    return p.is_dir() or p.suffix.lower() in IMAGE_SUFFIXES


STATUS_COLOURS = {COMPLIANT: (0, 200, 0), VIOLATION: (0, 0, 230)}  # BGR; unknown -> amber


def draw(fr: FrameResult) -> np.ndarray:
    import cv2

    img = fr.image.copy()
    for p in fr.persons:
        x1, y1, x2, y2 = map(int, p.person.xyxy)
        colour = STATUS_COLOURS.get(p.status, (0, 170, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
        tid = "" if p.person.track_id is None else f"#{p.person.track_id} "
        cv2.putText(img, f"{tid}{p.status}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        for b in p.boxes:
            bx1, by1, bx2, by2 = map(int, b.xyxy)
            c = (0, 0, 230) if b.name.startswith("no_") else (200, 200, 0)
            cv2.rectangle(img, (bx1, by1), (bx2, by2), c, 1)
            cv2.putText(img, f"{b.name} {b.conf:.2f}", (bx1, by2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
    return img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.infer", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="experiment YAML (default: %(default)s)")
    ap.add_argument("--source", help="webcam index, video, image or image folder (overrides config)")
    ap.add_argument("--weights", help="weights file (overrides config)")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    ap.add_argument("--show", action="store_true", help="show an annotated preview window")
    ap.add_argument("--save-dir", help="write annotated frames here")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.weights:
        cfg["weights"] = args.weights
    source = str(args.source if args.source is not None else cfg.get("source", "0"))

    pipe = Pipeline(cfg)
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "config": cfg.get("config_path"),
        "weights": Path(cfg["weights"]).name,
        "source": source,
        "conf": pipe.predict_kwargs["conf"],
        "iou": pipe.predict_kwargs["iou"],
        "imgsz": pipe.predict_kwargs["imgsz"],
        "required_items": pipe.required_items,
        "class_names": pipe.names,
        "git_commit": _git_commit(),
    }
    totals: Counter = Counter()
    n = 0
    with RunLogger(_resolve(cfg.get("log_dir", "runs/logs")), meta) as log:
        for fr in pipe.run(source):
            log.frame(fr.frame_idx, fr.source, fr.image.shape[:2], fr.boxes, fr.persons, fr.orphans, fr.infer_ms)
            totals.update(fr.counts)
            n += 1
            if args.show or save_dir:
                img = draw(fr)
                if save_dir:
                    import cv2

                    cv2.imwrite(str(save_dir / f"{fr.frame_idx:06d}.jpg"), img)
                if args.show:
                    import cv2

                    cv2.imshow("ppe.infer", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if args.max_frames and n >= args.max_frames:
                break
    print(f"{n} frames; person decisions: {dict(totals)}; log: {log.path}")
    return 0


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
