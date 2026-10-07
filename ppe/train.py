"""Train a detector from one experiment config (meant for a Colab T4).

The baseline recipe repeats the training notebook: COCO-pretrained weights,
50 epochs, 640 px, batch 16, Ultralytics defaults otherwise. Only the seed
and the model size change between the Phase 1 configs. Ultralytics picks the
saved best.pt on val, so val is a selection split; test is never touched here.

Run (from the repository root):
    python -m ppe.train --config configs/train/p1_yolo11s_seed0.yaml
Then evaluate the result:
    python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml \
        --weights runs/train/p1_yolo11s_seed0/weights/best.pt --run-name p1_yolo11s_seed0 --model yolo11s
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from ppe.data import REPO_ROOT, resolve_data_yaml
from ppe.infer import set_seed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.train", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg_path = Path(args.config)
    with (cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}.")
    os.environ.setdefault("WANDB_MODE", "disabled")
    seed = int(cfg["seed"])
    set_seed(seed)

    project = REPO_ROOT / "runs" / "train"
    data_yaml, _ = resolve_data_yaml(cfg["data"], project / cfg["run_name"] / "_data")
    model = YOLO(cfg["model"])
    model.train(
        data=str(data_yaml),
        epochs=int(cfg["epochs"]),
        imgsz=int(cfg["imgsz"]),
        batch=int(cfg["batch"]),
        seed=seed,
        deterministic=bool(cfg.get("deterministic", True)),
        device=cfg.get("device", 0),
        workers=int(cfg.get("workers", 2)),
        project=str(project),
        name=cfg["run_name"],
        exist_ok=True,
        plots=True,
    )
    best = project / cfg["run_name"] / "weights" / "best.pt"
    print(f"\nbest.pt (selected on val by Ultralytics): {best.relative_to(REPO_ROOT)}")
    print("Evaluate with:\n  python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml "
          f"--weights {best.relative_to(REPO_ROOT).as_posix()} --run-name {cfg['run_name']} "
          f"--model {Path(cfg['model']).stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
