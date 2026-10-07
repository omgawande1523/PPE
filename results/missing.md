# Results not yet measured

Each line is a result the brief asks for that is not in `results/results.csv`, with the command that produces it.
Nothing here has a value until that command has run and appended its rows.

## Phase 1: seed variance of the training recipe (needs a GPU; Colab T4)

Notebook: `notebooks/colab_phase1_train.ipynb` runs all six. One run by hand:

| Model | Seed | Train | Evaluate (appends val + test rows) |
|---|---:|---|---|
| YOLO11n | 0 | `python -m ppe.train --config configs/train/p1_yolo11n_seed0.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11n_seed0/weights/best.pt --run-name p1_yolo11n_seed0 --model yolo11n` |
| YOLO11n | 1 | `python -m ppe.train --config configs/train/p1_yolo11n_seed1.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11n_seed1/weights/best.pt --run-name p1_yolo11n_seed1 --model yolo11n` |
| YOLO11n | 2 | `python -m ppe.train --config configs/train/p1_yolo11n_seed2.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11n_seed2/weights/best.pt --run-name p1_yolo11n_seed2 --model yolo11n` |
| YOLO11s | 0 | `python -m ppe.train --config configs/train/p1_yolo11s_seed0.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11s_seed0/weights/best.pt --run-name p1_yolo11s_seed0 --model yolo11s` |
| YOLO11s | 1 | `python -m ppe.train --config configs/train/p1_yolo11s_seed1.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11s_seed1/weights/best.pt --run-name p1_yolo11s_seed1 --model yolo11s` |
| YOLO11s | 2 | `python -m ppe.train --config configs/train/p1_yolo11s_seed2.yaml` | `python -m ppe.eval --config configs/eval/baseline_yolo11s_test.yaml --weights runs/train/p1_yolo11s_seed2/weights/best.pt --run-name p1_yolo11s_seed2 --model yolo11s` |

The retrained YOLO11s seed 0 is not expected to equal `weights/best.pt` exactly: GPU training is not bit-reproducible, and the notebook's run folder (`PPE_YOLOv119`) suggests best.pt was the ninth run.

## Later phases

Edge speed, memory and energy (Phase 3), cross-site shift (Phase 2) and everything after: not started.
