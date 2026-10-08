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

## Phase 2: real site shift (external sets and own footage)

The synthetic conditions are measured (`results/phase2_shift.md`). Every real-site cell is missing.
The shift command evaluates whatever is on disk, so after any of the steps below, re-run
`python -m ppe.shift --config configs/shift/p2_shift_yolo11s.yaml` (or add `--only sh17` to append just that site; the clean reference is always re-measured with it).

| Cell | Why it is missing | Command once the data is on disk |
|---|---|---|
| SH17 (1,620-image validation list) | Kaggle is not reachable from the cloud container; the dataset has no other mirror with labels | Download `mugheesahmad/sh17-dataset-for-ppe-detection` from Kaggle, unzip, `python -m ppe.shift --convert sh17 --src path/to/sh17`, then the shift command |
| CHV (1,330 images) | Only on Google Drive and Baidu, neither reachable from the cloud container | Download from the link in github.com/ZijianWang-ZW/PPE_detection, unzip, `python -m ppe.shift --convert chv --src path/to/chv`, then the shift command |
| Own site (at least 200 frames, staged violations) | Not recorded yet; needs site or workshop access and labelling | Label with the 11 classes into `datasets/own_site/images/test` and `labels/test`, then the shift command |

Both are browser downloads (Kaggle needs a free account); evaluation needs no GPU.

## Phase 3: edge boards, TFLite INT8, TensorRT, YOLO11n

Measured on the cloud CPU only (`results/phase3_edge.md`). Board procedure: `docs/edge_boards.md`.

| Result | Why it is missing | Command |
|---|---|---|
| Raspberry Pi 4/5 speed, memory, energy | no board in the cloud | `python -m ppe.bench --config configs/bench/p3_bench_pi5.yaml` (or `p3_bench_pi4.yaml`), then `python -m ppe.bench --energy RUN_ID --power-log meter.csv` |
| Pi accuracy of NCNN FP16 / TFLite (ARM numerics) | needs the board | `python -m ppe.quant --config configs/quant/p3_quant_pi.yaml --device-label pi5_8gb` |
| Jetson TensorRT FP16/INT8: build, accuracy, shift widening, speed, energy | TensorRT needs an NVIDIA GPU | `python -m ppe.export --config configs/export/p3_export_jetson.yaml`; `python -m ppe.quant --config configs/quant/p3_quant_jetson.yaml --device-label jetson_orin_nano_8gb_maxn`; `python -m ppe.bench --config configs/bench/p3_bench_jetson_orin_nano.yaml` |
| TFLite INT8 (dynamic-range and full-integer) | onnx2tf INT8 conversion killed for memory (~13-14 GB) on the 15 GB VM | on a machine with more RAM: `python -m ppe.export --config configs/export/p3_export_yolo11s.yaml --only tflite_int8 tflite_int8static`, then `python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml --model tflite_int8 tflite_int8static` and `--compare` |
| Every YOLO11n Phase 3 row | YOLO11n not trained yet (Phase 1 above) | `python -m ppe.export --config configs/export/p3_export_yolo11n.yaml`, then quant and bench configs pointed at the YOLO11n exports |

## Phase 4: held-out site, real temporal signal, YOLO11n

Measured with leave-one-condition-out on synthetic shift (`results/phase4_estimator.md`). The brief's done-criterion is the
error on a held-out SITE; no external site is on disk, so that number does not exist yet.

| Result | Why it is missing | Command |
|---|---|---|
| Estimator error on a held-out site (SH17, CHV or own footage) | no external site converted yet (Phase 2 rows above) | convert the site as in Phase 2, then `python -m ppe.estimator --config configs/estimator/p4_estimator_yolo11s.yaml` (sites on disk are scored automatically; cached signals are reused) |
| Temporal inconsistency on real video (tracked persons whose PPE flickers) | the datasets are still images; the measured signal is a still-image proxy (shaken, re-noised copies) | record site video, run `python -m ppe.infer --source VIDEO` (it writes the JSON-lines runtime log with track IDs to runs/logs/); a per-track flicker reader for that log is Phase 5 work, and true recall needs a labelled sample of frames |
| Lead time on a real drift (a camera or site that changes over days) | no time-stamped footage; streams are synthetic mixes of test images | same as the row above, with labels on a sample of frames per day |
| Every YOLO11n Phase 4 row | YOLO11n not trained yet (Phase 1 above) | copy the config with `weights:` pointed at the YOLO11n best.pt and `feature_layer` checked, then the same command |

## Phase 5: promotion, real teacher, human check, real site

Measured on synthetic streams (`results/phase5_loop.md`). The done-criterion asks for one promotion and one rejection;
two rejections are logged and there is **no promotion**.

| Result | Why it is missing | Command |
|---|---|---|
| A promoted candidate | every candidate (2 loop runs, 2 val-only recipes) lost golden/val no_* recall; needs a decision: Colab fine-tune with full replay, or a gate change | after the decision: a new `configs/loop/p5_site_shift_*.yaml`, then `python -m ppe.loop --config` it |
| Real teacher (YOLO-World) pseudo-labels and their error rate | YOLO-World needs the CLIP text encoder, whose download (openaipublic.azureedge.net) is blocked in the cloud container | on the laptop or Colab: `pip install git+https://github.com/ultralytics/CLIP.git`, then `python -m ppe.loop --config configs/loop/p5_site_shift_yolo_world.yaml` |
| Human pseudo-label error rate and human minutes | nobody has checked the 20-frame samples yet (runs/ is not committed: re-run the scenario to regenerate them) | fill `runs/loop/<run>/human_check/human_check.csv`, then `python -m ppe.loop --human-check runs/loop/<run>` |
| Loop on a real site stream | no external site or own footage (Phase 2 rows) | add a stream segment type that reads site frames, then the site-shift config |
| Monitor recalibrated after a promotion, measured on the remaining stream | no promotion happened | runs automatically once a candidate is promoted |

## Later phases

Phase 6: draft in `paper/` (`python -m ppe.paper`). It cites only rows that exist; every claim the
missing results above would support is flagged in `paper/flags.md`. Two numbers in the phase reports are not rows yet:

| Result | Why it is missing | Command |
|---|---|---|
| Count of quantisation-widening intervals that exclude 0 (results/phase3_edge.md says 8/126 OpenVINO INT8, 23/126 ONNX INT8) | computed in the report, not written to results.csv | add a summary row to `python -m ppe.quant --compare`, then cite it |
| best.pt file size in the bench table | not written by ppe.export (only exported files are) | add a `model_size_mb` row for `pytorch_fp32` in `ppe.export` |
