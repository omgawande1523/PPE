# Phase 3: edge cost and quantisation (cloud CPU measured; boards missing)

Every number below is copied from `results/results.csv`. Regenerate everything, in order, with:

```
pip install -r requirements-edge.txt
python -m ppe.export --config configs/export/p3_export_yolo11s.yaml        # exports + model_size_mb rows
python -m ppe.quant  --config configs/quant/p3_quant_yolo11s.yaml          # shift table per model + paired comparison
python -m ppe.bench  --config configs/bench/p3_bench_cloud_cpu.yaml        # speed and memory, this machine only
python -m ppe.quant  --table p3_quant_yolo11s_compare_20261007T200711Z     # the full comparison tables (results/phase3_quant_table.md)
```

Runs cited: exports `p3_export_yolo11s_20261007T201818Z`; per-model shift runs `p3_quant_yolo11s_<model>_20261007T…`
(listed in the table below); comparison `p3_quant_yolo11s_compare_20261007T200711Z`; bench `p3_bench_cloud_cpu_20261007T203823Z`.
Model: `weights/best.pt` (YOLO11s). Data: golden set v1 (Construction-PPE test, 141 images), the same 18 seeded
synthetic cells as Phase 2. Operating confidence 0.28, chosen on val in Phase 1 and not re-chosen for any export.
INT8 calibration: 300 train images drawn with seed 0. The test split was not used for calibration or for any choice.

## Answer to question 2b: does quantisation make the shift loss worse?

**It depends on how the model is quantised, and the difference is large enough to matter for violations.**

* **OpenVINO INT8** (NNCF, Ultralytics' own export path) costs almost nothing on clean images (mAP@0.5 0.564 vs
  0.568 for FP32, micro recall −0.008, 95% paired interval excludes 0) and does **not** measurably widen the shift
  drop: 8 of 126 recall-widening intervals exclude 0, about what 126 uncorrected 95% intervals give by chance, and
  they point both ways (it loses more under haze s3 and motion blur s2, less under JPEG).
* **ONNX Runtime static INT8** (built here; QDQ, MinMax calibration, decode head kept in float) looks equally
  harmless on clean images (mAP@0.5 0.559, micro recall +0.002) but **widens the drop under low light and haze**:
  under low light s3 the INT8 model loses 0.115 more micro recall than FP32 (interval excludes 0), its violation
  macro recall falls to 0.050 against 0.111 for FP32, no_goggle recall to 0/33 against 3/33, and it flags 16 of
  47 labelled violators against 22 (person-level widening −0.277, interval excludes 0). 23 of 126 recall-widening
  intervals exclude 0, almost all of them negative; the exception is JPEG, where INT8 loses less.
* **FP16** (OpenVINO) is indistinguishable from FP32: 0 of 126 widening intervals exclude 0.
* A clean-images check alone would have passed both INT8 models. The ONNX INT8 failure only appears under shift.
  This is the finding: a quantised model must be verified on shifted data, not only on the clean golden set.

What this does not prove: these are synthetic conditions on 141 images; the four violation classes have 23 to 58
boxes each and 47 labelled violators, so a single violator is 0.021 of person-level recall and per-class intervals
are wide. The comparison is on an x86 CPU; ARM runtimes (TFLite, NCNN on a Pi) and TensorRT INT8 were not measured
(see Missing). No correction for multiple comparisons was applied; count the intervals rather than trusting any one.

### Clean golden set and two shifted cells, per model (conf 0.28)

Box recall per class; violators flagged = person-level violation recall (n = 47 labelled violators);
false alerts = share of violation flags that match no labelled violation.

| Model | Cell | mAP@0.5 | Recall (micro) | Viol. macro recall | no_helmet (40) | no_goggle (33) | no_gloves (58) | no_boots (23) | Violators flagged | False alerts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pytorch_fp32 | clean | 0.570 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| onnx_fp32 | clean | 0.568 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| onnx_int8 | clean | 0.559 | 0.718 | 0.179 | 0.250 | 0.242 | 0.224 | 0.000 | 0.723 | 0.158 |
| openvino_fp32 | clean | 0.568 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| openvino_fp16 | clean | 0.566 | 0.714 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| openvino_int8 | clean | 0.564 | 0.707 | 0.169 | 0.250 | 0.273 | 0.155 | 0.000 | 0.596 | 0.129 |
| ncnn_fp32 | clean | 0.568 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| ncnn_fp16 | clean | 0.568 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| tflite_fp32 | clean | 0.568 | 0.715 | 0.179 | 0.225 | 0.303 | 0.190 | 0.000 | 0.574 | 0.161 |
| onnx_fp32 | low_light_s3 | 0.269 | 0.228 | 0.111 | 0.250 | 0.091 | 0.103 | 0.000 | 0.468 | 0.447 |
| onnx_int8 | low_light_s3 | 0.177 | 0.115 | 0.050 | 0.150 | 0.000 | 0.052 | 0.000 | 0.340 | 0.458 |
| openvino_fp16 | low_light_s3 | 0.267 | 0.226 | 0.111 | 0.250 | 0.091 | 0.103 | 0.000 | 0.468 | 0.447 |
| openvino_int8 | low_light_s3 | 0.264 | 0.214 | 0.095 | 0.250 | 0.061 | 0.069 | 0.000 | 0.426 | 0.513 |
| onnx_fp32 | haze_s3 | 0.540 | 0.643 | 0.154 | 0.200 | 0.242 | 0.172 | 0.000 | 0.596 | 0.103 |
| onnx_int8 | haze_s3 | 0.480 | 0.596 | 0.098 | 0.200 | 0.121 | 0.069 | 0.000 | 0.574 | 0.071 |
| openvino_fp16 | haze_s3 | 0.536 | 0.642 | 0.161 | 0.200 | 0.273 | 0.172 | 0.000 | 0.617 | 0.100 |
| openvino_int8 | haze_s3 | 0.515 | 0.611 | 0.153 | 0.125 | 0.273 | 0.172 | 0.043 | 0.596 | 0.103 |

All 18 cells, every class, the differences and the widening with 95% paired intervals: `results/phase3_quant_table.md`.

Other observations, each from the rows above:
* On the deployed predict path the exported FP32 models (ONNX, OpenVINO, NCNN, TFLite) give exactly the same
  true-positive counts at conf 0.28 as PyTorch in all 19 cells (`onnx_fp32 vs pytorch_fp32` differences are 0.000
  with [0, 0] intervals). Their Ultralytics mAP@0.5 differs by up to 0.03 (e.g. 0.568 vs 0.570 clean); inferred
  cause: Ultralytics validation letterboxes PyTorch images to rectangles and exported models to a fixed 640 x 640
  square. The deployed path letterboxes both to the square.
* The PyTorch run inside `ppe.quant` reproduces the Phase 2 shift run exactly: `python -m ppe.eval --compare
  p2_shift_yolo11s_20261007T181833Z p3_quant_yolo11s_pytorch_fp32_20261007T191238Z` reports 0 of 2,549 rows differ.
* NCNN FP16 equals NCNN FP32 here because the x86 build of NCNN computes in fp32 (FP16 is weight storage only on
  x86; inferred from the identical outputs). On a Pi's ARM cores NCNN uses fp16 arithmetic, so its accuracy must be
  measured on the Pi (`configs/quant/p3_quant_pi.yaml`).
* ONNX INT8 flags more violators on clean images (34 of 47 against 27) at the same false-alert share (0.158 vs
  0.161), so its clean behaviour is not simply "worse": quantisation moved the no_* confidences. Under low light the
  same model flags fewer (16 against 22). n = 47; one violator = 0.021.
* no_boots is 0/23 for every model on clean images (the Phase 1 floor), so quantisation effects on no_boots
  cannot be observed.

## Question 3: speed, memory and size

Device: `cloud_cpu_xeon_4vcpu` (shared cloud VM, Intel Xeon @ 2.10 GHz, 4 vCPU, x86_64). **Not an edge board.** These
numbers rank the formats against each other on x86; they do not predict a Raspberry Pi or a Jetson, whose rows are
missing. Run `p3_bench_cloud_cpu_20261007T203823Z`: batch 1, 640 px, conf 0.28, 200 timed frames after 20 warm-up,
3 runs per model in fresh processes, models interleaved across runs; frames are 50 val images decoded into memory.

| Model | File size (MB) | FPS, mean ± s.d. of 3 runs | FPS per run | Latency p50 (ms) | p95 (ms) | Inference only (ms) | Peak RSS (MB) | Model RSS (MB) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| pytorch_fp32 | 19.2 (best.pt file) | 7.54 ± 0.10 | 7.46 / 7.50 / 7.65 | 136 | 177 | 129.4 | 886 | 218 |
| onnx_fp32 | 37.9 | 11.70 ± 0.10 | 11.80 / 11.69 / 11.60 | 81 | 115 | 75.2 | 888 | 325 |
| onnx_int8 | 10.2 | 15.42 ± 1.01 | 14.92 / 14.76 / 16.59 | 59 | 96 | 54.8 | 738 | 171 |
| openvino_fp32 | 38.1 | 29.52 ± 0.86 | 30.45 / 28.75 / 29.35 | 32 | 45 | 30.0 | 846 | 282 |
| openvino_fp16 | 19.3 | 27.42 ± 4.88 | 32.35 / 22.61 / 27.30 | 35 | 54 | 33.0 | 865 | 302 |
| openvino_int8 | 10.3 | 29.06 ± 1.91 | 29.32 / 27.04 / 30.83 | 33 | 44 | 30.4 | 835 | 272 |
| ncnn_fp32 | 37.9 | 8.33 ± 0.47 | 8.87 / 8.02 / 8.09 | 117 | 148 | 116.3 | 882 | 319 |
| ncnn_fp16 | 19.1 | 8.48 ± 0.22 | 8.62 / 8.22 / 8.59 | 116 | 141 | 114.1 | 882 | 318 |
| tflite_fp32 | 38.0 | 3.54 ± 0.08 | 3.48 / 3.52 / 3.63 | 277 | 331 | 276.2 | 1369 | 807 |
| tflite_fp16 | 19.2 | 3.70 ± 0.23 | 3.58 / 3.56 / 3.97 | 270 | 306 | 265.1 | 1354 | 791 |

Peak RSS is the whole Python process (interpreter, torch, the runtime and the model); model RSS is the growth from
loading the model and running the warm-up. Energy: not measured (no power meter on a VM).

What these show, and what they do not:
* On this x86 CPU, OpenVINO is about 4x PyTorch and 2.5x ONNX Runtime; its INT8 is **not faster** than its FP32
  here (29.1 vs 29.5 FPS, within the run-to-run spread). ONNX Runtime INT8 is 1.3x ONNX FP32 and is the smallest file
  (10.2 MB). INT8 and FP16 roughly quarter and halve the file size.
* The VM is shared: the same model varied by up to 30% between runs (OpenVINO FP16: 22.6 to 32.4 FPS), and an
  earlier full run of this bench 20 minutes before (its rows were removed from results.csv because it recorded the
  CPU as "207", a bug fixed in commit 185f070) gave 4% to 11% higher FPS for every model, with the same ranking. Read differences smaller than the spread as no difference.
* NCNN and TFLite are built for ARM (NEON, XNNPACK on ARM); their x86 speed says nothing about the Pi. The paper's
  20 FPS figure is still unmeasured on any edge board.

## Missing (with the command that produces each)

| Result | Why missing | Command |
|---|---|---|
| Raspberry Pi 4 / 5: FPS, latency, memory, energy per frame for NCNN, TFLite, ONNX | no board in the cloud | `python -m ppe.bench --config configs/bench/p3_bench_pi5.yaml` on the Pi, then `--energy RUN_ID --power-log meter.csv`; procedure in `docs/edge_boards.md` |
| Pi accuracy of NCNN FP16 and TFLite (ARM numerics) | needs the board | `python -m ppe.quant --config configs/quant/p3_quant_pi.yaml --device-label pi5_8gb` |
| Jetson: TensorRT FP16 and INT8 export, accuracy, shift widening, speed, energy | TensorRT needs an NVIDIA GPU | `python -m ppe.export --config configs/export/p3_export_jetson.yaml`, `python -m ppe.quant --config configs/quant/p3_quant_jetson.yaml --device-label ...`, `python -m ppe.bench --config configs/bench/p3_bench_jetson_orin_nano.yaml` |
| TFLite INT8 (Ultralytics' dynamic-range file and the full-integer file) | onnx2tf's INT8 conversion of the 640 px model was killed for memory at ~13-14 GB on the 15 GB VM, with 300, 100 and 30 calibration images alike | on a machine with more RAM (the laptop if it has 32 GB): `python -m ppe.export --config configs/export/p3_export_yolo11s.yaml --only tflite_int8 tflite_int8static`, then `python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml --model tflite_int8 tflite_int8static` and `python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml --compare` |
| ONNX FP16 | Ultralytics 8.3.21 exports ONNX FP16 only from a CUDA device | on the Jetson or Colab: covered by TensorRT FP16 |
| Energy on any device | no power meter on a VM | the board commands above |
| Every YOLO11n row (exports, accuracy, speed) | YOLO11n weights do not exist yet (Phase 1 Colab retrains pending) | after training: `python -m ppe.export --config configs/export/p3_export_yolo11n.yaml`, then the quant and bench configs with the yolo11n weights |
| SH17, CHV, own-site cells for each format | same as Phase 2 (datasets not downloaded) | Phase 2 commands, then re-run `ppe.quant` |
