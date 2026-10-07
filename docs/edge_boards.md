# Phase 3 on the boards: Raspberry Pi and Jetson

Everything in `results/phase3_edge.md` that says `cloud_cpu_xeon_4vcpu` was measured on a cloud VM.
Speed, memory and energy on the boards are **missing** until the steps below have run on them.
Each step appends rows to `results/results.csv` with the board's name in the `device` column.

## The protocol (same on every board)

| Setting | Value | Why |
|---|---|---|
| Input | 640 x 640, batch 1, frames from the val split decoded into memory | what one camera loop does; disk speed is not measured |
| Operating confidence | 0.28 (chosen on val in Phase 1) | NMS cost depends on how many boxes survive |
| Warm-up | 20 frames (50 on the Jetson) | first frames include lazy init and, on TensorRT, kernel selection |
| Timed frames | 200 per run (500 on the Jetson) | |
| Runs | **3 per model**, each in a fresh process, models interleaved (A B C, A B C, A B C) | separates run-to-run spread from model differences; drift hits every model equally |
| Cool-down | 60 s between runs; SoC temperature logged at the start and end of every run | a throttled run is visible, not hidden in a mean |
| Idle power | 60 s with nothing running, before the first run | energy above idle is reported next to total energy |
| Reported | mean and sample s.d. of FPS over the 3 runs, p50/p95 latency pooled, peak RSS, energy per frame (mJ) total and above idle | |

Conditions to keep fixed and write down in the PR or the notes: power supply, cooling (fan or heatsink),
ambient temperature, OS image, headless (no desktop), Ethernet only, nothing else running.

## 0. On the laptop: build the exports once

```
pip install -r requirements-edge.txt
python -m ppe.export --config configs/export/p3_export_yolo11s.yaml
```

This writes `weights/export/yolo11s/` and its `manifest.json` (sha256 of every export). Copy that folder
to the board; after copying, check one hash on the board with `sha256sum` (or `python -c "from ppe.data import
sha256_path; ..."` for folders) against the manifest. The NCNN and TFLite files are the Pi's; TensorRT
engines are built on the Jetson itself (step J2).

## Raspberry Pi 4 / 5

### P1. Set up (Raspberry Pi OS Bookworm 64-bit Lite, Python 3.11)

```
sudo apt update && sudo apt install -y git python3-venv libgl1
git clone https://github.com/omgawande1523/PPE.git && cd PPE
git checkout claude/phase3-edge-cost-3r3ys2         # or the branch this work merged into
python3 -m venv .venv && . .venv/bin/activate
pip install ultralytics==8.3.21 torch==2.6.0 torchvision==0.21.0 opencv-python-headless PyYAML==6.0.2 \
            python-dotenv==1.0.1 lap==0.5.12 psutil ncnn onnxruntime==1.30.0 tflite-runtime
python -m ppe.data --download construction-ppe       # or copy datasets/construction-ppe from the laptop
scp -r LAPTOP:PPE/weights/export/yolo11s weights/export/   # from step 0
```

If `tflite-runtime` has no wheel for the Pi's Python, install `tensorflow` instead (slower to import,
same interpreter). Write down which one was used: it is recorded in the notes of every bench row.

### P2. Fix the clocks and check for throttling

```
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
vcgencmd get_throttled        # must print throttled=0x0 before AND after the bench
vcgencmd measure_temp
```

A Pi 5 needs the active cooler; a Pi 4 needs at least a heatsink and fan. If `get_throttled` is not 0x0
after the bench, the numbers are not valid: improve cooling and run again.

### P3. Power meter (USB-C inline meter, e.g. FNIRSI FNB58 or UM25C/UM34C)

1. Wire: official supply (27 W for Pi 5, 15 W for Pi 4) -> meter -> Pi. Nothing else on the meter.
2. Log to a PC at 10 Hz or faster. Export CSV with a time column and either watts or volts and amps.
3. Make the meter's time column unix seconds. If the meter software logs time since start, note the unix
   time when you press record (`date +%s` on the PC) and put it in `power.time_offset_s` of the bench config.
   Both the PC and the Pi must be NTP-synced (`timedatectl` says "System clock synchronized: yes").
4. Set `power.csv_columns` in `configs/bench/p3_bench_pi5.yaml` (or `p3_bench_pi4.yaml`) to the CSV's
   column names, and `device_label` to the board, e.g. `pi5_8gb`.

### P4. Run

Start the meter's log first, then on the Pi:

```
python -m ppe.bench --config configs/bench/p3_bench_pi5.yaml        # about 1-2 h: 8 models x 3 runs
```

It sits idle for 60 s first (the idle-power window): do not touch the board. Stop the meter log after it
prints `Appended ... rows`. Copy the meter CSV to the Pi and join it:

```
python -m ppe.bench --energy RUN_ID --power-log meter.csv           # RUN_ID is printed by the bench
```

### P5. Accuracy on the Pi (NCNN computes in fp16 on ARM, so it can differ from the cloud numbers)

```
python -m ppe.quant --config configs/quant/p3_quant_pi.yaml --device-label pi5_8gb    # clean golden set only
```

## Jetson (Orin Nano, JetPack 6)

### J1. Set up

```
sudo apt update && sudo apt install -y git python3-venv python3-libnvinfer
git clone https://github.com/omgawande1523/PPE.git && cd PPE && git checkout claude/phase3-edge-cost-3r3ys2
python3 -m venv --system-site-packages .venv && . .venv/bin/activate     # system site: TensorRT's Python bindings
pip install ultralytics==8.3.21 PyYAML==6.0.2 python-dotenv==1.0.1 lap==0.5.12 psutil onnx onnxslim
```

PyPI's aarch64 torch has no CUDA. Install the torch and torchvision wheels NVIDIA publishes for your
JetPack version (NVIDIA's "Installing PyTorch for Jetson Platform" page, or the Ultralytics Jetson guide),
and `onnxruntime-gpu` from the Jetson Zoo. Check: `python -c "import torch; print(torch.cuda.is_available())"`
prints True. Record the torch, TensorRT and JetPack versions (`cat /etc/nv_tegra_release`).

### J2. Build the engines on the Jetson (INT8 calibrated on the same 300 train images)

```
sudo nvpmodel -m 0 && sudo jetson_clocks        # max power mode, clocks pinned; note the mode name
python -m ppe.export --config configs/export/p3_export_jetson.yaml
```

### J3. Accuracy of the engines (only measurable on the Jetson)

```
python -m ppe.quant --config configs/quant/p3_quant_jetson.yaml --device-label jetson_orin_nano_8gb_maxn
```

This is the full 18-cell shift table plus clean, for TensorRT FP16 and INT8 against ONNX FP32 on the same board.

### J4. Speed, memory, energy

Power, two sources:
* primary: a USB-C or barrel-jack meter at the supply, exactly as P3 (set `power.source: csv`);
* cross-check: the module's INA3221 VDD_IN rail, read by the config's `power.command`. First check the path:
  `cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/in1_label` must say `VDD_IN`; fix the path if not.

```
python -m ppe.bench --config configs/bench/p3_bench_jetson_orin_nano.yaml
python -m ppe.bench --energy RUN_ID --power-log meter.csv       # when the primary meter was used
tegrastats --interval 1000 --logfile tegrastats.log &           # optional: GPU/EMC load and temperatures alongside
```

## After the boards

Commit the new `results/results.csv` rows and the meter CSVs (put them under `results/power_logs/`), then
regenerate the report tables with the commands at the top of `results/phase3_edge.md`.
