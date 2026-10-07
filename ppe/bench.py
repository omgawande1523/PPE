"""Speed, memory and energy per frame of each exported model on THIS device (Phase 3).

    python -m ppe.bench --config configs/bench/p3_bench_cloud_cpu.yaml
    python -m ppe.bench --config configs/bench/p3_bench_pi5.yaml               # on a Raspberry Pi 5
    python -m ppe.bench --config configs/bench/p3_bench_jetson_orin_nano.yaml  # on a Jetson
    python -m ppe.bench --energy RUN_ID --power-log meter.csv                  # join a USB meter log afterwards

What is measured, per model, in `runs` separate runs (default 3), each in a
fresh Python process so that memory and caches do not leak between models:
  * end-to-end latency of model.predict on one decoded frame (letterbox,
    inference, NMS; batch 1, conf 0.28, 640 px), over `frames` frames after
    `warmup` frames; FPS = frames / total time. Ultralytics' own split into
    preprocess / inference / postprocess is recorded too.
  * peak resident memory of the process (includes Python, torch and the
    runtime, which is what the board has to hold), and the growth from
    loading the model and running it.
  * optionally power, from one of three sources set in the config:
      file:    a sysfs node read at poll_hz (Jetson INA3221 rails),
      command: a command printing one instantaneous reading,
      csv:     a USB power meter's own log, joined afterwards by time with --energy.
    Energy per frame = mean power over the timed window x window length / frames,
    reported in total and above the idle power measured before the first run.

Frames are decoded into memory before timing, so disk speed is not measured.
Accuracy is not measured here (ppe.eval / ppe.quant do that).
Rows go to results/results.csv with device = the config's device_label.
Numbers from one device are never comparable to another device's.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_path
from ppe.eval import append_rows, fmt, git_commit

BENCH_DIR = REPO_ROOT / "runs" / "bench"


# ----------------------------------------------------------------------------- device facts

def rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except ImportError:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e3
    return float("nan")


def peak_rss_mb() -> float:
    try:
        import resource

        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 1e3 if sys.platform != "darwin" else kb / 1e6
    except ImportError:                      # Windows
        import psutil

        return psutil.Process().memory_info().peak_wset / 1e6


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware", "model\t")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def board_model() -> str:
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(p).read_text(encoding="utf-8").strip("\x00\n ")
        except OSError:
            continue
    return ""


def temperature_c() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()) / 1000
    except (OSError, ValueError):
        return float("nan")


def device_facts() -> dict:
    import ultralytics

    facts = {"cpu": cpu_name(), "board": board_model(), "cores": os.cpu_count(), "machine": platform.machine(),
             "os": platform.platform(), "python": platform.python_version(), "ultralytics": ultralytics.__version__}
    for mod in ("torch", "onnxruntime", "openvino", "ncnn", "tensorflow", "tflite_runtime", "tensorrt"):
        try:
            facts[mod] = __import__(mod).__version__
        except Exception:      # absent, or no __version__
            pass
    try:
        gov = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
        facts["governor"] = gov
    except OSError:
        pass
    return facts


# ----------------------------------------------------------------------------- power

class Sampler:
    """Polls a power reading in a background thread: [(unix time, watts)]."""

    def __init__(self, spec: dict):
        self.spec, self.samples, self._stop = spec, [], threading.Event()
        self.scale = float(spec.get("scale", 1.0))
        self.period = 1.0 / float(spec.get("poll_hz", 10))
        self._t = threading.Thread(target=self._loop, daemon=True)

    def read(self) -> float:
        if self.spec["source"] == "file":
            return float(Path(self.spec["file"]).read_text().split()[0]) * self.scale
        out = subprocess.run(self.spec["command"], shell=True, capture_output=True, text=True, timeout=5).stdout
        return float(out.strip().split()[0]) * self.scale

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.time(), self.read()))
            except Exception:   # a missed reading is skipped, not invented
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def mean_power(samples: list[tuple[float, float]], t0: float, t1: float) -> tuple[float, int]:
    w = [p for t, p in samples if t0 <= t <= t1]
    return (float(np.mean(w)) if w else float("nan")), len(w)


def read_power_log(path: Path, spec: dict) -> list[tuple[float, float]]:
    """USB meter CSV -> [(unix time, watts)]. Column names come from the config's power.csv_columns."""
    cols = spec.get("csv_columns", {"time": "time", "watts": "watts"})
    out = []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            t = float(r[cols["time"]])
            if "watts" in cols:
                w = float(r[cols["watts"]])
            else:
                w = float(r[cols["volts"]]) * float(r[cols["amps"]])
            out.append((t + float(spec.get("time_offset_s", 0)), w * float(spec.get("scale", 1.0))))
    return out


# ----------------------------------------------------------------------------- one run (child process)

def load_frames(cfg: dict) -> list[np.ndarray]:
    import cv2

    src = cfg["images"]
    with (REPO_ROOT / src["data"]).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if src["split"] == "test":
        sys.exit("Benchmark frames come from train or val; test is the golden set.")
    d = REPO_ROOT / data["path"] / data[src["split"]]
    paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[: int(src["n"])]
    if not paths:
        sys.exit(f"No images in {d}. Run: python -m ppe.data --download construction-ppe")
    return [cv2.imread(str(p)) for p in paths]


def worker(cfg: dict, name: str, run: int) -> dict:
    """Time one model once; print a JSON line. Runs in its own process."""
    from ultralytics import YOLO

    from ppe.infer import set_seed

    set_seed(int(cfg["seed"]))
    frames = load_frames(cfg)
    rss_before = rss_mb()
    spec = cfg["models"][name]
    model = YOLO(str(REPO_ROOT / spec["weights"]), task="detect")
    kw = dict(conf=float(cfg["conf"]), iou=float(cfg["nms_iou"]), imgsz=int(cfg["imgsz"]), device=cfg["device"],
              half=bool(spec.get("half", False)), verbose=False)
    for i in range(int(cfg["warmup"])):
        model.predict(frames[i % len(frames)], **kw)
    rss_loaded = rss_mb()
    n = int(cfg["frames"])
    lat, pre, inf, post = [], [], [], []
    temp0 = temperature_c()
    t_start = time.time()
    for i in range(n):
        t0 = time.perf_counter()
        r = model.predict(frames[i % len(frames)], **kw)[0]
        lat.append((time.perf_counter() - t0) * 1000)
        pre.append(r.speed["preprocess"])
        inf.append(r.speed["inference"])
        post.append(r.speed["postprocess"])
    t_end = time.time()
    return {"model": name, "run": run, "frames": n, "t_start": t_start, "t_end": t_end,
            "latency_ms": lat, "preprocess_ms": pre, "inference_ms": inf, "postprocess_ms": post,
            "rss_before_load_mb": rss_before, "rss_after_warmup_mb": rss_loaded, "peak_rss_mb": peak_rss_mb(),
            "temp_start_c": temp0, "temp_end_c": temperature_c(),
            "threads": os.environ.get("OMP_NUM_THREADS", "runtime default")}


# ----------------------------------------------------------------------------- orchestration

def load_cfg(path: str) -> dict:
    p = Path(path)
    with (p if p.is_absolute() else REPO_ROOT / p).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def summarise(cfg: dict, results: list[dict], base: dict, power: dict | None, idle_w: float) -> list[dict]:
    rows = []
    facts = base.pop("_facts")
    fact_note = "; ".join(f"{k}={v}" for k, v in facts.items() if v)

    def add(prec, metric, value, n, notes):
        rows.append({**base, "precision_mode": prec, "metric": metric, "value": fmt(float(value)), "n": n,
                     "notes": notes})

    for name in cfg["models"]:
        rs = [r for r in results if r["model"] == name]
        if not rs:
            continue
        spec = cfg["models"][name]
        w = REPO_ROOT / spec["weights"]
        base_w = f"{spec['weights']}@{sha256_path(w)[:12]}"
        rows_before = len(rows)
        n = rs[0]["frames"]
        proto = (f"batch 1, {cfg['imgsz']} px, conf {cfg['conf']}, {n} timed frames after {cfg['warmup']} warm-up, "
                 f"{len(rs)} runs in fresh processes; frames pre-decoded from {cfg['images']['split']} "
                 f"({cfg['images']['n']} images cycled); {fact_note}")
        fps = [r["frames"] / (sum(r["latency_ms"]) / 1000) for r in rs]
        for r, f in zip(rs, fps):
            add(name, f"fps_e2e_run{r['run']}", f, n, f"{proto}; temp {r['temp_start_c']}->{r['temp_end_c']} C")
        add(name, "fps_e2e_mean", np.mean(fps), n, f"{proto}; mean of {len(rs)} runs")
        add(name, "fps_e2e_std", np.std(fps, ddof=1) if len(fps) > 1 else float("nan"), n,
            f"{proto}; sample std over {len(rs)} runs")
        lat = np.concatenate([r["latency_ms"] for r in rs])
        add(name, "latency_ms_p50", np.percentile(lat, 50), len(lat), f"{proto}; all runs pooled; end to end")
        add(name, "latency_ms_p95", np.percentile(lat, 95), len(lat), f"{proto}; all runs pooled; end to end")
        for k in ("preprocess_ms", "inference_ms", "postprocess_ms"):
            add(name, f"{k}_mean", np.mean(np.concatenate([r[k] for r in rs])), len(lat),
                f"{proto}; Ultralytics' own timing of this stage")
        add(name, "peak_rss_mb", max(r["peak_rss_mb"] for r in rs), len(rs),
            f"{proto}; max over runs of the process's peak resident memory (Python + runtime + model)")
        add(name, "model_rss_mb", np.mean([r["rss_after_warmup_mb"] - r["rss_before_load_mb"] for r in rs]), len(rs),
            f"{proto}; resident memory after warm-up minus before loading the model, mean over runs")
        if power is not None:
            pw = [mean_power(power["samples"], r["t_start"], r["t_end"]) for r in rs]
            if all(k for _, k in pw):
                e = [p * (r["t_end"] - r["t_start"]) / r["frames"] * 1000 for (p, _), r in zip(pw, rs)]
                src = power["desc"]
                add(name, "power_w_mean", np.mean([p for p, _ in pw]), sum(k for _, k in pw),
                    f"{proto}; {src}; mean over runs of the mean power in each timed window")
                add(name, "energy_mj_per_frame", np.mean(e), len(rs), f"{proto}; {src}; total board energy, mean of runs")
                if np.isfinite(idle_w):
                    ea = [(p - idle_w) * (r["t_end"] - r["t_start"]) / r["frames"] * 1000 for (p, _), r in zip(pw, rs)]
                    add(name, "energy_mj_per_frame_above_idle", np.mean(ea), len(rs),
                        f"{proto}; {src}; minus idle {idle_w:.3f} W")
        for r_ in rows[rows_before:]:
            r_["weights"] = base_w
    if power is not None and np.isfinite(idle_w):
        add("idle", "power_w_idle", idle_w, power.get("n_idle", 0), f"{power['desc']}; before the first run")
        rows[-1]["weights"] = ""
    return rows


def run(cfg: dict, command: str, only: list[str] | None) -> list[dict]:
    if not cfg.get("device_label"):
        sys.exit("Set device_label in the config (e.g. pi5_8gb, jetson_orin_nano_8gb). It names the device in results.csv.")
    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    out = BENCH_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in cfg["models"] if not only or n in only]
    missing = [n for n in names if not (REPO_ROOT / cfg["models"][n]["weights"]).exists()]
    for n in missing:
        print(f"[{n}] weights not found ({cfg['models'][n]['weights']}): skipped. "
              f"Build it with python -m ppe.export --config <export config>.", flush=True)
    names = [n for n in names if n not in missing]
    pspec = cfg.get("power") or {"source": "none"}
    sampler = Sampler(pspec) if pspec["source"] in ("file", "command") else None
    cfg_file = out / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    results: list[dict] = []
    idle_w, n_idle = float("nan"), 0
    ctx = sampler if sampler else _Null()
    with ctx:
        idle_s = float(pspec.get("idle_seconds", 30)) if pspec["source"] != "none" else 0
        t_idle0 = time.time()
        if idle_s:
            print(f"Idle power for {idle_s:.0f} s; leave the board alone ...", flush=True)
            time.sleep(idle_s)
        t_idle1 = time.time()
        for k in range(1, int(cfg["runs"]) + 1):           # interleave models across runs
            for name in names:
                print(f"[{name}] run {k}/{cfg['runs']} ...", flush=True)
                p = subprocess.run([sys.executable, "-m", "ppe.bench", "--worker", str(cfg_file), name, str(k)],
                                   cwd=REPO_ROOT, capture_output=True, text=True)
                line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
                if p.returncode or not line:
                    print(p.stdout[-2000:], p.stderr[-3000:], sep="\n")
                    sys.exit(f"[{name}] run {k} failed")
                r = json.loads(line[-1])
                results.append(r)
                print(f"[{name}] run {k}: {r['frames'] / (sum(r['latency_ms']) / 1000):.2f} FPS, "
                      f"peak RSS {r['peak_rss_mb']:.0f} MB", flush=True)
                if float(cfg.get("cooldown_s", 0)):
                    time.sleep(float(cfg["cooldown_s"]))
    power = None
    if sampler:
        idle_w, n_idle = mean_power(sampler.samples, t_idle0, t_idle1)
        power = {"samples": sampler.samples, "n_idle": n_idle,
                 "desc": f"power from {pspec['source']} {pspec.get('file') or pspec.get('command')} "
                         f"at {pspec.get('poll_hz', 10)} Hz x {pspec.get('scale', 1.0)}"}
        with (out / "power_samples.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows([("time", "watts")] + sampler.samples)
    (out / "windows.json").write_text(json.dumps({
        "idle": [t_idle0, t_idle1],
        "runs": [{k: r[k] for k in ("model", "run", "frames", "t_start", "t_end")} for r in results]}, indent=1))
    (out / "raw.json").write_text(json.dumps(results))
    base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": cfg["phase"],
            "model": cfg["model"], "weights": "", "device": cfg["device_label"], "dataset": "construction-ppe",
            "split": cfg["images"]["split"], "condition": "bench", "class": "all", "seed": cfg["seed"],
            "command": command, "_facts": device_facts()}
    rows = summarise(cfg, results, base, power, idle_w)
    if pspec["source"] == "csv":
        print(f"Energy: once the meter log is copied here, run "
              f"python -m ppe.bench --energy {run_id} --power-log PATH_TO_METER.csv")
    elif pspec["source"] == "none":
        print("Energy not measured (power.source: none). See docs/edge_boards.md for the USB meter procedure.")
    return rows


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def energy(run_id: str, log: Path) -> list[dict]:
    """Join a USB power meter's log with the timed windows of an earlier bench run."""
    out = BENCH_DIR / run_id
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    results = json.loads((out / "raw.json").read_text())
    win = json.loads((out / "windows.json").read_text())
    samples = read_power_log(log, cfg["power"])
    idle_w, n_idle = mean_power(samples, *win["idle"])
    power = {"samples": samples, "n_idle": n_idle, "desc": f"power from meter log {log.name}"}
    with (REPO_ROOT / "results" / "results.csv").open(newline="", encoding="utf-8") as fh:
        prev = next((r for r in csv.DictReader(fh) if r["run_id"] == run_id), None)
    if prev is None:
        sys.exit(f"Run {run_id} not in results.csv")
    base = {k: prev[k] for k in ("run_id", "date", "git_commit", "phase", "model", "device", "dataset", "split",
                                 "condition", "class", "seed")}
    base.update(weights="", command=f"python -m ppe.bench --energy {run_id} --power-log {log.name}",
                _facts={"meter_log_sha256": sha256_path(log)[:12]})
    rows = [r for r in summarise(cfg, results, base, power, idle_w)
            if r["metric"].startswith(("energy", "power"))]
    if not rows:
        sys.exit("The meter log has no samples inside the timed windows; check its clock (power.time_offset_s).")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.bench", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="bench YAML, e.g. configs/bench/p3_bench_cloud_cpu.yaml")
    ap.add_argument("--only", nargs="+", help="benchmark only these models from the config")
    ap.add_argument("--energy", metavar="RUN_ID", help="add energy rows to an earlier run from a meter log")
    ap.add_argument("--power-log", type=Path, help="USB power meter CSV (with --energy)")
    ap.add_argument("--worker", nargs=3, metavar=("CONFIG", "MODEL", "RUN"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        cfg = yaml.safe_load(Path(args.worker[0]).read_text(encoding="utf-8"))
        print(json.dumps(worker(cfg, args.worker[1], int(args.worker[2]))))
        return 0
    command = "python -m ppe.bench " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    if args.energy:
        if not args.power_log:
            ap.error("--energy needs --power-log")
        rows = energy(args.energy, args.power_log)
    else:
        if not args.config:
            ap.error("--config is required")
        rows = run(load_cfg(args.config), command, args.only)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {rows[0]['run_id']}")
    for r in rows:
        if r["metric"] in ("fps_e2e_mean", "latency_ms_p50", "peak_rss_mb", "energy_mj_per_frame"):
            print(f"  {r['precision_mode']:15s} {r['metric']:22s} {float(r['value']):9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
