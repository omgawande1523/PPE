"""Export a weights file to edge formats at FP32, FP16 and INT8 (Phase 3).

One command exports every target in the config that can be built on this
machine and skips the rest with the reason and the command for the machine
that can build it:

    python -m ppe.export --config configs/export/p3_export_yolo11s.yaml
    python -m ppe.export --config configs/export/p3_export_yolo11s.yaml --only onnx_int8 openvino_int8
    python -m ppe.export --config configs/export/p3_export_jetson.yaml     # on the Jetson: TensorRT

Targets (format_precision):
    onnx_fp32, onnx_int8                      ONNX Runtime; INT8 = static QDQ quantisation (see below)
    openvino_fp32, openvino_fp16, openvino_int8   OpenVINO (x86 only); INT8 via NNCF, Ultralytics' own path
    ncnn_fp32, ncnn_fp16                      NCNN (Raspberry Pi); FP16 = fp16 weight storage
    tflite_fp32, tflite_fp16, tflite_int8     TensorFlow Lite via onnx2tf (Raspberry Pi); tflite_int8 is what
                                              Ultralytics 8.3.21 calls int8: dynamic-range (int8 weights,
                                              float activations), no calibration
    tflite_int8static                         onnx2tf's *_integer_quant.tflite from the same conversion: int8
                                              weights and activations, calibrated, float input/output
    engine_fp16, engine_int8                  TensorRT; needs an NVIDIA GPU, so it is built on the Jetson itself

INT8 calibration uses a seeded random sample of the split named in the config
(train or val). The test split is the frozen golden set and is refused.

ONNX INT8 is not offered by Ultralytics 8.3.21, so it is built here with
onnxruntime.quantization.quantize_static: QDQ format, per-channel int8
weights, uint8 activations, MinMax calibration. The decode part of the
Detect head (DFL, Sigmoid, Concat and the box arithmetic in model.23, but
not its convolutions) stays in float, the same rule Ultralytics applies to
its OpenVINO INT8 export: boxes in pixels and class scores in 0..1 would
otherwise share one quantisation scale. This rule was fixed before any
quantised model was evaluated.

Every export is written under out_dir with a name Ultralytics recognises,
its sha256 and size go to <out_dir>/manifest.json, and one model_size_mb row
per export goes to results/results.csv.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_path
from ppe.eval import append_rows, fmt, git_commit
from ppe.infer import set_seed

TARGETS = {
    "onnx": ["fp32", "int8"],
    "openvino": ["fp32", "fp16", "int8"],
    "ncnn": ["fp32", "fp16"],
    "tflite": ["fp32", "fp16", "int8", "int8static"],
    "engine": ["fp16", "int8"],
}


def target_path(out_dir: Path, stem: str, fmt_: str, prec: str) -> Path:
    """File or directory name of one export; the suffix is what Ultralytics' AutoBackend keys on."""
    return {
        "onnx": out_dir / f"{stem}_{prec}.onnx",
        "openvino": out_dir / f"{stem}_{prec}_openvino_model",
        "ncnn": out_dir / f"{stem}_{prec}_ncnn_model",
        "tflite": out_dir / f"{stem}_{prec}.tflite",
        "engine": out_dir / f"{stem}_{prec}.engine",
    }[fmt_]


def size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


# ----------------------------------------------------------------------------- calibration

def calibration_images(cfg: dict, n_max: int | None = None) -> list[Path]:
    """Seeded draw of n_images from the calibration split; n_max keeps the first n_max of that same draw."""
    cal = cfg["calibration"]
    if cal["split"] not in ("train", "val"):
        sys.exit(f"INT8 calibration split must be train or val, not {cal['split']!r} (test is the golden set).")
    with (REPO_ROOT / cal["data"]).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    d = REPO_ROOT / data["path"] / data[cal["split"]]
    if not d.is_dir():
        sys.exit(f"Dataset not found at {d}. Run: python -m ppe.data --download construction-ppe")
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    n = min(int(cal["n_images"]), len(imgs))
    idx = np.random.default_rng(int(cal["seed"])).choice(len(imgs), n, replace=False)
    if n_max is not None:
        idx = idx[:n_max]
    return [imgs[i] for i in sorted(idx)]


def calibration_yaml(cfg: dict, images: list[Path], work: Path) -> Path:
    """Data YAML whose train and val both list the calibration images (Ultralytics calibrates on 'val')."""
    with (REPO_ROOT / cfg["calibration"]["data"]).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    work.mkdir(parents=True, exist_ok=True)
    lst = work / "calibration_images.txt"
    lst.write_text("\n".join(str(p) for p in images) + "\n", encoding="utf-8")
    out = work / "calibration.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": str(work), "train": str(lst), "val": str(lst), "names": data["names"]}, fh,
                       sort_keys=False)
    return out


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Square letterbox with grey 114 padding, as Ultralytics predicts with a fixed-shape exported model."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = img
    return out


# ----------------------------------------------------------------------------- exporters

def ultralytics_export(pt: Path, fmt_: str, prec: str, cfg: dict, calib: Path | None, work: Path) -> Path:
    """Run Ultralytics' exporter on a private copy of the weights; return what it produced."""
    from ultralytics import YOLO

    work.mkdir(parents=True, exist_ok=True)
    local = work / pt.name
    shutil.copy2(pt, local)
    kw = dict(format=fmt_, imgsz=int(cfg["imgsz"]), batch=1, device=cfg.get("device", "cpu"),
              half=prec == "fp16", int8=prec.startswith("int8"), simplify=True)
    if prec.startswith("int8"):
        kw.update(data=str(calib), split="val")
    out = YOLO(str(local), task="detect").export(**kw)
    if not out:
        raise RuntimeError(f"Ultralytics returned nothing for {fmt_} {prec}")
    return Path(out)


def add_tflite_metadata(path: Path, metadata: dict) -> None:
    """Append Ultralytics metadata as a zip member, which AutoBackend reads when tflite_support is absent."""
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metadata.txt", repr(metadata))


def _is_zip_with_members(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            return bool(zf.namelist())
    except zipfile.BadZipFile:
        return False


def onnx_int8(fp32_onnx: Path, images: list[Path], imgsz: int, out: Path, work: Path) -> dict:
    """Static QDQ INT8 of an ONNX model with onnxruntime, calibrated on the given images."""
    import cv2
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    work.mkdir(parents=True, exist_ok=True)
    pre = work / "pre.onnx"
    quant_pre_process(str(fp32_onnx), str(pre), skip_symbolic_shape=True)
    model = onnx.load(str(pre))
    head = [n for n in model.graph.node if n.name.startswith("/model.23/")]
    exclude = [n.name for n in head if n.op_type != "Conv" or "/dfl/" in n.name]
    input_name = model.graph.input[0].name

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(images)

        def get_next(self):
            p = next(self.it, None)
            if p is None:
                return None
            x = letterbox(cv2.imread(str(p)), imgsz)[:, :, ::-1].transpose(2, 0, 1)[None]
            return {input_name: np.ascontiguousarray(x, dtype=np.float32) / 255.0}

    quantize_static(str(pre), str(out), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude)
    # Carry Ultralytics' metadata (names, stride, imgsz) over; quantize_static drops it.
    src, q = onnx.load(str(fp32_onnx)), onnx.load(str(out))
    del q.metadata_props[:]
    for p in src.metadata_props:
        q.metadata_props.add(key=p.key, value=p.value)
    onnx.save(q, str(out))
    return {"excluded_nodes": len(exclude), "head_nodes": len(head),
            "method": "onnxruntime quantize_static, QDQ, per-channel QInt8 weights, QUInt8 activations, MinMax"}


def export_target(pt: Path, fmt_: str, prec: str, cfg: dict, images: list[Path], calib: Path | None,
                  dest: Path, work: Path) -> dict:
    info: dict = {}
    if fmt_ == "onnx" and prec == "int8":
        fp32 = target_path(dest.parent, cfg["stem"], "onnx", "fp32")
        if not fp32.exists():
            fp32_tmp = ultralytics_export(pt, "onnx", "fp32", cfg, None, work / "fp32")
            shutil.move(str(fp32_tmp), fp32)
        info = onnx_int8(fp32, images, int(cfg["imgsz"]), dest, work)
        info["from"] = fp32.name
        return info
    if fmt_ == "onnx" and prec == "fp16":
        raise SystemExit("ONNX FP16 needs a CUDA device in Ultralytics 8.3.21; use engine_fp16 on the Jetson.")
    try:
        produced = ultralytics_export(pt, fmt_, prec, cfg, calib, work)
    except Exception as e:
        # Ultralytics' last TFLite step embeds metadata with tflite_support, which does not import on
        # Python 3.12+ (it needs the removed 'imp' module). The models are written by then.
        produced = work / f"{pt.stem}_saved_model"
        if fmt_ != "tflite" or not any(produced.glob("*.tflite")):
            raise
        info["metadata"] = f"tflite_support failed ({type(e).__name__}); metadata appended as a zip member instead"
    if fmt_ == "tflite":
        # onnx2tf writes several .tflite files; Ultralytics returns the saved_model folder.
        folder = produced if produced.is_dir() else produced.parent
        want = {"fp32": "_float32.tflite", "fp16": "_float16.tflite", "int8": "_int8.tflite",
                "int8static": "_integer_quant.tflite"}[prec]
        hits = sorted(folder.glob(f"*{want}"))
        if not hits:
            raise RuntimeError(f"No *{want} in {folder}: {sorted(p.name for p in folder.glob('*.tflite'))}")
        produced = hits[0]
        info["tflite_file"] = produced.name
        if prec == "int8static":
            info["note"] = "onnx2tf full integer quantisation (per-tensor), float32 input and output"
        if prec == "int8":
            info["note"] = ("Ultralytics 8.3.21 renames onnx2tf's *_dynamic_range_quant.tflite to *_int8.tflite: "
                            "int8 weights, float activations (dynamic-range quantisation), calibration data unused")
        if not _is_zip_with_members(produced):
            meta = yaml.safe_load((folder / "metadata.yaml").read_text(encoding="utf-8"))
            add_tflite_metadata(produced, meta)
    if dest.exists():
        shutil.rmtree(dest) if dest.is_dir() else dest.unlink()
    shutil.move(str(produced), dest)
    return info


def available(fmt_: str) -> str | None:
    """None if this machine can build the format, else why not."""
    if fmt_ == "engine":
        try:
            import torch

            if not torch.cuda.is_available():
                return "TensorRT needs an NVIDIA GPU; build it on the Jetson (configs/export/p3_export_jetson.yaml)"
        except ImportError:
            return "torch not installed"
        return None
    if fmt_ == "openvino" and platform.machine().lower() not in ("x86_64", "amd64"):
        return "OpenVINO targets x86 CPUs"
    return None


# ----------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.export", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="export YAML, e.g. configs/export/p3_export_yolo11s.yaml")
    ap.add_argument("--only", nargs="+", help="targets to build, e.g. onnx_int8 tflite_fp16")
    ap.add_argument("--skip-existing", action="store_true",
                    help="do not rebuild an export already on disk; record its size and sha256 only")
    args = ap.parse_args(argv)
    cfg_path = Path(args.config)
    with (cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    import ultralytics

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}.")
    set_seed(int(cfg["seed"]))
    pt = REPO_ROOT / cfg["weights"]
    if not pt.exists():
        sys.exit(f"Weights not found: {pt}. {cfg.get('missing_hint', '')}")
    out_dir = REPO_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.get("stem", pt.stem)
    cfg["stem"] = stem
    targets = [t for t in cfg["targets"] if not args.only or t in args.only]
    for t in targets:
        f, p = t.split("_")
        if p not in TARGETS.get(f, []):
            sys.exit(f"Unknown target {t}; choose from {[f'{a}_{b}' for a, bs in TARGETS.items() for b in bs]}")

    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    command = "python -m ppe.export " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    work_root = Path(tempfile.mkdtemp(prefix="ppe_export_"))
    cal = cfg["calibration"]
    images = calibration_images(cfg) if any("int8" in t for t in targets) else []
    calib = calibration_yaml(cfg, images, work_root / "calib") if images else None
    # onnx2tf holds every calibration image in memory as float32 (300 at 640 px ran out of 15 GB here),
    # so TFLite may use the first n_images_tflite of the same seeded draw.
    images_tfl = calibration_images(cfg, cal.get("n_images_tflite")) if images else []
    calib_tfl = calibration_yaml(cfg, images_tfl, work_root / "calib_tflite") if images else None

    def cal_note_for(imgs):
        return (f"INT8 calibration: {len(imgs)} images drawn with seed {cal['seed']} from {cal['data']} "
                f"split {cal['split']} (never test)")

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": cfg["phase"],
            "model": cfg["model"], "weights": f"{cfg['weights']}@{sha256_path(pt)[:12]}", "device": cfg.get("device", "cpu"),
            "dataset": cfg["calibration"]["dataset"], "condition": "export", "class": "all", "seed": cfg["seed"],
            "command": command}
    rows, skipped = [], []
    for t in targets:
        f, p = t.split("_")
        why = available(f)
        if why:
            print(f"[{t}] skipped: {why}", flush=True)
            skipped.append((t, why))
            continue
        dest = target_path(out_dir, stem, f, p)
        print(f"[{t}] exporting to {dest.relative_to(REPO_ROOT)} ...", flush=True)
        t0 = time.perf_counter()
        if args.skip_existing and dest.exists():
            info, cal_note = {"rebuilt": "no, existing file recorded"}, ""
            if "int8" in p:
                cal_note = cal_note_for(images_tfl if f == "tflite" else images) + " (as configured; file not rebuilt)"
        else:
            try:
                tfl = f == "tflite"
                info = export_target(pt, f, p, cfg, images_tfl if tfl else images, calib_tfl if tfl else calib,
                                     dest, work_root / t)
                cal_note = cal_note_for(images_tfl if tfl else images)
            except (Exception, SystemExit) as e:   # one failing exporter must not stop the others
                print(f"[{t}] FAILED: {e}", flush=True)
                skipped.append((t, f"export failed: {e}"))
                continue
        secs = time.perf_counter() - t0
        digest = sha256_path(dest)
        manifest[t] = {"path": dest.relative_to(REPO_ROOT).as_posix(), "sha256": digest, "size_mb": round(size_mb(dest), 3),
                       "source": base["weights"], "run_id": run_id, "calibration": cal_note if "int8" in p else "",
                       **info}
        notes = f"{dest.relative_to(REPO_ROOT).as_posix()}@{digest[:12]}; export {secs:.0f} s"
        if "int8" in p and not (f == "tflite" and p == "int8"):
            notes += f"; {cal_note}"
        if info:
            notes += "; " + "; ".join(f"{k}: {v}" for k, v in info.items())
        row = {**base, "precision_mode": t, "split": cal["split"] if "int8" in p else "",
               "metric": "model_size_mb", "value": fmt(size_mb(dest)), "n": 1, "notes": notes}
        rows.append(row)
        append_rows([row])                 # written as each export finishes: a later crash keeps it
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[{t}] ok, {size_mb(dest):.1f} MB, {secs:.0f} s", flush=True)
    shutil.rmtree(work_root, ignore_errors=True)
    if rows:
        print(f"\nAppended {len(rows)} rows to results/results.csv as run {run_id}; manifest {manifest_path.relative_to(REPO_ROOT)}")
    for t, why in skipped:
        print(f"  not built: {t}: {why}")
    return 0 if rows or not targets else 1


if __name__ == "__main__":
    raise SystemExit(main())
