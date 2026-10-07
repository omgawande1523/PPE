"""Site shift (Phase 2): synthetic conditions on the test split and external sites.

One command produces the whole shift table and appends it to results/results.csv:

    python -m ppe.shift --config configs/shift/p2_shift_yolo11s.yaml

It evaluates, with the same weights and the Phase 1 operating confidence:
  * the Construction-PPE test split (golden set v1), clean, as the reference;
  * the same 141 images under six synthetic conditions at three severities
    (low light, dust/haze, motion blur, JPEG compression, rain, downscaling
    for a distant camera), each image corrupted with its own seeded draw;
  * every external site that has been converted and is on disk (SH17, CHV,
    own-site footage), on the classes that site labels.
For each cell: per-class mAP@0.5 and mAP@0.5:0.95 (Ultralytics protocol),
per-class precision/recall/F1 at the operating confidence, person-level
violation recall, and the drop against the clean reference with a 95%
bootstrap interval (paired over images for the synthetic conditions).

Other commands:
    python -m ppe.shift --convert sh17 --src path/to/sh17           # SH17 -> datasets/sh17
    python -m ppe.shift --convert chv --src path/to/chv             # CHV  -> datasets/chv
    python -m ppe.shift --table RUN_ID                              # markdown shift table from results.csv
    python -m ppe.shift --preview --config ... --out runs/preview   # one example image per cell, no model
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import sys
import xml.etree.ElementTree as ET
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_file, sha256_path
from ppe.eval import (FIELDS, RESULTS_CSV, VIOLATION_CLASSES, FIGURES_DIR, ap50_from_curve, append_rows, fmt,
                      git_commit, label_for, match_per_class, person_level, predict_split, prf_at)
from ppe.infer import set_seed

CONDITIONS = ["low_light", "haze", "motion_blur", "jpeg", "rain", "downscale"]
SEVERITIES = 3


# ============================================================================= corruptions
# Each takes a BGR uint8 image, the parameter dict of its condition, the
# severity index (0, 1, 2) and a numpy Generator, and returns a BGR uint8 image.

def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)


def low_light(img, p, s, rng):
    x = _to_float(img) ** float(p["gamma"][s]) * float(p["gain"][s])
    x = x + rng.normal(0.0, float(p["noise_sigma"][s]), x.shape).astype(np.float32)
    return _to_uint8(x)


def haze(img, p, s, rng):
    h = img.shape[0]
    depth = 1.0 - 0.5 * (np.arange(h, dtype=np.float32) / max(h - 1, 1))      # 1 at the top, 0.5 at the bottom
    t = np.exp(-float(p["beta"][s]) * depth)[:, None, None]
    airlight = np.array(p["airlight_bgr"], dtype=np.float32)[None, None, :]
    return _to_uint8(_to_float(img) * t + airlight * (1.0 - t))


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    import cv2

    length = max(int(length), 1)
    size = length if length % 2 else length + 1
    k = np.zeros((size, size), np.float32)
    k[size // 2, (size - length) // 2:(size - length) // 2 + length] = 1.0
    rot = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (size, size))
    return k / max(k.sum(), 1e-6)


def motion_blur(img, p, s, rng):
    import cv2

    length = round(float(p["length_frac"][s]) * max(img.shape[:2]))
    return cv2.filter2D(img, -1, _line_kernel(length, float(rng.uniform(0, 180))), borderType=cv2.BORDER_REFLECT)


def jpeg_bytes(img, p, s) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(p["quality"][s])])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def jpeg(img, p, s, rng):
    import cv2

    return cv2.imdecode(np.frombuffer(jpeg_bytes(img, p, s), np.uint8), cv2.IMREAD_COLOR)


def rain(img, p, s, rng):
    import cv2

    h, w = img.shape[:2]
    n = int(float(p["density"][s]) * h * w)
    length = float(p["length_frac"][s]) * max(h, w)
    base = rng.uniform(-20, 20)                                      # wind direction for this image, degrees
    mask = np.zeros((h, w), np.float32)
    xs, ys = rng.uniform(0, w, n), rng.uniform(0, h, n)
    angs = np.deg2rad(base + rng.normal(0, 4, n))
    lens = length * rng.uniform(0.6, 1.0, n)
    for x, y, a, ln in zip(xs, ys, angs, lens):
        x2, y2 = x + ln * np.sin(a), y + ln * np.cos(a)
        cv2.line(mask, (int(x), int(y)), (int(x2), int(y2)), 1.0, 1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None] * float(p["alpha"][s])
    x = cv2.GaussianBlur(_to_float(img), (3, 3), 0) * float(p["darken"][s])
    x = x * (1 - mask) + 0.85 * mask
    return _to_uint8(x)


def downscale(img, p, s, rng):
    import cv2

    h, w = img.shape[:2]
    sc = float(p["scale"][s])
    nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full_like(img, 114)                                     # Ultralytics letterbox grey
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def downscale_labels(rows: list[list[str]], h: int, w: int, p: dict, s: int) -> list[list[str]]:
    """Shrink YOLO boxes exactly as downscale() shrinks the content (same integer offsets)."""
    sc = float(p["scale"][s])
    nh, nw = max(1, round(h * sc)), max(1, round(w * sc))
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    out = []
    for r in rows:
        c, cx, cy, bw, bh = r[0], *map(float, r[1:5])
        out.append([c, f"{(x0 + cx * nw) / w:.6f}", f"{(y0 + cy * nh) / h:.6f}",
                    f"{bw * nw / w:.6f}", f"{bh * nh / h:.6f}"])
    return out


CORRUPT = {"low_light": low_light, "haze": haze, "motion_blur": motion_blur, "jpeg": jpeg, "rain": rain,
           "downscale": downscale}


def image_rng(seed: int, cond: str, s: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, CONDITIONS.index(cond), s, zlib.crc32(name.encode("utf-8"))])


def make_cell(images: list[Path], cond: str, s: int, params: dict, seed: int, out_root: Path) -> Path:
    """Write the corrupted copy of a split: out_root/images/test and out_root/labels/test.

    Corrupted images are written losslessly (PNG), except the jpeg condition,
    whose encoded bytes are the corruption and are written as they are.
    """
    import cv2

    img_dir, lab_dir = out_root / "images" / "test", out_root / "labels" / "test"
    if out_root.exists():
        shutil.rmtree(out_root)
    img_dir.mkdir(parents=True)
    lab_dir.mkdir(parents=True)
    for src in images:
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"Cannot read {src}")
        rng = image_rng(seed, cond, s, src.name)
        if cond == "jpeg":
            (img_dir / f"{src.stem}.jpg").write_bytes(jpeg_bytes(img, params, s))
        else:
            cv2.imwrite(str(img_dir / f"{src.stem}.png"), CORRUPT[cond](img, params, s, rng))
        lab_src = label_for(src)
        rows = [ln.split() for ln in lab_src.read_text(encoding="utf-8").splitlines() if ln.strip()] \
            if lab_src.exists() else []
        if cond == "downscale":
            rows = downscale_labels(rows, img.shape[0], img.shape[1], params, s)
        (lab_dir / f"{src.stem}.txt").write_text("".join(" ".join(r) + "\n" for r in rows), encoding="utf-8")
    return out_root


# ============================================================================= external-set converters

def _target_index() -> dict[str, int]:
    with (REPO_ROOT / "configs" / "data" / "construction_ppe.yaml").open(encoding="utf-8") as fh:
        names = yaml.safe_load(fh)["names"]
    return {v.lower(): int(k) for k, v in names.items()}


# Published class order of SH17 (sh17.yaml in github.com/ahmadmughees/SH17dataset), used only
# when the downloaded copy carries no names file of its own.
SH17_NAMES = ["person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot", "tool", "glasses", "gloves",
              "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest"]

# source class name (lower case) -> our class name. Anything not listed is dropped.
MAPS = {
    "sh17": {"person": "person", "helmet": "helmet", "safety-vest": "vest", "gloves": "gloves", "shoes": "boots"},
    "chv": {"person": "person", "vest": "vest", "helmet": "helmet",
            "blue": "helmet", "red": "helmet", "white": "helmet", "yellow": "helmet"},
}
NOTES = {
    "sh17": ("glasses excluded (brief); 'shoes' in SH17 is any footwear, our 'boots' is safety boots; "
             "no no_* classes, so violation recall cannot be measured"),
    "chv": "helmet colours merged into helmet; no gloves, boots, goggles or no_* classes",
}


def _find_names(src: Path) -> list[str] | None:
    for pat in ("*.yaml", "*.yml"):
        for f in sorted(src.rglob(pat)):
            try:
                d = yaml.safe_load(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(d, dict) and "names" in d:
                n = d["names"]
                n = [n[k] for k in sorted(n)] if isinstance(n, dict) else list(n)
                if n and all(isinstance(x, str) for x in n):
                    print(f"Class names from {f}")
                    return [x.strip() for x in n]
    for pat in ("classes.txt", "*.names"):
        for f in sorted(src.rglob(pat)):
            n = [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if n:
                print(f"Class names from {f}")
                return n
    return None


def _read_voc(xml_path: Path) -> tuple[list[tuple[str, float, float, float, float]], int, int]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    objs = []
    for o in root.iter("object"):
        bb = o.find("bndbox")
        objs.append((o.findtext("name").strip(), *(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return objs, w, h


def convert(name: str, src: Path, list_file: Path | None, names_arg: str | None, use_all: bool) -> None:
    """Copy an external set into datasets/<name>/{images,labels}/test with our class indices."""
    from PIL import Image

    src = src.resolve()
    if not src.is_dir():
        sys.exit(f"Not a directory: {src}")
    mapping = MAPS[name]
    target = _target_index()
    src_names = names_arg.split(",") if names_arg else _find_names(src)
    if src_names is None and name == "sh17":
        src_names = SH17_NAMES
        print("No names file found; using the published SH17 class order.")
    images = {p.stem: p for p in sorted(src.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES}
    if not images:
        sys.exit(f"No images under {src}")
    if list_file is None and name == "sh17" and not use_all:
        cands = sorted(src.rglob("val_files.txt"))
        if not cands:
            sys.exit("SH17's val_files.txt not found; pass --list FILE, or --all to convert every image.")
        list_file = cands[0]
    if list_file is not None:
        wanted = [Path(ln.strip().replace("\\", "/")).stem for ln in list_file.read_text(encoding="utf-8").splitlines()
                  if ln.strip()]
        missing = [s for s in wanted if s not in images]
        if missing:
            sys.exit(f"{len(missing)} images in {list_file.name} are not under {src}, e.g. {missing[:3]}")
        stems = wanted
    else:
        stems = sorted(images)

    dest = REPO_ROOT / "datasets" / name
    if dest.exists():
        shutil.rmtree(dest)
    img_out, lab_out = dest / "images" / "test", dest / "labels" / "test"
    img_out.mkdir(parents=True)
    lab_out.mkdir(parents=True)
    kept: dict[str, int] = {}
    dropped: dict[str, int] = {}
    no_label = 0
    manifest = []
    for stem in stems:
        img = images[stem]
        txt = next((c for c in (label_for(img) if "images" in img.parts else None, img.with_suffix(".txt"))
                    if c is not None and c.exists()), None)
        xml = img.with_suffix(".xml") if img.with_suffix(".xml").exists() else None
        if txt is None and xml is None:
            cands = [p for p in src.rglob(f"{stem}.txt")] + [p for p in src.rglob(f"{stem}.xml")]
            txt = next((p for p in cands if p.suffix == ".txt"), None)
            xml = next((p for p in cands if p.suffix == ".xml"), None) if txt is None else None
        if txt is None and xml is None:
            no_label += 1
            continue
        out_rows = []
        if txt is not None:
            if src_names is None:
                sys.exit(f"No class names found under {src}; pass --names a,b,c in the source's index order.")
            for ln in txt.read_text(encoding="utf-8").splitlines():
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cname = src_names[int(float(parts[0]))].lower()
                if cname in mapping:
                    ours = mapping[cname]
                    out_rows.append(f"{target[ours]} {' '.join(parts[1:5])}")
                    kept[ours] = kept.get(ours, 0) + 1
                else:
                    dropped[cname] = dropped.get(cname, 0) + 1
            label_src = txt
        else:
            objs, w, h = _read_voc(xml)
            if w <= 0 or h <= 0:
                with Image.open(img) as im:
                    w, h = im.size
            for cname, x1, y1, x2, y2 in objs:
                cname = cname.lower()
                if cname in mapping:
                    ours = mapping[cname]
                    out_rows.append(f"{target[ours]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                                    f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
                    kept[ours] = kept.get(ours, 0) + 1
                else:
                    dropped[cname] = dropped.get(cname, 0) + 1
            label_src = xml
        out_img = img_out / img.name
        try:
            os.link(img, out_img)
        except OSError:
            shutil.copy2(img, out_img)
        (lab_out / f"{img.stem}.txt").write_text("".join(r + "\n" for r in out_rows), encoding="utf-8")
        manifest.append({"image": img.name, "image_sha256": sha256_file(img),
                         "source_label": label_src.relative_to(src).as_posix(), "n_boxes": len(out_rows)})
    if not manifest:
        sys.exit("Nothing converted.")
    with (dest / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    summary = {"dataset": name, "images": len(manifest), "images_without_labels_skipped": no_label,
               "list_file": list_file.name if list_file else None, "source_names": src_names,
               "mapping": mapping, "kept_instances": kept, "dropped_instances": dropped, "notes": NOTES[name]}
    (dest / "conversion.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {len(manifest)} images to {dest.relative_to(REPO_ROOT)}")


# ============================================================================= evaluation

def runtime_yaml(data_cfg: Path | None, root: Path, out_dir: Path, names: dict) -> Path:
    """Data YAML for Ultralytics with train/val/test all pointing at root/images/test."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": str(root.resolve()), "train": "images/test", "val": "images/test",
                        "test": "images/test", "names": names}, fh, sort_keys=False)
    return out


def list_images(root: Path) -> list[Path]:
    d = root / "images" / "test"
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else []


def run_cell(model, data_yaml: Path, images: list[Path], cfg: dict, work: Path, tag: str) -> dict:
    """Ultralytics val (per-class AP) and the deployed predict path (boxes kept for thresholding)."""
    seed = int(cfg["seed"])
    set_seed(seed)
    m = model.val(data=str(data_yaml), split="test", imgsz=int(cfg["imgsz"]), batch=int(cfg["batch"]),
                  conf=float(cfg["conf_floor"]), iou=float(cfg["nms_iou"]), device=cfg["device"],
                  half=cfg["precision_mode"] == "fp16", plots=False, verbose=False,
                  project=str(work), name=f"ul_{tag}", exist_ok=True)
    ul = {"all": (float(m.box.map50), float(m.box.map))}
    for i, c in enumerate(m.box.ap_class_index):
        _, _, ap50, ap = m.box.class_result(i)
        ul[int(c)] = (float(ap50), float(ap))
    set_seed(seed)
    preds = predict_split(model, images, cfg)
    return {"ul": ul, "preds": preds}


def per_image_counts(preds: list[dict], nc: int, t: float, iou_thr: float, items: list[str],
                     names: dict[int, str]) -> dict[str, np.ndarray]:
    """Per image: TP and GT counts per class at t, and person-level violator hits, for bootstrapping."""
    tp = np.zeros((len(preds), nc))
    gt = np.zeros((len(preds), nc))
    hit = np.zeros(len(preds))
    viol = np.zeros(len(preds))
    for i, fr in enumerate(preds):
        per = match_per_class([fr], nc, iou_thr)
        for c in range(nc):
            tp[i, c] = prf_at(per[c], t)[0]
            gt[i, c] = per[c]["n_gt"]
        pl = person_level([fr], names, t, items, iou_thr)
        hit[i], viol[i] = pl["hit"]["any"], pl["gt_viol"]["any"]
    return {"tp": tp, "gt": gt, "hit": hit, "viol": viol, "name": [fr["image"] for fr in preds]}


def _ratio(num, den):
    return num / den if den else float("nan")


def _stats(cnt: dict, idx: np.ndarray, classes: list[int], viol_cls: list[int]) -> dict[str, float]:
    tp, gt = cnt["tp"][idx].sum(0), cnt["gt"][idx].sum(0)
    out = {f"c{c}": _ratio(tp[c], gt[c]) for c in classes}
    out["all"] = _ratio(tp[classes].sum(), gt[classes].sum())
    vr = [_ratio(tp[c], gt[c]) for c in viol_cls if gt[c] > 0]
    out["viol_macro"] = float(np.mean(vr)) if vr else float("nan")
    out["person_any"] = _ratio(cnt["hit"][idx].sum(), cnt["viol"][idx].sum())
    return out


def bootstrap_delta(ref: dict, cond: dict, classes: list[int], viol_cls: list[int], n_boot: int, seed: int,
                    paired: bool) -> dict[str, tuple[float, float]]:
    """95% percentile interval of recall(cond) - recall(ref).

    Paired: the same resampled image indices for both (same images, different condition).
    Unpaired: independent resamples of each set (different images).
    """
    rng = np.random.default_rng(seed)
    n_r, n_c = len(ref["tp"]), len(cond["tp"])
    if paired and ref["name"] != cond["name"]:
        raise ValueError("paired bootstrap needs the same images in the same order")
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        ir = rng.integers(0, n_r, n_r)
        ic = ir if paired else rng.integers(0, n_c, n_c)
        a, b = _stats(ref, ir, classes, viol_cls), _stats(cond, ic, classes, viol_cls)
        for k in a:
            d = b[k] - a[k]
            if np.isfinite(d):
                samples.setdefault(k, []).append(d)
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()
            if len(v) >= 0.9 * n_boot}


def check_operating_conf(cfg: dict) -> str:
    t = float(cfg["operating_conf"])
    src = cfg.get("operating_conf_source")
    if not src or not RESULTS_CSV.exists():
        return f"operating conf {t:.2f} from config"
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == src and r["metric"] == "operating_conf"]
    if not rows:
        return f"operating conf {t:.2f} from config (run {src} not in results.csv)"
    if abs(float(rows[0]["value"]) - t) > 1e-9:
        sys.exit(f"Config operating_conf {t} differs from {src}'s {rows[0]['value']} in results.csv.")
    return f"operating conf {t:.2f}, chosen on val in run {src}"


def check_reference(rows: list[dict], p1_run: str | None) -> None:
    """Compare the clean rows with the same rows of the Phase 1 run; note the outcome on every clean row."""
    if not p1_run or not RESULTS_CSV.exists():
        return
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        p1 = {(r["class"], r["metric"]): r["value"] for r in csv.DictReader(fh)
              if r["run_id"] == p1_run and r["split"] == "test"}
    if not p1:
        return
    clean = [r for r in rows if r["condition"] == "clean"]
    shared = [r for r in clean if (r["class"], r["metric"]) in p1]
    same = sum(p1[(r["class"], r["metric"])] == r["value"] for r in shared)
    msg = f"clean re-measurement: {same} of {len(shared)} rows shared with {p1_run} are identical"
    print(msg, flush=True)
    for r in clean:
        r["notes"] += f"; {msg}"


def evaluate(cfg: dict, command: str, only: list[str] | None, collect: dict | None = None) -> list[dict]:
    """All cells of one model. If collect is a dict, per-image counts of each synthetic cell land in it
    (key = condition tag, "clean" for the reference), for paired comparisons between models (ppe.quant)."""
    from ultralytics import YOLO
    import ultralytics

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}. "
                 f"pip install -r requirements.txt")
    seed = int(cfg["seed"])
    weights = REPO_ROOT / cfg["weights"]
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")
    t = float(cfg["operating_conf"])
    t_note = check_operating_conf(cfg)
    iou_thr = float(cfg["match_iou"])
    items = list(cfg["required_items"])
    n_boot = int(cfg.get("bootstrap", 0))

    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    work = REPO_ROOT / "runs" / "shift" / run_id

    with (REPO_ROOT / cfg["data"]).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    ref_root = REPO_ROOT / data["path"]
    ref_split_dir = ref_root / data[cfg["split"]]
    if not ref_split_dir.is_dir():
        sys.exit(f"Dataset not found at {ref_split_dir}. Run: python -m ppe.data --download construction-ppe")
    from ppe.golden import verify

    verify(REPO_ROOT / cfg["golden_manifest"])
    golden = Path(cfg["golden_manifest"]).stem

    model = YOLO(str(weights), task="detect")
    names = {int(k): v for k, v in model.names.items()}
    nc = len(names)
    by_name = {v: k for k, v in names.items()}
    viol_cls = [by_name[c] for c in VIOLATION_CLASSES]
    ds_names = data["names"]

    base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": cfg["phase"],
            "model": cfg["model"], "weights": f"{cfg['weights']}@{sha256_path(weights)[:12]}",
            "precision_mode": cfg["precision_mode"], "device": cfg["device"], "seed": seed, "command": command}
    rows: list[dict] = []
    op = f"deployed predict path, {t_note}, match IoU {iou_thr}"

    def add(dataset, split, cond, cls, metric, value, n, notes):
        rows.append({**base, "dataset": dataset, "split": split, "condition": cond, "class": cls, "metric": metric,
                     "value": fmt(value), "n": n, "notes": notes})

    def cell_rows(dataset, split, cond, res, classes, ctx, person=True):
        """Absolute metrics of one cell on the given class indices."""
        preds = res["preds"]
        per = match_per_class(preds, nc, iou_thr)
        ul_note = f"{len(preds)} images; {ctx}; Ultralytics val, conf {cfg['conf_floor']}, NMS IoU {cfg['nms_iou']}"
        tot = sum(per[c]["n_gt"] for c in classes)
        ul_classes = [c for c in classes if c in res["ul"]]
        add(dataset, split, cond, "all", "map50", float(np.mean([res["ul"][c][0] for c in ul_classes])), tot,
            f"{ul_note}; mean over {len(ul_classes)} classes with labels")
        add(dataset, split, cond, "all", "map50_95", float(np.mean([res["ul"][c][1] for c in ul_classes])), tot,
            f"{ul_note}; mean over {len(ul_classes)} classes with labels")
        tp_all = fp_all = fn_all = 0
        f1s = []
        for c in classes:
            if per[c]["n_gt"] == 0:
                continue
            tp, fp, fn, p, r, f1 = prf_at(per[c], t)
            tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
            f1s.append(f1)
            note = f"{op}; {ctx}; tp={tp} fp={fp} fn={fn}"
            if c in res["ul"]:
                add(dataset, split, cond, names[c], "map50", res["ul"][c][0], per[c]["n_gt"], ul_note)
                add(dataset, split, cond, names[c], "map50_95", res["ul"][c][1], per[c]["n_gt"], ul_note)
            add(dataset, split, cond, names[c], "precision", p, per[c]["n_gt"], note)
            add(dataset, split, cond, names[c], "recall", r, per[c]["n_gt"], note)
            add(dataset, split, cond, names[c], "f1", f1, per[c]["n_gt"], note)
            add(dataset, split, cond, names[c], "ap50_predict", ap50_from_curve(per[c]), per[c]["n_gt"],
                f"{ctx}; AP@0.5 from the predict path")
        n_all = tp_all + fn_all
        add(dataset, split, cond, "all", "precision", _ratio(tp_all, tp_all + fp_all), n_all, f"{op}; {ctx}; micro")
        add(dataset, split, cond, "all", "recall", _ratio(tp_all, n_all), n_all, f"{op}; {ctx}; micro")
        add(dataset, split, cond, "all", "macro_f1", float(np.mean(f1s)) if f1s else float("nan"), n_all, f"{op}; {ctx}")
        vc = [c for c in viol_cls if c in classes and per[c]["n_gt"] > 0]
        if vc:
            add(dataset, split, cond, "violation_classes", "macro_recall",
                float(np.mean([prf_at(per[c], t)[4] for c in vc])), sum(per[c]["n_gt"] for c in vc),
                f"{op}; {ctx}; unweighted mean of {', '.join(names[c] for c in vc)} box recall")
        if person and vc:
            pl = person_level(preds, names, t, items, iou_thr)
            pnote = (f"{op}; {ctx}; per labelled person; {pl['n_gt_persons']} labelled persons, "
                     f"{pl['n_gt_unmatched']} not detected")
            for key in ["any"] + pl["items"]:
                from ppe.associate import ITEM_CLASSES
                cls = "person_any_violation" if key == "any" else f"person_{ITEM_CLASSES[key]}"
                n_v = pl["gt_viol"][key]
                add(dataset, split, cond, cls, "violation_recall", _ratio(pl["hit"][key], n_v), n_v,
                    f"{pnote}; flagged {pl['hit'][key]} of {n_v} labelled violators")
                n_a = pl["alerts"][key]
                add(dataset, split, cond, cls, "false_alert_rate", _ratio(n_a - pl["true_alerts"][key], n_a), n_a,
                    f"{pnote}; share of {n_a} predicted violation flags that match no labelled violation")
            add(dataset, split, cond, "person_any_violation", "nonviolator_flag_rate",
                _ratio(pl["nonviol_flagged"], pl["nonviol_gt"]), pl["nonviol_gt"],
                f"{pnote}; labelled non-violators flagged for any item")
        return per

    def delta_rows(dataset, split, cond, ref_res, ref_per, res, per, ref_cnt, cnt, classes, paired, ctx):
        """Cell minus clean reference, on the given classes, with 95% bootstrap intervals."""
        kind = ("paired over the same images" if paired else
                "different images: independent resamples of each set")
        ci = bootstrap_delta(ref_cnt, cnt, classes, viol_cls, n_boot, seed, paired) if n_boot else {}
        ref_desc = f"reference: construction-ppe test, clean, same run ({golden})"
        ci_note = f"95% percentile interval, {n_boot} bootstrap resamples, {kind}"

        def put(cls, metric, value, n, key, extra=""):
            add(dataset, split, cond, cls, metric, value, n, f"{ctx}; {ref_desc}; negative = worse{extra}")
            if key in ci:
                add(dataset, split, cond, cls, f"{metric}_ci95_low", ci[key][0], n, f"{ctx}; {ci_note}")
                add(dataset, split, cond, cls, f"{metric}_ci95_high", ci[key][1], n, f"{ctx}; {ci_note}")

        for c in classes:
            if per[c]["n_gt"] == 0 or ref_per[c]["n_gt"] == 0:
                continue
            r0, r1 = prf_at(ref_per[c], t)[4], prf_at(per[c], t)[4]
            floor = "; clean recall is 0, so a drop cannot be observed (floor)" if r0 == 0 else ""
            put(names[c], "recall_delta", r1 - r0, per[c]["n_gt"], f"c{c}",
                f"; clean {r0:.4f} (n={ref_per[c]['n_gt']}), here {r1:.4f} (n={per[c]['n_gt']}){floor}")
            if c in res["ul"] and c in ref_res["ul"]:
                put(names[c], "map50_delta", res["ul"][c][0] - ref_res["ul"][c][0], per[c]["n_gt"], "",
                    f"; clean {ref_res['ul'][c][0]:.4f}, here {res['ul'][c][0]:.4f}")
        a0, a1 = _stats(ref_cnt, np.arange(len(ref_cnt["tp"])), classes, viol_cls), \
            _stats(cnt, np.arange(len(cnt["tp"])), classes, viol_cls)
        n_all = int(cnt["gt"][:, classes].sum())
        put("all", "recall_delta", a1["all"] - a0["all"], n_all, "all", "; micro over the evaluated classes")
        ulc = [c for c in classes if c in res["ul"] and c in ref_res["ul"]]
        put("all", "map50_delta", float(np.mean([res["ul"][c][0] for c in ulc]) -
                                        np.mean([ref_res["ul"][c][0] for c in ulc])), n_all, "",
            f"; mean AP@0.5 over the {len(ulc)} evaluated classes")
        if np.isfinite(a1["viol_macro"]) and np.isfinite(a0["viol_macro"]):
            put("violation_classes", "macro_recall_delta", a1["viol_macro"] - a0["viol_macro"],
                int(cnt["gt"][:, viol_cls].sum()), "viol_macro")
        if np.isfinite(a1["person_any"]) and np.isfinite(a0["person_any"]):
            put("person_any_violation", "violation_recall_delta", a1["person_any"] - a0["person_any"],
                int(cnt["viol"].sum()), "person_any")

    all_cls = list(range(nc))
    names_yaml = {int(k): v for k, v in ds_names.items()}

    # ---- reference: clean test split, from the original files
    print(f"[clean] {len(list_images(ref_root))} images ...", flush=True)
    ref_images = list_images(ref_root)
    ref_res = run_cell(model, runtime_yaml(None, ref_root, work / "clean", names_yaml), ref_images, cfg, work, "clean")
    ref_ctx = f"construction-ppe test = {golden} (verified), clean"
    ref_per = cell_rows(cfg["dataset"], cfg["split"], "clean", ref_res, all_cls, ref_ctx)
    check_reference(rows, cfg.get("operating_conf_source"))
    ref_cnt = per_image_counts(ref_res["preds"], nc, t, iou_thr, items, names)
    if collect is not None:
        collect["clean"] = ref_cnt

    # ---- synthetic conditions
    for cond in CONDITIONS:
        if only and cond not in only:
            continue
        params = cfg["corruptions"][cond]
        for s in range(SEVERITIES):
            tag = f"{cond}_s{s + 1}"
            print(f"[{tag}] corrupting and evaluating ...", flush=True)
            root = make_cell(ref_images, cond, s, params, seed, work / "data" / tag)
            res = run_cell(model, runtime_yaml(None, root, work / tag, names_yaml), list_images(root), cfg, work, tag)
            pstr = ", ".join(f"{k}={v[s]}" if isinstance(v, list) and not isinstance(v[0], list) and len(v) == 3
                             else f"{k}={v}" for k, v in params.items())
            ctx = f"{golden} images, {cond} severity {s + 1} ({pstr}), per-image seed ({seed}, {cond}, {s}, crc32(name))"
            if cond == "downscale":
                ctx += "; labels shrunk with the content"
            per = cell_rows(cfg["dataset"], cfg["split"], tag, res, all_cls, ctx)
            cnt = per_image_counts(res["preds"], nc, t, iou_thr, items, names)
            cnt["name"] = ref_cnt["name"]      # same images; corrupted copies may be .png
            if collect is not None:
                collect[tag] = cnt
            delta_rows(cfg["dataset"], cfg["split"], tag, ref_res, ref_per, res, per, ref_cnt, cnt, all_cls, True, ctx)

    # ---- external sites
    for ds, spec in (cfg.get("external") or {}).items():
        if only and ds not in only:
            continue
        with (REPO_ROOT / spec["data"]).open(encoding="utf-8") as fh:
            dcfg = yaml.safe_load(fh)
        root = REPO_ROOT / dcfg["path"]
        imgs = list_images(root)
        if not imgs:
            how = (f"python -m ppe.shift --convert {ds} --src PATH" if ds in MAPS
                   else "label own-site frames into datasets/own_site/images|labels/test")
            print(f"[{ds}] not on disk: skipped. To add it: {how}; then re-run this command.", flush=True)
            continue
        print(f"[{ds}] {len(imgs)} images ...", flush=True)
        classes = all_cls if spec["classes"] == "all" else [by_name[c] for c in spec["classes"]]
        conv = root / "conversion.json"
        ctx = f"{ds}, {len(imgs)} images; evaluated classes: {', '.join(names[c] for c in classes)}"
        if conv.exists():
            ctx += f"; {json.loads(conv.read_text(encoding='utf-8'))['notes']}"
        res = run_cell(model, runtime_yaml(None, root, work / ds, names_yaml), imgs, cfg, work, ds)
        per = cell_rows(ds, "test", "clean", res, classes, ctx, person=spec["classes"] == "all")
        cnt = per_image_counts(res["preds"], nc, t, iou_thr, items, names)
        delta_rows(ds, "test", "clean", ref_res, ref_per, res, per, ref_cnt, cnt, classes, False, ctx)

    save_heatmap(rows, run_id)
    return rows


# ============================================================================= reporting

def save_heatmap(rows: list[dict], run_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [r for r in rows if r["metric"] == "recall_delta" and r["dataset"] == "construction-ppe"]
    if not cells:
        return
    conds = list(dict.fromkeys(r["condition"] for r in cells))
    classes = list(dict.fromkeys(r["class"] for r in cells))
    m = np.full((len(classes), len(conds)), np.nan)
    for r in cells:
        m[classes.index(r["class"]), conds.index(r["condition"])] = float(r["value"])
    out = FIGURES_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(conds), 1.0 + 0.4 * len(classes)))
    im = ax.imshow(m, cmap="RdBu", vmin=-0.6, vmax=0.6)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:+.2f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(conds)), conds, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set_title("Box recall change vs clean test (conf 0.28)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out / "shift_recall_delta.png", dpi=130)
    plt.close(fig)


def table(run_id: str) -> int:
    """Print the shift table of one run as markdown, straight from results.csv."""
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == run_id]
    if not rows:
        sys.exit(f"Run {run_id} not in results.csv")
    get = {(r["dataset"], r["condition"], r["class"], r["metric"]): r for r in rows}

    def v(ds, cond, cls, metric, signed=False):
        r = get.get((ds, cond, cls, metric))
        if r is None or r["value"] in ("", "nan"):
            return "n/a"
        x = float(r["value"])
        return f"{x:+.3f}" if signed else f"{x:.3f}"

    def ci(ds, cond, cls, metric):
        lo, hi = get.get((ds, cond, cls, f"{metric}_ci95_low")), get.get((ds, cond, cls, f"{metric}_ci95_high"))
        return f" [{float(lo['value']):+.2f}, {float(hi['value']):+.2f}]" if lo and hi else ""

    cells = list(dict.fromkeys((r["dataset"], r["condition"]) for r in rows))
    print(f"Run `{run_id}`, commit `{rows[0]['git_commit']}`. Regenerate: `{rows[0]['command']}`\n")
    print("### Overall, violation classes and person level (change vs clean test, 95% interval)\n")
    print("| Dataset | Condition | Images | mAP@0.5 | Δ mAP@0.5 | Recall (micro) | Δ recall | "
          "Violation macro recall | Δ | Violators flagged | Δ |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds, cond in cells:
        n_img = get.get((ds, cond, "all", "map50"), {}).get("notes", "").split(" images")[0]
        print(f"| {ds} | {cond} | {n_img} | {v(ds, cond, 'all', 'map50')} | {v(ds, cond, 'all', 'map50_delta', 1)} | "
              f"{v(ds, cond, 'all', 'recall')} | {v(ds, cond, 'all', 'recall_delta', 1)}"
              f"{ci(ds, cond, 'all', 'recall_delta')} | {v(ds, cond, 'violation_classes', 'macro_recall')} | "
              f"{v(ds, cond, 'violation_classes', 'macro_recall_delta', 1)}"
              f"{ci(ds, cond, 'violation_classes', 'macro_recall_delta')} | "
              f"{v(ds, cond, 'person_any_violation', 'violation_recall')} | "
              f"{v(ds, cond, 'person_any_violation', 'violation_recall_delta', 1)}"
              f"{ci(ds, cond, 'person_any_violation', 'violation_recall_delta')} |")
    classes = list(dict.fromkeys(r["class"] for r in rows if r["metric"] == "recall"
                                 and r["class"] not in ("all",)))
    print("\n### Box recall per class at conf 0.28 (Δ vs clean test in brackets)\n")
    print("| Dataset | Condition | " + " | ".join(classes) + " |")
    print("|---|---|" + "---:|" * len(classes))
    for ds, cond in cells:
        vals = []
        for c in classes:
            if (ds, cond, c, "recall") not in get:
                vals.append("")
                continue
            d = "" if cond == "clean" and ds == "construction-ppe" else f" ({v(ds, cond, c, 'recall_delta', 1)})"
            vals.append(v(ds, cond, c, "recall") + d)
        print(f"| {ds} | {cond} | " + " | ".join(vals) + " |")
    print("\n### Violation-class recall change with 95% paired interval\n")
    print("| Condition | " + " | ".join(f"{c} (n)" for c in VIOLATION_CLASSES) + " |")
    print("|---|" + "---|" * len(VIOLATION_CLASSES))
    for ds, cond in cells:
        if ds != "construction-ppe" or cond == "clean":
            continue
        vals = []
        for c in VIOLATION_CLASSES:
            r = get.get((ds, cond, c, "recall_delta"))
            floor = " floor" if r and "floor" in r["notes"] else ""
            vals.append(f"{v(ds, cond, c, 'recall_delta', 1)}{ci(ds, cond, c, 'recall_delta')} ({r['n']}){floor}"
                        if r else "")
        print(f"| {cond} | " + " | ".join(vals) + " |")
    return 0


def preview(cfg: dict, out: Path) -> int:
    """Write one image per condition and severity side by side, for checking the corruptions by eye."""
    import cv2

    with (REPO_ROOT / cfg["data"]).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    imgs = list_images(REPO_ROOT / data["path"])[:1]
    if not imgs:
        sys.exit("Dataset not found. Run: python -m ppe.data --download construction-ppe")
    src = cv2.imread(str(imgs[0]))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cond in CONDITIONS:
        tiles = [src] + [CORRUPT[cond](src, cfg["corruptions"][cond], s, image_rng(int(cfg["seed"]), cond, s,
                                                                                 imgs[0].name))
                         for s in range(SEVERITIES)]
        rows.append(np.concatenate([cv2.resize(x, (320, 320)) for x in tiles], axis=1))
    cv2.imwrite(str(out / "corruptions.jpg"), np.concatenate(rows, axis=0))
    print(f"Wrote {out / 'corruptions.jpg'} (rows: {', '.join(CONDITIONS)}; columns: clean, severity 1-3)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.shift", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="shift YAML, e.g. configs/shift/p2_shift_yolo11s.yaml")
    ap.add_argument("--only", nargs="+", help="evaluate only these conditions/sites (clean is always included)")
    ap.add_argument("--convert", choices=sorted(MAPS), help="convert an external dataset into datasets/<name>")
    ap.add_argument("--src", type=Path, help="unzipped external dataset (with --convert)")
    ap.add_argument("--list", type=Path, help="file listing the images to convert (default: SH17 val_files.txt)")
    ap.add_argument("--all", action="store_true", help="convert every image, not only the evaluation list")
    ap.add_argument("--names", help="source class names in index order, comma separated, if the source has no names file")
    ap.add_argument("--table", metavar="RUN_ID", help="print the shift table of a run from results.csv")
    ap.add_argument("--preview", action="store_true", help="write a grid of one corrupted example per cell")
    ap.add_argument("--out", type=Path, default=Path("runs/shift/preview"))
    args = ap.parse_args(argv)
    if args.table:
        return table(args.table)
    if args.convert:
        if not args.src:
            ap.error("--convert needs --src")
        convert(args.convert, args.src, args.list, args.names, args.all)
        return 0
    if not args.config:
        ap.error("--config is required")
    cfg_path = Path(args.config)
    with (cfg_path if cfg_path.is_absolute() else REPO_ROOT / cfg_path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.preview:
        return preview(cfg, args.out if args.out.is_absolute() else REPO_ROOT / args.out)
    command = "python -m ppe.shift " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    rows = evaluate(cfg, command, args.only)
    assert all(set(r) == set(FIELDS) for r in rows)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {rows[0]['run_id']}")
    print(f"Table: python -m ppe.shift --table {rows[0]['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
