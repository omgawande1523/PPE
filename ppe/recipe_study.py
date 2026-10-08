"""Fine-tune recipe study on the validation split only (Phase 5).

The first site-shift run's candidate improved the new-site sample but was rejected by
the gate for forgetting (no_* recall and mAP fell on the golden set). Trying recipes
until one passes the golden gate would tune on the golden set, which the brief forbids.
So recipes are compared here WITHOUT the golden set: each is fine-tuned on the same
mined frames as that run, then scored with the gate's own three rules, but on the clean
VALIDATION split instead of the golden set (a "dev gate"), plus the new-site sample.
The recipe a loop config then uses is chosen from these rows alone.

    python -m ppe.recipe_study --config configs/loop/p5_recipe_study.yaml

It needs the mined frames of the site-shift run named in the config
(runs/loop/<run_id>/finetune_data, written by python -m ppe.loop --config
configs/loop/p5_site_shift.yaml). The test split is never read here.
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ppe import improve as im
from ppe import verify as vf
from ppe.data import REPO_ROOT, sha256_path
from ppe.eval import FIELDS, append_rows, fmt, git_commit, label_for
from ppe.loop import WORK, load_yaml
from ppe.monitor import list_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.recipe_study", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_yaml(args.config)
    base_cfg = load_yaml(cfg["loop_config"])
    command = "python -m ppe.recipe_study " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    src = WORK / cfg["mined_from_run"] / "finetune_data"
    if not (src / "images" / "train").is_dir():
        sys.exit(f"{src} not found. Run: python -m ppe.loop --config {cfg['loop_config']} and set mined_from_run.")
    seed = int(base_cfg["seed"])
    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    work = WORK / run_id
    data = load_yaml(base_cfg["data"])
    root = REPO_ROOT / data["path"]

    from ultralytics import YOLO
    names = {int(k): v for k, v in YOLO(str(REPO_ROOT / base_cfg["weights"])).names.items()}
    mined = sorted(p for p in (src / "images" / "train").iterdir() if not p.name.startswith("replay_"))
    # replay pool = train images not used under the condition in the stream: rebuild it as ppe.loop does
    from ppe.loop import build_stream
    train = list_dir(root / data["train"])
    _, shifted_src = build_stream(base_cfg, train)
    replay_pool = [p for p in train if p.name not in shifted_src]

    from ppe.shift import make_cell
    site = base_cfg["site"]
    cond, s = site["condition"], int(site["severity"]) - 1
    site_root = WORK / "cells" / f"val_{cond}_s{s + 1}"
    if not (site_root / "images" / "test").is_dir():
        shift_cfg = load_yaml(base_cfg["shift_config"])
        make_cell(list_dir(root / data["val"]), cond, s, shift_cfg["corruptions"][cond], int(shift_cfg["seed"]),
                  site_root)
    val_dir, site_dir = root / data["val"], site_root / "images" / "test"
    incumbent = REPO_ROOT / base_cfg["weights"]

    rows: list[dict] = []
    base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": 5,
            "model": base_cfg["model"], "precision_mode": "fp32", "device": base_cfg["device"], "seed": seed,
            "command": command, "dataset": "construction-ppe"}

    def add(weights: Path, split, cond_, cls, metric, value, n, notes):
        rows.append({**base, "weights": f"{weights.relative_to(REPO_ROOT).as_posix()}@{sha256_path(weights)[:12]}",
                     "split": split, "condition": cond_, "class": cls, "metric": metric, "value": fmt(value), "n": n,
                     "notes": notes})

    inc_val = vf.measure(incumbent, val_dir, base_cfg, work / "verify", "incumbent_val")
    inc_site = vf.measure(incumbent, site_dir, base_cfg, work / "verify", "incumbent_site")
    tol = float(base_cfg["verify"]["map50_tolerance"])
    def train_and_measure(name, rc, n_replay):
        pick = np.random.default_rng([seed, zlib.crc32(b"replay")]).choice(len(replay_pool), n_replay, replace=False)
        replay = [replay_pool[i] for i in sorted(pick)]
        ds = work / f"data_{name}"
        if ds.exists():
            shutil.rmtree(ds)
        (ds / "images" / "train").mkdir(parents=True)
        (ds / "labels" / "train").mkdir(parents=True)
        for p in mined:
            shutil.copy2(p, ds / "images" / "train" / p.name)
            shutil.copy2(src / "labels" / "train" / f"{p.stem}.txt", ds / "labels" / "train" / f"{p.stem}.txt")
        for p in replay:
            shutil.copy2(p, ds / "images" / "train" / f"replay_{p.name}")
            shutil.copy2(label_for(p), ds / "labels" / "train" / f"replay_{p.stem}.txt")
        import yaml
        dy = ds / "data.yaml"
        dy.write_text(yaml.safe_dump({"path": str(ds.resolve()), "train": "images/train", "val": "images/train",
                                      "names": names}, sort_keys=False), encoding="utf-8")
        t0 = time.time()
        cand = im.finetune(incumbent, dy, rc["finetune"], seed, base_cfg["device"], work / "train", name)
        secs = time.time() - t0
        cv = vf.measure(cand, val_dir, base_cfg, work / "verify", f"{name}_val")
        cs = vf.measure(cand, site_dir, base_cfg, work / "verify", f"{name}_site")
        return cand, secs, cv, cs

    summary = []
    for name, rc in cfg["recipes"].items():
        if rc.get("weights"):          # an existing candidate (the rejected run's): measure only
            cand, secs, n_replay = REPO_ROOT / rc["weights"], 0.0, int(rc["n_replay"])
            cv = vf.measure(cand, val_dir, base_cfg, work / "verify", f"{name}_val")
            cs = vf.measure(cand, site_dir, base_cfg, work / "verify", f"{name}_site")
            rc = {**rc, "finetune": "as in the loop config (existing candidate)"}
        else:
            cand, secs, cv, cs, n_replay = None, 0.0, None, None, int(rc["n_replay"])
        if cand is None:
            cand, secs, cv, cs = train_and_measure(name, rc, n_replay)
        ok, lines = vf.gate(cv, inc_val, cs, inc_site, tol, ref="val-clean")
        summary.append((name, ok, lines))
        print(f"\nRecipe {name}: dev gate {'PASS' if ok else 'FAIL'} ({secs:.0f} s)")
        for ln in lines:
            print("   ", ln)
        desc = f"recipe {name}: {n_replay} replay + {len(mined)} mined frames; {rc['finetune']}"
        for who, w, mv, ms in (("incumbent", incumbent, inc_val, inc_site), ("candidate", cand, cv, cs)):
            for part, m, cnd in (("val clean (dev gate)", mv, "clean"), ("new-site sample", ms, f"{cond}_s{s + 1}")):
                for c, v in m["recall"].items():
                    add(w, "val", cnd, c, f"recall_{who}_{name}", v, m["gt"].get(c, ""), f"{desc}; {part}")
                add(w, "val", cnd, "all", f"map50_{who}_{name}", m["map50"], m["n_images"], f"{desc}; {part}")
        add(cand, "val", "clean", "all", f"dev_gate_pass_{name}", float(ok), 1,
            f"{desc}; gate rules on val clean instead of golden: " + " | ".join(lines))
        if secs:
            add(cand, "train", "stream", "all", f"finetune_seconds_{name}", secs, len(mined) + n_replay, desc)

    assert all(set(r) == set(FIELDS) for r in rows)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {run_id}")
    for name, ok, _ in summary:
        print(f"  {name}: dev gate {'PASS' if ok else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
