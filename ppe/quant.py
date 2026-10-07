"""Does quantisation widen the site-shift drop? (Phase 3)

Runs the Phase 2 shift table (clean golden set + the 18 synthetic cells, same
seeded corruptions, operating confidence 0.28 chosen on val in Phase 1) once
per exported model, then compares each pair named in the config on the same
images:

    diff      = metric(candidate) - metric(reference)              on each cell, clean included
    widening  = drop(candidate) - drop(reference)                  on each corrupted cell,
                where drop = metric(cell) - metric(clean) for that model
                (negative widening = the candidate loses MORE under the shift)

Box recall per class (the four no_* classes first), micro recall, the
violation macro recall and person-level violation recall get 95% bootstrap
intervals paired over images: one resample of image indices is applied to
both models and to both the clean and the corrupted copy. mAP@0.5 differences
come from the per-model rows without an interval.

Each model's own shift rows also land in results/results.csv under its own
run id (so `python -m ppe.shift --table RUN_ID` prints its table).

Run (from the repository root):
    python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml                 # everything
    python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml --model onnx_int8  # one model's cells
    python -m ppe.quant --config configs/quant/p3_quant_yolo11s.yaml --compare          # pairs, from saved counts
    python -m ppe.quant --table RUN_ID                                                  # markdown from results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import pickle
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe.data import REPO_ROOT
from ppe.eval import RESULTS_CSV, VIOLATION_CLASSES, append_rows, fmt, git_commit
from ppe.shift import _stats
from ppe.shift import evaluate as shift_evaluate

COUNTS_DIR = REPO_ROOT / "runs" / "quant"


def load_cfg(path: str) -> dict:
    p = Path(path)
    with (p if p.is_absolute() else REPO_ROOT / p).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with (REPO_ROOT / cfg["shift_config"]).open(encoding="utf-8") as fh:
        cfg["shift"] = yaml.safe_load(fh)
    return cfg


def model_cfg(cfg: dict, name: str) -> dict:
    spec = cfg["models"][name]
    s = copy.deepcopy(cfg["shift"])
    s.update(run_name=f"{cfg['run_name']}_{name}", phase=cfg["phase"], model=cfg["model"], weights=spec["weights"],
             precision_mode=name, batch=int(spec.get("batch", 1)))
    if "bootstrap" in cfg:
        s["bootstrap"] = int(cfg["bootstrap"])
    return s


def counts_path(cfg: dict, name: str) -> Path:
    return COUNTS_DIR / cfg["run_name"] / f"{name}.pkl"


def run_model(cfg: dict, name: str, command: str) -> None:
    """Shift table of one model; rows to results.csv, per-image counts to runs/quant/ for --compare."""
    s = model_cfg(cfg, name)
    weights = REPO_ROOT / s["weights"]
    if not weights.exists():
        sys.exit(f"[{name}] weights not found: {s['weights']}. Run: python -m ppe.export --config "
                 f"{cfg.get('export_config', 'configs/export/p3_export_yolo11s.yaml')}")
    only = cfg["models"][name].get("only")
    collect: dict = {}
    print(f"\n===== {name}: {s['weights']} (batch {s['batch']}) =====", flush=True)
    rows = shift_evaluate(s, command, only, collect)
    append_rows(rows)
    out = counts_path(cfg, name)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": rows[0]["run_id"], "weights": rows[0]["weights"], "git_commit": rows[0]["git_commit"],
            "conf": s["operating_conf"], "seed": s["seed"]}
    with out.open("wb") as fh:
        pickle.dump({"meta": meta, "counts": collect}, fh)
    print(f"[{name}] appended {len(rows)} rows as run {meta['run_id']}; counts in {out.relative_to(REPO_ROOT)}")


def ul_map50(run_id: str) -> dict[tuple[str, str], float]:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        return {(r["condition"], r["class"]): float(r["value"]) for r in csv.DictReader(fh)
                if r["run_id"] == run_id and r["metric"] == "map50" and r["dataset"] == "construction-ppe"}


def compare(cfg: dict, command: str) -> list[dict]:
    names = {int(k): v for k, v in cfg["names"].items()}
    nc = len(names)
    by_name = {v: k for k, v in names.items()}
    viol_cls = [by_name[c] for c in VIOLATION_CLASSES]
    classes = list(range(nc))
    n_boot = int(cfg.get("bootstrap", cfg["shift"].get("bootstrap", 1000)))
    seed = int(cfg["shift"]["seed"])

    stamp = datetime.now(timezone.utc)
    run_id = f"{cfg['run_name']}_compare_{stamp.strftime('%Y%m%dT%H%M%SZ')}"
    rows: list[dict] = []
    loaded = {}
    for ref, cand in cfg["comparisons"]:
        for n in (ref, cand):
            if n not in loaded:
                p = counts_path(cfg, n)
                if not p.exists():
                    sys.exit(f"No counts for {n}: run python -m ppe.quant --config ... --model {n}")
                with p.open("rb") as fh:
                    loaded[n] = pickle.load(fh)
        A, B = loaded[ref], loaded[cand]
        if A["meta"]["conf"] != B["meta"]["conf"]:
            sys.exit(f"{ref} and {cand} used different operating confidences")
        cells = [c for c in A["counts"] if c in B["counts"]]
        mA, mB = ul_map50(A["meta"]["run_id"]), ul_map50(B["meta"]["run_id"])
        base = {"run_id": run_id, "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(),
                "phase": cfg["phase"], "model": cfg["model"], "weights": B["meta"]["weights"],
                "precision_mode": f"{cand} vs {ref}", "device": cfg["shift"]["device"],
                "dataset": cfg["shift"]["dataset"], "split": cfg["shift"]["split"], "seed": seed, "command": command}
        src = (f"candidate {cand} run {B['meta']['run_id']}, reference {ref} run {A['meta']['run_id']}; "
               f"conf {A['meta']['conf']} (chosen on val in Phase 1)")
        clean_A, clean_B = A["counts"]["clean"], B["counts"]["clean"]
        n_img = len(clean_A["tp"])
        if clean_A["name"] != clean_B["name"]:
            sys.exit("Reference and candidate saw different images")
        rng = np.random.default_rng(seed)
        boots = [rng.integers(0, n_img, n_img) for _ in range(n_boot)]
        full = np.arange(n_img)
        key_cls = {f"c{c}": names[c] for c in classes} | {"all": "all", "viol_macro": "violation_classes",
                                                         "person_any": "person_any_violation"}
        metric_of = {"viol_macro": "macro_recall", "person_any": "violation_recall"}

        def put(cond, key, metric, value, samples, n, extra):
            if not np.isfinite(value):
                return
            cls = key_cls[key]
            base_metric = metric_of.get(key, "recall")
            m = f"{base_metric}_{metric}"
            note = f"{src}; {extra}; negative = candidate worse"
            rows.append({**base, "condition": cond, "class": cls, "metric": m, "value": fmt(float(value)), "n": n,
                         "notes": note})
            s = np.array([x for x in samples if np.isfinite(x)])
            if len(s) >= 0.9 * n_boot:
                ci_note = f"95% percentile interval, {n_boot} bootstrap resamples of the {n_img} images, paired " \
                          f"across both models and both conditions, seed {seed}"
                for suf, q in (("ci95_low", 2.5), ("ci95_high", 97.5)):
                    rows.append({**base, "condition": cond, "class": cls, "metric": f"{m}_{suf}",
                                 "value": fmt(float(np.percentile(s, q))), "n": n, "notes": ci_note})

        def n_of(cnt, key):
            if key.startswith("c"):
                return int(cnt["gt"][:, int(key[1:])].sum())
            if key == "all":
                return int(cnt["gt"].sum())
            if key == "viol_macro":
                return int(cnt["gt"][:, viol_cls].sum())
            return int(cnt["viol"].sum())

        sA0 = _stats(clean_A, full, classes, viol_cls)
        sB0 = _stats(clean_B, full, classes, viol_cls)
        bA0 = [_stats(clean_A, i, classes, viol_cls) for i in boots]
        bB0 = [_stats(clean_B, i, classes, viol_cls) for i in boots]
        for cell in cells:
            cA, cB = A["counts"][cell], B["counts"][cell]
            sA, sB = _stats(cA, full, classes, viol_cls), _stats(cB, full, classes, viol_cls)
            bA = bA0 if cell == "clean" else [_stats(cA, i, classes, viol_cls) for i in boots]
            bB = bB0 if cell == "clean" else [_stats(cB, i, classes, viol_cls) for i in boots]
            for key in sA:
                n = n_of(cB, key)
                put(cell, key, "quant_diff", sB[key] - sA[key], [b[key] - a[key] for a, b in zip(bA, bB)], n,
                    f"{cand} {sB[key]:.4f} vs {ref} {sA[key]:.4f} on {cell}")
                if cell != "clean":
                    dA, dB = sA[key] - sA0[key], sB[key] - sB0[key]
                    put(cell, key, "delta_widening", dB - dA,
                        [(b[key] - b0[key]) - (a[key] - a0[key]) for a, b, a0, b0 in zip(bA, bB, bA0, bB0)], n,
                        f"drop under {cell}: {cand} {dB:+.4f}, {ref} {dA:+.4f} (vs each model's own clean)")
            for cls in ["all"] + [names[c] for c in classes]:
                a, b = mA.get((cell, cls)), mB.get((cell, cls))
                if a is None or b is None:
                    continue
                rows.append({**base, "condition": cell, "class": cls, "metric": "map50_quant_diff",
                             "value": fmt(b - a), "n": "", "notes": f"{src}; Ultralytics mAP@0.5 {b:.4f} vs {a:.4f}"})
                if cell != "clean" and ("clean", cls) in mA and ("clean", cls) in mB:
                    w = (b - mB[("clean", cls)]) - (a - mA[("clean", cls)])
                    rows.append({**base, "condition": cell, "class": cls, "metric": "map50_delta_widening",
                                 "value": fmt(w), "n": "", "notes": f"{src}; drop {cand} "
                                 f"{b - mB[('clean', cls)]:+.4f}, {ref} {a - mA[('clean', cls)]:+.4f}"})
        print(f"[{cand} vs {ref}] {len(cells)} cells compared", flush=True)
    return rows


def table(run_id: str) -> int:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == run_id]
    if not rows:
        sys.exit(f"Run {run_id} not in results.csv")
    get = {(r["precision_mode"], r["condition"], r["class"], r["metric"]): r for r in rows}
    pairs = list(dict.fromkeys(r["precision_mode"] for r in rows))
    cells = list(dict.fromkeys(r["condition"] for r in rows))
    print(f"Run `{run_id}`, commit `{rows[0]['git_commit']}`. Regenerate: `{rows[0]['command']}`\n")

    def cell(pair, cond, cls, metric):
        r = get.get((pair, cond, cls, metric))
        if r is None:
            return "n/a"
        lo, hi = get.get((pair, cond, cls, f"{metric}_ci95_low")), get.get((pair, cond, cls, f"{metric}_ci95_high"))
        s = f"{float(r['value']):+.3f}"
        if lo and hi:
            l, h = float(lo["value"]), float(hi["value"])
            s += f" [{l:+.2f}, {h:+.2f}]" + ("*" if l > 0 or h < 0 else "")
        return s

    cols = [("all", "map50_quant_diff", "mAP@0.5"), ("all", "recall_quant_diff", "recall"),
            ("violation_classes", "macro_recall_quant_diff", "viol. macro recall")] + \
        [(c, "recall_quant_diff", c) for c in VIOLATION_CLASSES] + \
        [("person_any_violation", "violation_recall_quant_diff", "violators flagged")]
    for pair in pairs:
        print(f"### {pair}: candidate minus reference on the same images (95% paired interval; * excludes 0)\n")
        print("| Cell | " + " | ".join(c[2] for c in cols) + " |")
        print("|---|" + "---:|" * len(cols))
        for cond in cells:
            print(f"| {cond} | " + " | ".join(cell(pair, cond, c, m) for c, m, _ in cols) + " |")
        wcols = [(c, m.replace("quant_diff", "delta_widening"), h) for c, m, h in cols]
        print(f"\n### {pair}: widening of the shift drop (candidate drop minus reference drop; negative = "
              f"quantised model loses more)\n")
        print("| Cell | " + " | ".join(c[2] for c in wcols) + " |")
        print("|---|" + "---:|" * len(wcols))
        sig = 0
        tot = 0
        for cond in cells:
            if cond == "clean":
                continue
            vals = [cell(pair, cond, c, m) for c, m, _ in wcols]
            sig += sum(v.endswith("*") for v in vals[1:])
            tot += sum("[" in v for v in vals[1:])
            print(f"| {cond} | " + " | ".join(vals) + " |")
        print(f"\n{sig} of {tot} recall widening intervals exclude 0 (no correction for multiple comparisons).\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.quant", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="quant YAML, e.g. configs/quant/p3_quant_yolo11s.yaml")
    ap.add_argument("--model", nargs="+", help="run only these models' shift tables (no comparison)")
    ap.add_argument("--compare", action="store_true", help="only compare, from counts saved by earlier --model runs")
    ap.add_argument("--table", metavar="RUN_ID", help="print a comparison run from results.csv")
    args = ap.parse_args(argv)
    if args.table:
        return table(args.table)
    if not args.config:
        ap.error("--config is required")
    cfg = load_cfg(args.config)
    with (REPO_ROOT / cfg["shift"]["data"]).open(encoding="utf-8") as fh:
        cfg["names"] = yaml.safe_load(fh)["names"]
    command = "python -m ppe.quant " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    if not args.compare:
        needed = args.model or list(dict.fromkeys(n for pair in cfg["comparisons"] for n in pair))
        for n in needed:
            if n not in cfg["models"]:
                sys.exit(f"Unknown model {n}; the config has {list(cfg['models'])}")
            run_model(cfg, n, command)
        if args.model:
            return 0
    rows = compare(cfg, command)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {rows[0]['run_id']}")
    print(f"Table: python -m ppe.quant --table {rows[0]['run_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
