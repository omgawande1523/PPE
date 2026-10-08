"""The agentic loop (Phase 5): monitor -> diagnose -> improve -> verify, over a stream, with a state file.

One command runs one scenario end to end and appends its numbers to
results/results.csv and its decision to results/promotions.csv:

    python -m ppe.loop --config configs/loop/p5_camera_fault.yaml
    python -m ppe.loop --config configs/loop/p5_transient.yaml
    python -m ppe.loop --config configs/loop/p5_site_shift.yaml
    python -m ppe.loop --config configs/loop/p5_poisoned.yaml
    python -m ppe.loop --table                       # promotions.csv and the Phase 5 rows, as markdown
    python -m ppe.loop --human-check runs/loop/RUN_ID  # record a filled-in human check of the pseudo-labels
    python -m ppe.loop --rollback runs/loop/RUN_ID     # restore the weights in use before the last promotion

The stream. No real site exists yet (results/missing.md), so a scenario's stream
is SYNTHETIC: windows of training images, each segment either clean, under one of
the Phase 2 corruptions (same code and parameters), or under a camera fault
(black frames, lens covered, frozen frame, defocus). The frames come from the
train split because it is the only pool large enough; the detector has seen those
images clean, which is a limitation the report states.

Per window the orchestrator:
  1. runs the deployed model (ppe.monitor.predict_frames) and the monitor's
     confidence-drift estimate of the recall drop (no labels);
  2. diagnoses the window (ppe.diagnose): camera fault -> alert, no retrain;
     short alarm -> temporal, wait; `persist_windows` alarms -> persistent shift;
  3. on a persistent shift, buffers frames until `min_pool`, mines the
     `n_mine` most-disagreeing frames, pseudo-labels them with the teacher,
     writes the human-check sample and fine-tunes with replay (ppe.improve);
  4. runs the gate on the golden set and the new-site sample (ppe.verify),
     promotes or discards, and logs the decision.
At most `max_retrains` retrains per run, and none within `cooldown_windows` of
the last one.

Labelled measurements that the loop never sees: for every window, the true recall
drop of the active model (its recall on the window's frames minus its recall on
the same source images clean). They are written next to the estimates so the
paper can show how the monitor and diagnosis behaved.

Data rules: the golden set (test split) is read only by the gate and for the
report-only "golden under the site's condition" numbers. The new-site sample is
the validation split under the scenario's condition. Training uses only train
images (mined frames and replay).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from ppe import diagnose as dg
from ppe import improve as im
from ppe import verify as vf
from ppe.data import REPO_ROOT, sha256_path
from ppe.eval import FIELDS, RESULTS_CSV, VIOLATION_CLASSES, append_rows, fmt, git_commit, label_for, load_labels
from ppe.infer import set_seed
from ppe.monitor import Monitor, list_dir, predict_frames, recall_of, tp_gt

WORK = REPO_ROOT / "runs" / "loop"


def load_yaml(p: str | Path) -> dict:
    p = Path(p)
    with (p if p.is_absolute() else REPO_ROOT / p).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ============================================================================= stream

def build_stream(cfg: dict, train: list[Path]) -> tuple[list[list[dict]], set[str]]:
    """Window-by-window frame specs. Returns the windows and the train images used under a corruption or fault."""
    W = int(cfg["window_size"])
    order = np.random.default_rng([int(cfg["seed"]), zlib.crc32(b"stream")]).permutation(len(train))
    pos = 0
    windows, shifted = [], set()
    for seg_i, seg in enumerate(cfg["stream"]["segments"]):
        if seg["kind"] == "condition" and seg["condition"] == "downscale":
            sys.exit("downscale changes the labels; it is not supported in a loop stream")
        n = int(seg["windows"]) * W
        if pos + n > len(train):
            sys.exit(f"stream needs more than the {len(train)} train images")
        srcs = [train[i] for i in order[pos:pos + n]]
        pos += n
        if seg["kind"] != "clean":
            shifted.update(p.name for p in srcs)
        for w in range(int(seg["windows"])):
            k = len(windows)
            win = []
            for f, src in enumerate(srcs[w * W:(w + 1) * W]):
                spec = {"id": f"w{k:03d}f{f:02d}_{src.stem}", "src": src, "segment": seg_i, **seg}
                if seg["kind"] == "fault" and seg["fault"] == "frozen":
                    spec["frozen_src"] = srcs[0]
                win.append(spec)
            windows.append(win)
    return windows, shifted


def segment_name(spec: dict) -> str:
    if spec["kind"] == "condition":
        return f"{spec['condition']}_s{spec['severity']}"
    return spec["kind"] if spec["kind"] == "clean" else f"fault_{spec['fault']}"


def render(spec: dict, cfg: dict, shift_cfg: dict) -> np.ndarray:
    """The frame the camera delivers (deterministic per spec)."""
    import cv2

    from ppe.shift import CORRUPT, image_rng

    if spec["kind"] == "fault" and spec["fault"] == "frozen":
        return cv2.imread(str(spec["frozen_src"]), cv2.IMREAD_COLOR)
    img = cv2.imread(str(spec["src"]), cv2.IMREAD_COLOR)
    if spec["kind"] == "clean":
        return img
    if spec["kind"] == "condition":
        cond, s = spec["condition"], int(spec["severity"]) - 1
        if cond == "jpeg":
            from ppe.shift import jpeg_bytes
            return cv2.imdecode(np.frombuffer(jpeg_bytes(img, shift_cfg["corruptions"][cond], s), np.uint8),
                                cv2.IMREAD_COLOR)
        return CORRUPT[cond](img, shift_cfg["corruptions"][cond], s, image_rng(int(shift_cfg["seed"]), cond, s,
                                                                                 spec["src"].name))
    fc = cfg["faults"]
    rng = np.random.default_rng([int(cfg["seed"]), zlib.crc32(spec["id"].encode())])
    if spec["fault"] == "black":
        x = img.astype(np.float32) * float(fc["black_gain"]) + rng.normal(0, float(fc["black_noise"]), img.shape)
    elif spec["fault"] == "covered":
        x = cv2.GaussianBlur(img, (0, 0), float(fc["covered_sigma"])).astype(np.float32) * float(fc["covered_gain"])
    elif spec["fault"] == "blur":
        x = cv2.GaussianBlur(img, (0, 0), float(fc["defocus_sigma"])).astype(np.float32)
    else:
        raise ValueError(spec["fault"])
    return np.clip(x, 0, 255).astype(np.uint8)


def frame_labels(spec: dict, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """True labels of a frame (used only for measurement and the source-label teacher)."""
    src = spec.get("frozen_src", spec["src"])
    h, w = img.shape[:2]
    return load_labels(label_for(src), w, h)


# ============================================================================= run

class Run:
    def __init__(self, cfg: dict, command: str):
        import ultralytics

        if ultralytics.__version__ != cfg["ultralytics_version"]:
            sys.exit(f"Ultralytics {ultralytics.__version__} installed, config pins {cfg['ultralytics_version']}.")
        self.cfg, self.command = cfg, command
        self.seed = int(cfg["seed"])
        set_seed(self.seed)
        self.stamp = datetime.now(timezone.utc)
        self.run_id = f"{cfg['run_name']}_{self.stamp.strftime('%Y%m%dT%H%M%SZ')}"
        self.work = WORK / self.run_id
        self.work.mkdir(parents=True, exist_ok=True)
        self.data = load_yaml(cfg["data"])
        self.shift_cfg = load_yaml(cfg["shift_config"])
        self.root = REPO_ROOT / self.data["path"]
        if not (self.root / self.data["train"]).is_dir():
            sys.exit(f"Dataset not found at {self.root}. Run: python -m ppe.data --download construction-ppe")
        self.rows: list[dict] = []
        self.state_path = self.work / "state.json"
        self.state = {"run_id": self.run_id, "scenario": cfg["scenario"], "config": cfg["config_path"],
                      "current": cfg["weights"], "history": [], "mode": "monitoring", "retrains": 0,
                      "last_retrain_window": None, "events": [], "windows": []}
        vf.save_state(self.state_path, self.state)
        self.base = {"run_id": self.run_id, "date": self.stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(),
                     "phase": cfg["phase"], "model": cfg["model"], "precision_mode": "fp32", "device": cfg["device"],
                     "seed": self.seed, "command": command}

    # ------------------------------------------------------------------ bookkeeping
    def add(self, dataset, split, cond, cls, metric, value, n, notes, weights=None):
        w = REPO_ROOT / (weights or self.state["current"])
        self.rows.append({**self.base, "weights": f"{w.relative_to(REPO_ROOT).as_posix()}@{sha256_path(w)[:12]}",
                          "dataset": dataset, "split": split, "condition": cond, "class": cls, "metric": metric,
                          "value": fmt(value), "n": n, "notes": notes})

    def event(self, k: int, kind: str, **kw):
        e = {"window": k, "event": kind, **kw}
        self.state["events"].append(e)
        vf.save_state(self.state_path, self.state)
        print(f"  [w{k:02d}] {kind}: {json.dumps(kw, default=str)[:300]}", flush=True)

    def load_model(self):
        from ultralytics import YOLO

        w = REPO_ROOT / self.state["current"]
        self.model = YOLO(str(w), task="detect")
        self.names = {int(k): v for k, v in self.model.names.items()}
        self.nc = len(self.names)
        self.monitor = Monitor(self.model, w, self.cfg, self.data, self.shift_cfg)
        return w

    # ------------------------------------------------------------------ main loop
    def run(self) -> list[dict]:
        cfg = self.cfg
        print(f"Run {self.run_id}: scenario {cfg['scenario']}")
        if cfg.get("golden_manifest"):
            from ppe.golden import verify
            verify(REPO_ROOT / cfg["golden_manifest"])
        train = list_dir(self.root / self.data["train"])
        windows, self.shifted_src = build_stream(cfg, train)
        self.train = train
        self.load_model()
        mon = self.monitor

        # healthy-frame reference for the blur check: the monitor's seeded train reference images
        pick = sorted(np.random.default_rng(self.seed).choice(len(train), int(cfg["monitor"]["reference_images"]),
                                                              replace=False))
        import cv2
        ref_stats = [dg.health(cv2.imread(str(train[i]), cv2.IMREAD_COLOR), None)[0] for i in pick]
        self.ref_sharp_q01 = dg.reference_sharpness(ref_stats)
        diag = dg.Diagnoser(cfg["diagnose"], mon.threshold, self.ref_sharp_q01)
        ds = "construction-ppe"
        self.add(ds, "val", "clean", "all", "monitor_alarm_threshold", mon.threshold, mon.calib["clean_val_windows"],
                 f"max(min_threshold {cfg['monitor']['min_threshold']}, q{cfg['monitor']['threshold_quantile']} of the "
                 f"estimate over {mon.calib['clean_val_windows']} clean val windows of {cfg['window_size']} images); "
                 f"ridge fitted on val cells {mon.calib['fit_cells']}")
        self.add(ds, "train", "clean", "all", "health_sharpness_q01", self.ref_sharp_q01, len(pick),
                 "1st percentile of normalised Laplacian variance over the healthy reference train images")

        D = cfg["diagnose"]
        buffer: list[list[tuple[dict, dict]]] = []   # last persist_windows windows: (spec, detections)
        pool: list[tuple[dict, dict]] = []
        prev_small = None
        for k, win in enumerate(windows):
            imgs = [render(s, cfg, self.shift_cfg) for s in win]
            dets = predict_frames(self.model, imgs, cfg)
            est, feats = mon.estimate(dets)
            stats = []
            for im_ in imgs:
                st, prev_small = dg.health(im_, prev_small)
                stats.append(st)
            d = diag.window(k, stats, est)

            # labelled measurement, never used by the loop: true recall drop of the active model on this window
            clean_dets = predict_frames(self.model, [s.get("frozen_src", s["src"]) for s in win], cfg)
            t, iou = float(cfg["operating_conf"]), float(cfg["match_iou"])
            tp_x, gt_x, tp_c, gt_c = [], [], [], []
            for s, im_, dx, dc in zip(win, imgs, dets, clean_dets):
                a, b = tp_gt(dx, *frame_labels(s, im_), self.nc, t, iou)
                tp_x.append(a); gt_x.append(b)
                h, w = dc["hw"]
                a, b = tp_gt(dc, *load_labels(label_for(s.get("frozen_src", s["src"])), w, h), self.nc, t, iou)
                tp_c.append(a); gt_c.append(b)
            tp_x, gt_x, tp_c, gt_c = map(np.stack, (tp_x, gt_x, tp_c, gt_c))
            true = recall_of(tp_c, gt_c) - recall_of(tp_x, gt_x)
            viol = [c for c in range(self.nc) if self.names[c] in VIOLATION_CLASSES]
            true_v = recall_of(tp_c, gt_c, viol) - recall_of(tp_x, gt_x, viol)

            seg = segment_name(win[0])
            rec = {"window": k, "segment": seg, "est_drop": est, **feats, "true_drop": true,
                   "true_drop_violation": true_v, "finding": d["finding"], "fault_type": d["fault_type"],
                   "action": d["action"], "fault_frac": d["fault_frac"], "fault_counts": d["fault_counts"],
                   "alarm_run": d["alarm_run"], "weights": self.state["current"], "mode": self.state["mode"]}
            self.state["windows"].append(rec)
            vf.save_state(self.state_path, self.state)
            print(f"  [w{k:02d}] {seg:16s} est {est:+.3f} true {true:+.3f} faults {d['fault_frac']:.2f} "
                  f"-> {d['finding']} ({d['action']})", flush=True)
            cond = f"stream_w{k:02d}_{seg}"
            note = (f"{len(win)} train frames; finding {d['finding']} ({d['fault_type']}), action {d['action']}; "
                    f"mode {self.state['mode']}")
            self.add(ds, "train", cond, "all", "est_recall_drop", est, len(win),
                     f"monitor (confidence drift, no labels); {note}")
            self.add(ds, "train", cond, "all", "true_recall_drop", true, int(gt_x.sum()),
                     "labelled measurement, not seen by the loop: active model's recall on the same source images "
                     "clean minus on the delivered frames, at the operating confidence")
            self.add(ds, "train", cond, "violation", "true_recall_drop", true_v, int(gt_x[:, viol].sum()),
                     "labelled measurement, not seen by the loop; pooled no_* classes")
            self.add(ds, "train", cond, "all", "camera_fault_frame_share", d["fault_frac"], len(win),
                     f"share of frames failing an image-health check {d['fault_counts']}")
            self.add(ds, "train", cond, "all", "alarm", float(est >= mon.threshold), 1,
                     f"1 = estimate >= threshold {mon.threshold:.4f}")

            if d["finding"] == dg.CAMERA_FAULT:
                self.event(k, "alert_human_fix_camera", fault=d["fault"], fault_type=d["fault_type"],
                           share=d["fault_frac"])
                continue
            if d["finding"] == dg.TEMPORAL:
                self.event(k, "log_and_wait", alarm_run=d["alarm_run"], est=est)
            if d.get("recovered_after"):
                self.event(k, "recovered", after_windows=d["recovered_after"])
            buffer.append([(s, x) for s, x in zip(win, dets)])
            buffer = buffer[-int(D["persist_windows"]):]

            if self.state["mode"] == "monitoring" and d["finding"] == dg.PERSISTENT:
                lr = self.state["last_retrain_window"]
                if self.state["retrains"] >= int(D["max_retrains"]):
                    self.event(k, "retrain_blocked", why=f"max_retrains {D['max_retrains']} reached")
                elif lr is not None and k - lr < int(D["cooldown_windows"]):
                    self.event(k, "retrain_blocked", why=f"cooldown: last retrain at window {lr}")
                else:
                    self.state["mode"] = "collecting"
                    self.state["trigger_window"] = k
                    pool = [x for w_ in buffer for x in w_]
                    self.event(k, "persistent_shift_enter_improve", pooled=len(pool))
            elif self.state["mode"] == "collecting":
                pool.extend((s, x) for s, x in zip(win, dets))
            if self.state["mode"] == "collecting" and len(pool) >= int(cfg["improve"]["min_pool"]):
                self.improve_and_verify(k, pool)
                pool = []
                self.state["mode"] = "monitoring"
                self.state["last_retrain_window"] = k
                self.state["retrains"] += 1
                w = self.load_model()            # the active weights may have changed: recalibrate the monitor
                mon = self.monitor
                diag = dg.Diagnoser(cfg["diagnose"], mon.threshold, self.ref_sharp_q01)
                buffer = []
                vf.save_state(self.state_path, self.state)

        if self.state["mode"] == "collecting":
            self.event(len(windows) - 1, "stream_ended_while_collecting", pooled=len(pool),
                       needed=cfg["improve"]["min_pool"])
        if not self.state["retrains"]:
            self.log_no_candidate()
        self.add(ds, "train", "stream", "all", "retrains", self.state["retrains"], len(windows),
                 f"retrains in a {len(windows)}-window stream (scenario {cfg['scenario']})")
        vf.save_state(self.state_path, self.state)
        return self.rows

    # ------------------------------------------------------------------ no candidate (fault or transient scenarios)
    def log_no_candidate(self):
        ws = self.state["windows"]
        faults = [w for w in ws if w["finding"] == dg.CAMERA_FAULT]
        temps = [w for w in ws if w["finding"] == dg.TEMPORAL]
        pers = [w for w in ws if w["finding"] == dg.PERSISTENT]
        parts = []
        if faults:
            kinds = sorted({w["fault_type"] for w in faults})
            parts.append(f"camera fault ({', '.join(kinds)}) in windows {[w['window'] for w in faults]}: "
                         f"human alerted, retrain suppressed")
        if temps:
            parts.append(f"temporal alarm in windows {[w['window'] for w in temps]} recovered: logged and waited")
        if pers:
            parts.append(f"persistent shift in windows {[w['window'] for w in pers]} but no retrain ran "
                         f"(see state.json events)")
        if not parts:
            parts.append("no alarm")
        vf.log_decision({"date": self.stamp.strftime("%Y-%m-%d"), "run_id": self.run_id,
                         "scenario": self.cfg["scenario"], "candidate": "", "incumbent": self.state["current"],
                         "decision": "no_retrain", "reason": "; ".join(parts), "human_minutes": 0,
                         "command": self.command})

    # ------------------------------------------------------------------ improve + verify
    def improve_and_verify(self, k: int, pool: list[tuple[dict, dict]]):
        cfg, ic = self.cfg, self.cfg["improve"]
        ds = "construction-ppe"
        t0 = time.time()
        specs = [s for s, _ in pool]
        frames = [render(s, cfg, self.shift_cfg) for s in specs]
        scores = np.array([im.disagreement(self.model, f, d, cfg) for f, (_, d) in zip(frames, pool)])
        idx = im.mine(scores, int(ic["n_mine"]), self.seed)
        if len(idx) < int(ic["min_mined"]):
            self.event(k, "improve_skipped", why=f"{len(idx)} mined < min_mined {ic['min_mined']}")
            return
        m_specs, m_frames = [specs[i] for i in idx], [frames[i] for i in idx]
        self.event(k, "mined", pool=len(pool), mined=len(idx), score_mined_mean=float(scores[idx].mean()),
                   score_pool_mean=float(scores.mean()))

        tc = ic["teacher"]
        teacher = im.TEACHERS[tc["kind"]](tc, self.names, cfg)
        labels = teacher(m_frames, [label_for(s.get("frozen_src", s["src"])) for s in m_specs])
        truth = [frame_labels(s, f) for s, f in zip(m_specs, m_frames)]
        viol = [c for c in self.names if self.names[c] in VIOLATION_CLASSES]
        n_poison = 0
        if ic.get("poison"):
            labels, n_poison = im.poison(labels, ic["poison"], self.names, self.seed)
        err = im.label_error(labels, truth, float(cfg["match_iou"]))
        err_v = im.label_error(labels, truth, float(cfg["match_iou"]), viol)
        self.event(k, "pseudo_labelled", teacher=tc["kind"], poisoned_boxes=n_poison, **err)

        hc = self.work / "human_check"
        sample = im.human_sample(hc, m_frames, labels, self.names, [s["id"] for s in m_specs],
                                 int(ic["human_sample"]), self.seed)
        self.event(k, "human_check_written", frames=len(sample), folder=str(hc.relative_to(REPO_ROOT)),
                   status="pending: nobody has checked it")

        replay_pool = [p for p in self.train if p.name not in self.shifted_src]
        rp = np.random.default_rng([self.seed, zlib.crc32(b"replay")]).choice(len(replay_pool), int(ic["n_replay"]),
                                                                               replace=False)
        replay = [replay_pool[i] for i in sorted(rp)]
        data_yaml = im.build_dataset(self.work / "finetune_data",
                                     [(s["id"], f, l) for s, f, l in zip(m_specs, m_frames, labels)], replay,
                                     self.names)
        incumbent = REPO_ROOT / self.state["current"]
        t1 = time.time()
        cand = im.finetune(incumbent, data_yaml, ic["finetune"], self.seed, cfg["device"], self.work / "train",
                           "candidate")
        ft_s = time.time() - t1
        self.event(k, "fine_tuned", candidate=str(cand.relative_to(REPO_ROOT)), seconds=round(ft_s),
                   mined=len(idx), replay=len(replay))

        # ---- verify
        from ppe.golden import verify as verify_golden
        from ppe.shift import make_cell

        if cfg.get("golden_manifest"):
            verify_golden(REPO_ROOT / cfg["golden_manifest"])
        site = cfg["site"]
        cond, s = site["condition"], int(site["severity"]) - 1
        cells = {}
        for split in ("val", "test"):
            out_root = WORK / "cells" / f"{split}_{cond}_s{s + 1}"
            if not (out_root / "images" / "test").is_dir():
                make_cell(list_dir(self.root / self.data[split]), cond, s, self.shift_cfg["corruptions"][cond],
                          int(self.shift_cfg["seed"]), out_root)
            cells[split] = out_root / "images" / "test"
        golden_dir = self.root / self.data["test"]
        vw = self.work / "verify"
        res = {}
        for who, wpath in (("incumbent", incumbent), ("candidate", cand)):
            res[(who, "golden")] = vf.measure(wpath, golden_dir, cfg, vw, f"{who}_golden")
            res[(who, "site")] = vf.measure(wpath, cells["val"], cfg, vw, f"{who}_site")
            res[(who, "golden_site")] = vf.measure(wpath, cells["test"], cfg, vw, f"{who}_golden_site")
        tol = float(cfg["verify"]["map50_tolerance"])
        ok, lines = vf.gate(res[("candidate", "golden")], res[("incumbent", "golden")], res[("candidate", "site")],
                            res[("incumbent", "site")], tol)
        for ln in lines:
            print("   ", ln)
        tag = f"{self.run_id}_w{k:02d}"
        if ok:
            dst = vf.promote(self.state, cand, self.work / "promoted", int(cfg["verify"]["keep_last"]), tag)
            decision, reason = "promoted", "all gate rules passed: " + " | ".join(lines)
            cand_ref = str(dst.relative_to(REPO_ROOT))
        else:
            decision = "rejected"
            reason = "failed: " + " | ".join(ln for ln in lines if ln.startswith("FAIL")) + \
                     " || passed: " + " | ".join(ln for ln in lines if ln.startswith("PASS"))
            cand_ref = str(cand.relative_to(REPO_ROOT))
        self.event(k, decision, reason=reason)
        inc_ref = str(incumbent.relative_to(REPO_ROOT))
        vf.log_decision({
            "date": self.stamp.strftime("%Y-%m-%d"), "run_id": self.run_id, "scenario": cfg["scenario"],
            "candidate": f"{cand_ref}@{res[('candidate', 'golden')]['weights_sha']}",
            "incumbent": f"{inc_ref}@{res[('incumbent', 'golden')]['weights_sha']}",
            "candidate_recall": vf.recall_str(res[("candidate", "golden")]),
            "incumbent_recall": vf.recall_str(res[("incumbent", "golden")]),
            "candidate_map50": fmt(res[("candidate", "golden")]["map50"]),
            "incumbent_map50": fmt(res[("incumbent", "golden")]["map50"]),
            "candidate_site_recall": vf.recall_str(res[("candidate", "site")]),
            "incumbent_site_recall": vf.recall_str(res[("incumbent", "site")]),
            "map50_tolerance": tol,
            "pseudo_label_error_rate": f"{err['error_rate']:.6f} vs source labels ({err['n_pseudo']} boxes; "
                                       f"automated, synthetic stream only)",
            "pseudo_label_error_rate_human": f"pending ({len(sample)} frames in {hc.relative_to(REPO_ROOT)})",
            "human_minutes": 0, "decision": decision, "reason": reason, "command": self.command})

        # ---- rows
        tn = f"teacher {tc['kind']}" + (f", poisoned ({n_poison} boxes swapped)" if n_poison else "")
        self.add(ds, "train", "stream", "all", "retrain_trigger_window", self.state["trigger_window"], 1,
                 f"window where the persistent shift was declared; improve ran after window {k}")
        self.add(ds, "train", "stream", "all", "frames_pooled", len(pool), len(pool), "frames buffered for mining")
        self.add(ds, "train", "stream", "all", "frames_mined", len(idx), len(pool),
                 f"top augmentation-disagreement frames; mean score mined {scores[idx].mean():.4f} vs pool "
                 f"{scores.mean():.4f}")
        self.add(ds, "train", "stream", "all", "replay_frames", len(replay), len(replay_pool),
                 "seeded clean train images with original labels")
        for nm, e in (("all", err), ("violation", err_v)):
            self.add(ds, "train", "stream", nm, "pseudo_label_error_rate_vs_source", e["error_rate"], e["n_pseudo"],
                     f"{tn}; wrong pseudo boxes / pseudo boxes against the source images' labels (IoU "
                     f"{cfg['match_iou']}, same class); automated, NOT a human check")
            self.add(ds, "train", "stream", nm, "pseudo_label_miss_rate_vs_source", e["miss_rate"], e["n_true"],
                     f"{tn}; labelled objects without a pseudo box / labelled objects")
        self.add(ds, "train", "stream", "all", "poisoned_boxes", n_poison, err["n_pseudo"], tn)
        self.add(ds, "train", "stream", "all", "human_minutes", 0, len(sample),
                 f"human check pending: {len(sample)} frames written to {hc.relative_to(REPO_ROOT)}; "
                 f"the human pseudo-label error rate is missing until it is filled in")
        self.add(ds, "train", "stream", "all", "finetune_seconds", ft_s, len(idx) + len(replay),
                 f"{ic['finetune']['epochs']} epochs on cpu; {json.dumps(ic['finetune'], sort_keys=True)}")
        what = {"golden": ("golden_v1", "clean", "gate: golden set"),
                "site": ("val", f"{cond}_s{s + 1}", "gate: new-site sample (val under the site condition)"),
                "golden_site": ("golden_v1", f"{cond}_s{s + 1}", "report only, not a gate rule: golden set under "
                                                                    "the site condition")}
        for who, wpath in (("incumbent", inc_ref), ("candidate", cand_ref)):
            for part, (split, cnd, desc) in what.items():
                r = res[(who, part)]
                for c, v in r["recall"].items():
                    n = r["gt"].get(c, sum(r["gt"].values()) if c == "all" else
                                    sum(r["gt"][x] for x in VIOLATION_CLASSES))
                    self.add(ds, split, cnd, c, f"recall_{who}", v, n,
                             f"{desc}; {who}; operating conf {cfg['operating_conf']}; {r['n_images']} images",
                             weights=wpath)
                for mname in ("map50", "map50_95"):
                    self.add(ds, split, cnd, "all", f"{mname}_{who}", r[mname], r["n_images"],
                             f"{desc}; {who}; Ultralytics val conf 0.001", weights=wpath)
        self.add(ds, "golden_v1", "clean", "all", "promoted", float(ok), 1,
                 f"gate decision ({decision}); tolerance {tol}; " + " | ".join(lines))
        self.add(ds, "train", "stream", "all", "improve_seconds", time.time() - t0, len(pool),
                 "mining + teacher + fine-tune + verify")


# ============================================================================= other commands

def human_check(run_dir: Path) -> int:
    """Read a filled-in human_check.csv and append the human pseudo-label error rate and minutes."""
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    f = run_dir / "human_check" / "human_check.csv"
    with f.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)]
    done = [r for r in rows if r["wrong_boxes"].strip() != "" and r["minutes"].strip() != ""]
    if not done:
        print(f"{f}: no rows filled in yet")
        return 1
    boxes = sum(int(r["pseudo_boxes"]) for r in done)
    wrong = sum(int(r["wrong_boxes"]) for r in done)
    missing = sum(int(r["missing_boxes"] or 0) for r in done)
    minutes = sum(float(r["minutes"]) for r in done)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    stamp = datetime.now(timezone.utc)
    base = {"run_id": state["run_id"], "date": stamp.strftime("%Y-%m-%d"), "git_commit": git_commit(), "phase": 5,
            "model": "yolo11s", "weights": "", "precision_mode": "fp32", "device": "human", "dataset": "construction-ppe",
            "split": "train", "condition": "stream", "class": "all", "seed": "",
            "command": f"python -m ppe.loop --human-check {run_dir.relative_to(REPO_ROOT).as_posix()}"}
    checkers = sorted({r["checker"] for r in done})
    rows_out = [
        {**base, "metric": "pseudo_label_error_rate_human", "value": fmt(wrong / boxes if boxes else float("nan")),
         "n": boxes, "notes": f"{len(done)} frames checked by {checkers}; wrong boxes / pseudo boxes"},
        {**base, "metric": "pseudo_label_missing_per_frame_human", "value": fmt(missing / len(done)), "n": len(done),
         "notes": "objects without a pseudo box, per checked frame"},
        {**base, "metric": "human_minutes", "value": fmt(minutes), "n": len(done), "notes": "minutes spent checking"},
    ]
    append_rows(rows_out)
    print(f"Appended human check of {len(done)} frames: error {wrong}/{boxes}, {minutes} minutes")
    return 0


def do_rollback(run_dir: Path) -> int:
    run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
    sp = run_dir / "state.json"
    state = json.loads(sp.read_text(encoding="utf-8"))
    cur = vf.rollback(state)
    vf.save_state(sp, state)
    print(f"Rolled back; current weights: {cur}")
    return 0


def table() -> int:
    with vf.PROMOTIONS_CSV.open(encoding="utf-8") as fh:
        prom = list(csv.DictReader(fh))
    print("### Promotion log (results/promotions.csv)\n")
    print("| run | scenario | decision | golden mAP50 cand / inc | reason |")
    print("|---|---|---|---|---|")
    for r in prom:
        print(f"| {r['run_id']} | {r['scenario']} | {r['decision']} | {r['candidate_map50'] or '-'} / "
              f"{r['incumbent_map50'] or '-'} | {r['reason'][:400]} |")
    with RESULTS_CSV.open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["phase"] == "5"]
    runs = [r["run_id"] for r in prom if r["run_id"]]
    for run in runs:
        rr = [r for r in rows if r["run_id"] == run]
        if not rr:
            continue
        v = {(r["split"], r["condition"], r["class"], r["metric"]): r["value"] for r in rr}
        print(f"\n### {run}\n")
        print("| window | segment | est drop | true drop | true drop (no_*) | fault share | alarm |")
        print("|---|---|---|---|---|---|---|")
        for cond in sorted({r["condition"] for r in rr if r["condition"].startswith("stream_w")}):
            g = lambda c, m: v.get(("train", cond, c, m), "")
            print(f"| {cond[8:10]} | {cond[11:]} | {g('all', 'est_recall_drop')} | {g('all', 'true_recall_drop')} | "
                  f"{g('violation', 'true_recall_drop')} | {g('all', 'camera_fault_frame_share')} | "
                  f"{g('all', 'alarm')} |")
        cl = [r for r in rr if r["metric"].startswith("recall_")]
        if cl:
            print("\n| set | class | incumbent | candidate | n |")
            print("|---|---|---|---|---|")
            keys = sorted({(r["split"], r["condition"], r["class"]) for r in cl})
            for sp, cnd, c in keys:
                print(f"| {sp} {cnd} | {c} | {v.get((sp, cnd, c, 'recall_incumbent'), '')} | "
                      f"{v.get((sp, cnd, c, 'recall_candidate'), '')} | "
                      f"{next(r['n'] for r in cl if (r['split'], r['condition'], r['class']) == (sp, cnd, c))} |")
        for r in rr:
            if r["condition"] == "stream" or r["metric"] in ("promoted", "monitor_alarm_threshold") or \
                    r["metric"].startswith("map50"):
                print(f"- {r['split']} {r['condition']} {r['class']} {r['metric']}: {r['value']} (n={r['n']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.loop", description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="scenario YAML, e.g. configs/loop/p5_site_shift.yaml")
    ap.add_argument("--table", action="store_true", help="print the promotion log and Phase 5 rows")
    ap.add_argument("--human-check", metavar="RUN_DIR", help="record a filled-in human check")
    ap.add_argument("--rollback", metavar="RUN_DIR", help="restore the weights before the last promotion")
    args = ap.parse_args(argv)
    if args.table:
        return table()
    if args.human_check:
        return human_check(Path(args.human_check))
    if args.rollback:
        return do_rollback(Path(args.rollback))
    if not args.config:
        ap.error("--config is required")
    cfg = load_yaml(args.config)
    cfg["config_path"] = args.config
    command = "python -m ppe.loop " + " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))
    rows = Run(cfg, command).run()
    assert all(set(r) == set(FIELDS) for r in rows)
    append_rows(rows)
    print(f"\nAppended {len(rows)} rows to results/results.csv as run {rows[0]['run_id']}")
    print("Tables: python -m ppe.loop --table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
