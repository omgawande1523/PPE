"""Stage 2 of the loop: camera fault versus domain shift.

The brief's fault vocabulary (IoT data fault taxonomy) and responses:

  Finding                                         Fault type                     Response
  frozen or black frames, sudden blur, lens       stuck-at, degradation,         alert a human to fix the camera;
  covered                                         nonfunctional                  do not retrain
  brief drop that recovers (rain, glare)          temporal                       log it and wait; do not retrain
  sustained drop with a healthy image feed        persistent domain shift        enter the improve stage

Per frame, four image-health checks that need no model and no labels:

  nonfunctional   mean intensity below `black_mean` (black frame), or intensity
                  standard deviation below `covered_std` (lens covered: a flat image)
  stuck_at        mean absolute difference from the previous frame below `frozen_diff`
  degradation     sharpness below `blur_ratio` x the 1st percentile of healthy training
                  frames. Sharpness = variance of the Laplacian of the grey image after
                  dividing it by its own standard deviation, so a dark but sharp frame
                  (low light) is not mistaken for blur.

A window is a camera fault when at least `fault_frac` of its frames fail a
check; its estimate is then not counted towards a shift alarm. Otherwise the
monitor's estimated drop decides: a window alarms when the estimate is at or
above the monitor threshold; `persist_windows` consecutive alarmed windows are
a persistent shift; a run of alarms that ends sooner is logged as temporal.

Thresholds come from the config and from healthy training frames; nothing here
reads labels or the test split.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

HEALTHY, CAMERA_FAULT, TEMPORAL, PERSISTENT = "healthy", "camera_fault", "temporal", "persistent_shift"
ACTIONS = {HEALTHY: "none", CAMERA_FAULT: "alert_human_fix_camera", TEMPORAL: "log_and_wait",
           PERSISTENT: "improve"}
FAULT_TYPE = {"black": "nonfunctional", "covered": "nonfunctional", "frozen": "stuck_at", "blur": "degradation"}


def health(img: np.ndarray, prev_small: np.ndarray | None, side: int = 320) -> tuple[dict, np.ndarray]:
    """Image-health statistics of one BGR frame (0-1 intensity scale) and its small grey copy."""
    import cv2

    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    s = side / max(h, w)
    small = cv2.resize(g, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32) / 255.0
    sd = float(small.std())
    lap = cv2.Laplacian(small / max(sd, 1e-3), cv2.CV_32F)
    out = {"mean": float(small.mean()), "std": sd, "sharpness": float(lap.var()),
           "diff": float(np.abs(small - prev_small).mean()) if prev_small is not None and
           prev_small.shape == small.shape else float("nan")}
    return out, small


def reference_sharpness(stats: list[dict]) -> float:
    """1st percentile of sharpness over healthy training frames."""
    return float(np.percentile([s["sharpness"] for s in stats], 1))


def frame_faults(st: dict, cfg: dict, ref_sharp_q01: float) -> list[str]:
    """At most one fault per frame, checked in this order: black, covered, frozen, blur."""
    if st["mean"] < cfg["black_mean"]:
        return ["black"]
    if st["std"] < cfg["covered_std"]:
        return ["covered"]
    if np.isfinite(st["diff"]) and st["diff"] < cfg["frozen_diff"]:
        return ["frozen"]
    if st["sharpness"] < cfg["blur_ratio"] * ref_sharp_q01:
        return ["blur"]
    return []


@dataclass
class Diagnoser:
    cfg: dict
    threshold: float
    ref_sharp_q01: float
    alarm_run: int = 0
    log: list = field(default_factory=list)

    def window(self, k: int, stats: list[dict], est_drop: float) -> dict:
        """Decide one window. Returns the finding, fault type, action and evidence."""
        per = [frame_faults(s, self.cfg, self.ref_sharp_q01) for s in stats]
        frac = float(np.mean([bool(p) for p in per])) if per else 0.0
        kinds = Counter(x for p in per for x in p)
        out = {"window": k, "fault_frac": frac, "fault_counts": dict(kinds), "est_drop": est_drop,
               "threshold": self.threshold}
        if frac >= float(self.cfg["fault_frac"]):
            top = kinds.most_common(1)[0][0]
            out.update(finding=CAMERA_FAULT, fault_type=FAULT_TYPE[top], fault=top)
            # a camera fault does not count towards, or reset, a shift alarm run
        elif est_drop >= self.threshold:
            self.alarm_run += 1
            if self.alarm_run >= int(self.cfg["persist_windows"]):
                out.update(finding=PERSISTENT, fault_type="persistent_domain_shift")
            else:
                out.update(finding=TEMPORAL, fault_type="temporal",
                           note=f"alarm {self.alarm_run}/{self.cfg['persist_windows']} consecutive windows")
        else:
            if self.alarm_run:
                out["recovered_after"] = self.alarm_run
            self.alarm_run = 0
            out.update(finding=HEALTHY, fault_type="none")
        out["action"] = ACTIONS[out["finding"]]
        out["alarm_run"] = self.alarm_run
        self.log.append(out)
        return out
