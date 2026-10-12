"""JSON-lines runtime log: one line per processed frame.

Each line holds every raw detection (class, confidence, box, track ID) plus
the per-person compliance decision, so the Phase 4 label-free signals can be
computed later from the log alone.

Note: this module is ``ppe.logging``. Code inside the package uses absolute
imports, so ``import logging`` still resolves to the standard library.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


class RunLogger:
    """Append-only JSONL writer. Use as a context manager."""

    def __init__(self, log_dir: str | Path, run_meta: dict):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = self.log_dir / f"run_{stamp}.jsonl"
        self._fh = None
        self.run_meta = run_meta

    def __enter__(self) -> "RunLogger":
        self._fh = self.path.open("a", encoding="utf-8")
        self._write({"type": "run_start", "schema": SCHEMA_VERSION, "time": _now(), **self.run_meta})
        return self

    def __exit__(self, *exc) -> None:
        self._write({"type": "run_end", "time": _now()})
        self._fh.close()
        self._fh = None

    def frame(
        self,
        frame_idx: int,
        source: str,
        shape: tuple[int, int],
        boxes: list,
        persons: list,
        orphans: list,
        infer_ms: float,
    ) -> None:
        self._write(
            {
                "type": "frame",
                "time": _now(),
                "frame_idx": frame_idx,
                "source": source,
                "height": shape[0],
                "width": shape[1],
                "infer_ms": round(infer_ms, 2),
                "boxes": [b.to_dict() for b in boxes],
                "persons": [p.to_dict() for p in persons],
                "orphans": [b.to_dict() for b in orphans],
            }
        )

    def _write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._fh.flush()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def read_log(path: str | Path):
    """Yield records from a JSONL run log."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)

