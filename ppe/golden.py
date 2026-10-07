"""Frozen golden set: a manifest of files and sha256 hashes, and its check.

The golden set is human-labelled data that is never trained on, tuned on or
used to choose a threshold. It is the only data the verify gate judges a
candidate model on. A manifest is written once and never edited; a new
version gets a new file (golden_v2.csv, ...).

Run (from the repository root):
    python -m ppe.golden --verify golden/golden_v1.csv
    python -m ppe.golden --freeze golden/golden_v1.csv --images datasets/construction-ppe/images/test \
        --labels datasets/construction-ppe/labels/test        # only to create a NEW version
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from ppe.data import IMAGE_SUFFIXES, REPO_ROOT, sha256_file

FIELDS = ["kind", "path", "bytes", "sha256", "n_boxes"]


def _rel(p: Path) -> str:
    return p.resolve().relative_to(REPO_ROOT).as_posix()


def freeze(manifest: Path, images: Path, labels: Path) -> None:
    if manifest.exists():
        sys.exit(f"{manifest} exists. A golden manifest is never overwritten; write a new version instead.")
    rows = []
    class_counts: Counter = Counter()
    imgs = sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for img in imgs:
        lab = labels / f"{img.stem}.txt"
        if not lab.exists():
            sys.exit(f"No label file for {img}")
        lines = [ln.split() for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
        class_counts.update(int(ln[0]) for ln in lines)
        rows.append({"kind": "image", "path": _rel(img), "bytes": img.stat().st_size,
                     "sha256": sha256_file(img), "n_boxes": ""})
        rows.append({"kind": "label", "path": _rel(lab), "bytes": lab.stat().st_size,
                     "sha256": sha256_file(lab), "n_boxes": len(lines)})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Froze {len(imgs)} images to {_rel(manifest)}; boxes per class id: {dict(sorted(class_counts.items()))}")


def verify(manifest: Path, quiet: bool = False) -> int:
    """Check every file in the manifest. Returns the number of images; exits on any mismatch."""
    if not manifest.exists():
        sys.exit(f"Golden manifest not found: {manifest}")
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    problems = []
    for r in rows:
        p = REPO_ROOT / r["path"]
        if not p.exists():
            problems.append(f"missing {r['path']}")
        elif sha256_file(p) != r["sha256"]:
            problems.append(f"changed {r['path']}")
    # Files in the folders that are not in the manifest would silently join the set.
    for kind in ("image", "label"):
        dirs = {(REPO_ROOT / r["path"]).parent for r in rows if r["kind"] == kind}
        listed = {(REPO_ROOT / r["path"]).resolve() for r in rows if r["kind"] == kind}
        for d in dirs:
            for p in d.iterdir():
                if p.is_file() and p.resolve() not in listed:
                    problems.append(f"extra {_rel(p)}")
    if problems:
        sys.exit(f"Golden set {manifest.name} FAILED ({len(problems)} problems): " + "; ".join(problems[:10]))
    n_images = sum(r["kind"] == "image" for r in rows)
    if not quiet:
        print(f"Golden set {manifest.name} ok: {n_images} images, {len(rows) - n_images} label files, hashes match.")
    return n_images


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.golden", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", type=Path, metavar="MANIFEST")
    g.add_argument("--freeze", type=Path, metavar="MANIFEST")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--labels", type=Path)
    args = ap.parse_args(argv)
    if args.verify:
        verify(args.verify if args.verify.is_absolute() else REPO_ROOT / args.verify)
    else:
        if not (args.images and args.labels):
            ap.error("--freeze needs --images and --labels")
        freeze(args.freeze if args.freeze.is_absolute() else REPO_ROOT / args.freeze,
               REPO_ROOT / args.images, REPO_ROOT / args.labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
