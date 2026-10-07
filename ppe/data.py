"""Dataset download, checksum verification and run-time data YAML resolution.

Run (from the repository root):
    python -m ppe.data --download construction-ppe     # fetch, verify sha256, extract
    python -m ppe.data --check construction-ppe        # count images per split
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Pinned upstream archives. The sha256 was computed on 2026-10-07 from the
# Ultralytics release asset; a different hash means the dataset changed and
# every number in results.csv would no longer be comparable.
DATASETS = {
    "construction-ppe": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
        "sha256": "bef8dcb599aa4e9d9f5e602cb6fa7143d3c84d7f6a0ff40463d7f2a4c2632ccc",
        "dest": Path("datasets") / "construction-ppe",
        "config": Path("configs") / "data" / "construction_ppe.yaml",
        "splits": {"train": 1132, "val": 143, "test": 141},
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    """sha256 of a file, or of a directory (exported OpenVINO/NCNN/TFLite models) as sorted relative paths + bytes."""
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for p in sorted(q for q in path.rglob("*") if q.is_file()):
        h.update(p.relative_to(path).as_posix().encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


def download(name: str) -> Path:
    spec = DATASETS[name]
    dest = REPO_ROOT / spec["dest"]
    if (dest / "images").is_dir():
        print(f"{dest.relative_to(REPO_ROOT)} already exists; checking it instead of downloading.")
        check(name)
        return dest
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / f"{name}.zip"
        print(f"Downloading {spec['url']}")
        urllib.request.urlretrieve(spec["url"], archive)
        digest = sha256_file(archive)
        if digest != spec["sha256"]:
            sys.exit(f"sha256 mismatch for {name}: got {digest}, expected {spec['sha256']}. Not extracting.")
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    # The archive has images/, labels/, data.yaml and LICENSE at its root.
    print(f"Extracted to {dest.relative_to(REPO_ROOT)}")
    check(name)
    return dest


def check(name: str) -> None:
    spec = DATASETS[name]
    dest = REPO_ROOT / spec["dest"]
    bad = False
    for split, expected in spec["splits"].items():
        d = dest / "images" / split
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES) if d.is_dir() else 0
        flag = "ok" if n == expected else f"EXPECTED {expected}"
        bad |= n != expected
        print(f"{split:5s} {n:5d} images  {flag}")
    if bad:
        sys.exit(f"{name} is incomplete; run: python -m ppe.data --download {name}")


def resolve_data_yaml(config: str | Path, out_dir: Path) -> tuple[Path, dict]:
    """Write a copy of a data YAML whose 'path' is absolute, for Ultralytics.

    Committed configs keep a repository-relative path; Ultralytics would
    otherwise resolve it against its own global datasets directory.
    """
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    with cfg_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    root = Path(data["path"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    if not root.is_dir():
        sys.exit(f"Dataset not found at {root}. Run: python -m ppe.data --download construction-ppe")
    data["path"] = str(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "data.yaml"
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
    return out, data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.data", description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--download", choices=sorted(DATASETS))
    g.add_argument("--check", choices=sorted(DATASETS))
    args = ap.parse_args(argv)
    if args.download:
        download(args.download)
    else:
        check(args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
