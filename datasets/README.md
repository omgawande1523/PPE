# datasets/ (git-ignored)

Expected layout, all paths relative to the repository root:

- `datasets/construction-ppe/`: Ultralytics Construction-PPE (AGPL-3.0), YOLO format, train/val/test (1132/143/141 images).
  Get it with `python -m ppe.data --download construction-ppe`; the archive's sha256 is pinned in `ppe/data.py`. The split folders are the release's own; nothing is re-split.
- `datasets/sh17/`, `datasets/chv/`: external sets for Phase 2.
- `datasets/own_site/`: own footage for Phase 2.

Nothing in this folder is committed. The frozen golden set is recorded as file lists and hashes in `golden/`.
