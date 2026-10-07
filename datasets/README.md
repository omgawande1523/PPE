# datasets/ (git-ignored)

Expected layout, all paths relative to the repository root:

- `datasets/construction-ppe/`: Ultralytics Construction-PPE (AGPL-3.0), YOLO format, train/val/test (1132/143/141 images).
  Get it with `python -m ppe.data --download construction-ppe`; the archive's sha256 is pinned in `ppe/data.py`. The split folders are the release's own; nothing is re-split.
- `datasets/sh17/`: SH17 (8,099 manufacturing images, CC BY-NC-SA 4.0), only on Kaggle
  (`mugheesahmad/sh17-dataset-for-ppe-detection`). Download and unzip it anywhere, then
  `python -m ppe.shift --convert sh17 --src path/to/sh17`. By default only SH17's own validation list
  (`val_files.txt`, 1,620 images) is converted, so its training images stay free for later adaptation.
  Mapping: person, helmet, safety-vest to vest, gloves, shoes to boots. Glasses are excluded; every other class is dropped.
- `datasets/chv/`: CHV (1,330 construction images), Google Drive link in github.com/ZijianWang-ZW/PPE_detection.
  Unzip, then `python -m ppe.shift --convert chv --src path/to/chv`. Mapping: person, vest, and the four helmet colours to helmet.
  If the archive has no class-names file, pass `--names` with its class order.
- `datasets/own_site/`: own footage for Phase 2, labelled with the 11 Construction-PPE classes, in `images/test` and `labels/test`.

The converters write `manifest.csv` (source image sha256) and `conversion.json` (kept and dropped instances per class)
next to the converted images. `python -m ppe.shift --config configs/shift/p2_shift_yolo11s.yaml` evaluates every
set that is on disk and says which are missing.

Nothing in this folder is committed. The frozen golden set is recorded as file lists and hashes in `golden/`.
