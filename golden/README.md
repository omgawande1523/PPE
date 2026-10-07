# golden/

Frozen golden sets. A manifest lists every image and label file with its size and sha256. A manifest is never edited; a new version is a new file.

| Version | Manifest | Content | Frozen |
|---|---|---|---|
| v1 | `golden_v1.csv` | Construction-PPE test split: 141 images, 1,251 boxes (helmet 192, gloves 163, vest 178, boots 211, goggles 52, none 65, Person 236, no_helmet 40, no_goggle 33, no_gloves 58, no_boots 23) | 2026-10-07 |

Rules: never train, tune, pick a threshold or select a checkpoint on a golden set. It is used only to report results and by the verify gate.

Check it before use (`ppe.eval` does this automatically for the test split):

```bash
python -m ppe.golden --verify golden/golden_v1.csv
```

v1 labels are the Ultralytics dataset's own human labels; they have not been re-audited. Phase 2 adds own-site images to a later version.
