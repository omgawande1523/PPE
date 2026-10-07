# weights/

Promoted models; the last three are kept for rollback.

- `best.pt`: the Phase 0 baseline. YOLO11s, 11 classes, trained 50 epochs at 640 px with Ultralytics 8.3.21 (checkpoint metadata: version 8.3.21, date 2025-10-14).
  Its class names are `helmet, gloves, vest, boots, goggles, none, person, no_helmet, no_goggle, no_gloves, no_boots`; note `person` is lower case.

Other `*.pt` files are git-ignored until a model is promoted by the verify gate.
