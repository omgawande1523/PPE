# Paper draft (Phase 6)

| File | What it is |
|---|---|
| `draft.md.tmpl` | The draft, section by section from the brief's outline. Every measured number is a placeholder naming its row. Edit this file. |
| `numbers.yaml` | Short names for the run ids the draft cites. |
| `draft.md` | The rendered draft (generated; do not edit). |
| `trace.csv` | One line per number in the draft: placeholder, rendered text, stored value, n, run id, and the command that regenerates the row. |
| `flags.md` | Every claim the data does not yet support, collected from the `[FLAG: ...]` markers. |

```bash
python -m ppe.paper            # re-render draft.md, trace.csv and flags.md from results/results.csv and results/promotions.csv
python -m ppe.paper --check    # exit 1 if any placeholder matches no row or several rows, or a decimal was typed by hand
python -m ppe.paper --find p2 test low_light_s3 no_goggle   # list citable rows of a run
```

When a missing result is measured (results/missing.md), re-run its command, point `numbers.yaml` at the new run id if it changed,
write the paragraph with placeholders, remove the flag it answers, and re-render.
