"""Render the paper draft from results.csv and promotions.csv (Phase 6).

The draft is written as a template, paper/draft.md.tmpl, in which every measured number is a
placeholder that names the row it comes from. This module fills the placeholders from
results/results.csv and results/promotions.csv, writes paper/draft.md, and writes
paper/trace.csv with one line per number (rendered text, raw value, run id, n, command).
It refuses to render if a placeholder matches no row or more than one row, or if a decimal
number was typed into the template by hand.

    python -m ppe.paper                      # render paper/draft.md, paper/trace.csv, paper/flags.md
    python -m ppe.paper --check              # render to memory only; exit 1 on any problem
    python -m ppe.paper --find p1 test clean no_helmet    # list rows to cite (alias, then filters)

Placeholders (inside {{ }}):

    alias/split/condition/class/metric[@precision_mode][|fmt]
        one row of results.csv. alias maps to a run_id in paper/numbers.yaml.
        "*" in a field means "any" (still exactly one row must match).
    promo:scenario/column[/class][|fmt]
        one cell of promotions.csv; with a class, the per-class entry "cls=0.2121(7/33)".
    cfg:VALUE:path/to/config.yaml
        a configuration constant (not a result); checked to appear verbatim in that file.

Formats: .3f (default), .2f, .4f, .1f, signed (+0.123), abs (absolute value, .3f), int,
count (value x n, which must be a whole number), n (the row's n), frac (k/n from promotions),
text (verbatim).

Flags: any "[FLAG: ...]" in the template is collected, with its section, into paper/flags.md.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import yaml

from ppe.data import REPO_ROOT

PAPER = REPO_ROOT / "paper"
RESULTS = REPO_ROOT / "results" / "results.csv"
PROMOTIONS = REPO_ROOT / "results" / "promotions.csv"
FIELDS = ("split", "condition", "class", "metric")
PLACEHOLDER = re.compile(r"\{\{\s*(.+?)\s*\}\}")
FLAG = re.compile(r"\[FLAG: (.+?)\]", re.S)
# decimal numbers that may be typed by hand: metric names, software versions, the 95% interval level, section numbers
ALLOWED_DECIMALS = re.compile(r"mAP@0\.5(?::0\.95)?|\b\d+\.\d+\.\d+\b|Python 3\.10|\b95%|^#+ \d+\.\d+")
HAND_DECIMAL = re.compile(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])|\b\d+(?:\.\d+)?\s?%")


class PaperError(Exception):
    pass


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt_value(raw: str, n: str, fmt: str, key: str) -> str:
    if fmt == "text":
        return raw
    if fmt == "n":
        return str(int(float(n)))
    v = float(raw)
    if fmt == "count":
        c = v * float(n)
        if abs(c - round(c)) > 0.02:
            raise PaperError(f"{key}: value {raw} x n {n} = {c:.4f} is not a whole count")
        return str(int(round(c)))
    if fmt == "int":
        if v != int(v):
            raise PaperError(f"{key}: value {raw} is not an integer")
        return str(int(v))
    if fmt == "signed":
        s = f"{v:+.3f}"
        return "0.000" if s in ("+0.000", "-0.000") else s
    if fmt == "abs":
        return f"{abs(v):.3f}"
    if re.fullmatch(r"\.\df", fmt):
        return format(v, fmt)
    raise PaperError(f"{key}: unknown format {fmt!r}")


def resolve_result(spec: str, fmt: str, rows: list[dict], runs: dict) -> dict:
    mode = None
    if "@" in spec:
        spec, mode = spec.split("@", 1)
    parts = spec.split("/")
    if len(parts) != 5:
        raise PaperError(f"{spec}: need alias/split/condition/class/metric")
    alias, *vals = parts
    if alias not in runs:
        raise PaperError(f"{spec}: alias {alias!r} not in paper/numbers.yaml")
    run_id = runs[alias]
    hits = [r for r in rows if r["run_id"] == run_id
            and all(v == "*" or r[f] == v for f, v in zip(FIELDS, vals))
            and (mode is None or r["precision_mode"] == mode)]
    if len(hits) != 1:
        raise PaperError(f"{spec}{'@' + mode if mode else ''}: {len(hits)} rows match in {run_id} (need exactly 1)")
    r = hits[0]
    return {"rendered": fmt_value(r["value"], r["n"], fmt, spec), "raw": r["value"], "n": r["n"],
            "source": "results.csv", "run_id": run_id, "command": r["command"]}


PER_CLASS = re.compile(r"([\w]+)=([-\d.]+)\((\d+)/(\d+)\)")


def resolve_promo(spec: str, fmt: str, promos: list[dict]) -> dict:
    parts = spec.split("/")
    scen, col = parts[0], parts[1]
    hits = [p for p in promos if p["scenario"] == scen]
    if len(hits) != 1:
        raise PaperError(f"promo:{spec}: {len(hits)} promotions.csv rows for scenario {scen!r}")
    p = hits[0]
    if col not in p:
        raise PaperError(f"promo:{spec}: no column {col!r}")
    cell = p[col]
    n = ""
    if len(parts) == 3:
        found = {m[0]: m for m in PER_CLASS.findall(cell)}
        if parts[2] not in found:
            # "all=0.7130" has no counts
            m = re.search(rf"\b{re.escape(parts[2])}=([-\d.]+)", cell)
            if not m:
                raise PaperError(f"promo:{spec}: class {parts[2]!r} not in {col}")
            raw = m.group(1)
        else:
            _, raw, k, n = found[parts[2]]
            if fmt == "frac":
                return {"rendered": f"{k}/{n}", "raw": raw, "n": n, "source": "promotions.csv",
                        "run_id": p["run_id"], "command": p["command"]}
        cell = raw
    if fmt == "text":
        rendered = cell
    else:
        rendered = fmt_value(cell, n or "1", fmt, f"promo:{spec}")
    return {"rendered": rendered, "raw": cell, "n": n, "source": "promotions.csv",
            "run_id": p["run_id"], "command": p["command"]}


def resolve_cfg(spec: str) -> dict:
    value, path = spec.split(":", 1)
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    if not re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", text):
        raise PaperError(f"cfg:{spec}: {value} does not appear in {path}")
    return {"rendered": value, "raw": value, "n": "", "source": path, "run_id": "config", "command": ""}


def render(template: str, rows: list[dict], promos: list[dict], runs: dict) -> tuple[str, list[dict], list[str]]:
    trace: list[dict] = []
    errors: list[str] = []

    def sub(m: re.Match) -> str:
        body = m.group(1)
        spec, _, fmt = body.partition("|")
        fmt = fmt.strip() or ".3f"
        spec = spec.strip()
        try:
            if spec.startswith("promo:"):
                t = resolve_promo(spec[6:], fmt, promos)
            elif spec.startswith("cfg:"):
                t = resolve_cfg(spec[4:])
            else:
                t = resolve_result(spec, fmt, rows, runs)
        except (PaperError, OSError, ValueError) as e:
            errors.append(str(e))
            return "??"
        line = template.count("\n", 0, m.start()) + 1
        trace.append({"template_line": line, "placeholder": body, **t})
        return t["rendered"]

    out = PLACEHOLDER.sub(sub, template)
    # hand-typed decimals outside placeholders, flags and code spans
    bare = PLACEHOLDER.sub(" ", template)
    for i, line in enumerate(bare.splitlines(), 1):
        if line.lstrip().startswith(("<!--", "|---")):
            continue
        scrub = re.sub(r"`[^`]*`", " ", ALLOWED_DECIMALS.sub(" ", line))
        for m in HAND_DECIMAL.finditer(scrub):
            errors.append(f"template line {i}: hand-typed number {m.group(0).strip()!r} (use a placeholder)")
    return out, trace, errors


def flags(template: str) -> list[tuple[str, str]]:
    out = []
    section = ""
    for block in re.split(r"(?m)^(?=#)", template):
        head = block.splitlines()[0] if block.startswith("#") else ""
        if head.startswith("## ") or head.startswith("# "):
            section = head.lstrip("# ").strip()
        for m in FLAG.finditer(block):
            out.append((section, " ".join(m.group(1).split())))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ppe.paper", description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="render in memory only; exit 1 on any problem")
    ap.add_argument("--find", nargs="+", metavar="ALIAS_OR_FILTER",
                    help="alias, then any of split/condition/class/metric values to list matching rows")
    ap.add_argument("--template", default=str(PAPER / "draft.md.tmpl"))
    args = ap.parse_args(argv)
    runs = yaml.safe_load((PAPER / "numbers.yaml").read_text(encoding="utf-8"))["runs"]
    rows = load_rows(RESULTS)
    promos = load_rows(PROMOTIONS)

    if args.find:
        alias, *filters = args.find
        run_id = runs.get(alias, alias)
        for r in rows:
            if r["run_id"] == run_id and all(v in (r[f] for f in FIELDS) or v == r["precision_mode"] for v in filters):
                print("/".join([alias] + [r[f] for f in FIELDS]) + f"@{r['precision_mode']}", r["value"], f"n={r['n']}")
        return 0

    template = Path(args.template).read_text(encoding="utf-8")
    out, trace, errors = render(template, rows, promos, runs)
    fl = flags(template)
    if errors:
        print(f"{len(errors)} problem(s):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        return 1
    print(f"{len(trace)} numbers traced to results.csv / promotions.csv / configs; {len(fl)} flagged claims")
    if args.check:
        return 0
    (PAPER / "draft.md").write_text(
        "<!-- Generated by `python -m ppe.paper` from paper/draft.md.tmpl. Edit the template, not this file. -->\n\n" + out,
        encoding="utf-8")
    with (PAPER / "trace.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["template_line", "placeholder", "rendered", "raw", "n", "source", "run_id", "command"])
        w.writeheader()
        w.writerows(trace)
    lines = ["# Claims the data does not yet support", "",
             "Generated by `python -m ppe.paper` from the [FLAG: ...] markers in paper/draft.md.tmpl.", "",
             "| # | Section | Flag |", "|---:|---|---|"]
    lines += [f"| {i} | {s} | {t} |" for i, (s, t) in enumerate(fl, 1)]
    (PAPER / "flags.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote paper/draft.md, paper/trace.csv, paper/flags.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
