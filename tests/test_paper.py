"""Checks for the paper renderer: placeholders resolve to exactly one row, hand-typed numbers are refused.

Run: python -m pytest tests
"""

from pathlib import Path

import pytest

from ppe import paper

ROWS = [
    {"run_id": "r1", "split": "test", "condition": "clean", "class": "no_helmet", "metric": "recall",
     "precision_mode": "fp32", "value": "0.225000", "n": "40", "command": "python -m ppe.eval"},
    {"run_id": "r1", "split": "test", "condition": "clean", "class": "all", "metric": "recall",
     "precision_mode": "fp32", "value": "0.715428", "n": "1251", "command": "python -m ppe.eval"},
]
PROMOS = [{"scenario": "site_shift", "run_id": "p5", "command": "python -m ppe.loop", "decision": "rejected",
           "candidate_recall": "no_goggle=0.2121(7/33);all=0.7130", "map50_tolerance": "0.02"}]
RUNS = {"p1": "r1"}


def render(t):
    return paper.render(t, ROWS, PROMOS, RUNS)


def test_formats_and_trace():
    out, trace, errors = render("{{p1/test/clean/no_helmet/recall}} {{p1/test/clean/no_helmet/recall|count}}/"
                                "{{p1/test/clean/no_helmet/recall|n}} {{promo:site_shift/candidate_recall/no_goggle|frac}} "
                                "{{promo:site_shift/candidate_recall/all|.4f}} {{promo:site_shift/decision|text}}")
    assert errors == []
    assert out == "0.225 9/40 7/33 0.7130 rejected"
    assert trace[0]["raw"] == "0.225000" and trace[0]["run_id"] == "r1"


def test_ambiguous_or_missing_row_is_an_error():
    _, _, errors = render("{{p1/test/clean/*/recall}} {{p1/test/clean/vest/recall}}")
    assert len(errors) == 2


def test_hand_typed_decimal_is_refused_but_allowed_forms_pass():
    _, _, errors = render("recall 0.715 and 12% here")
    assert len(errors) == 2
    _, _, errors = render("## 3.1 Setup\nmAP@0.5:0.95, Ultralytics 8.3.21, 95% intervals, `arXiv:1903.12261`")
    assert errors == []


def test_count_must_be_whole():
    with pytest.raises(paper.PaperError):
        paper.fmt_value("0.5", "3", "count", "k")


def test_repository_draft_renders():
    if not (paper.RESULTS.exists() and (paper.PAPER / "draft.md.tmpl").exists()):
        pytest.skip("results or template missing")
    import yaml
    runs = yaml.safe_load((paper.PAPER / "numbers.yaml").read_text(encoding="utf-8"))["runs"]
    _, trace, errors = paper.render((paper.PAPER / "draft.md.tmpl").read_text(encoding="utf-8"),
                                    paper.load_rows(paper.RESULTS), paper.load_rows(paper.PROMOTIONS), runs)
    assert errors == []
    assert len(trace) > 0
