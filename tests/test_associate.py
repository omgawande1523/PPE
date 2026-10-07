"""Checks for the compliance rules that the old main.py / app.py got wrong.

Run: python -m pytest tests
"""

from ppe.associate import COMPLIANT, PRESENT, UNKNOWN, VIOLATION, Box, evaluate_frame

ALL = ["helmet", "vest", "gloves", "boots", "goggles"]


def person(x1, y1, x2, y2, tid=None):
    return Box("person", 0.9, (x1, y1, x2, y2), track_id=tid)


def item(name, x, y):
    return Box(name, 0.8, (x - 5, y - 5, x + 5, y + 5))


def test_no_helmet_is_violation_not_helmet_present():
    # Problem 2: substring matching counted no_helmet as a helmet.
    persons, _ = evaluate_frame([person(0, 0, 100, 200), item("no_helmet", 50, 10)], ["helmet"])
    assert persons[0].items == {"helmet": VIOLATION}
    assert persons[0].status == VIOLATION


def test_missing_item_is_unknown_not_violation():
    # Problem 5: a model miss must not become a false violation.
    persons, _ = evaluate_frame([person(0, 0, 100, 200)], ALL)
    assert set(persons[0].items.values()) == {UNKNOWN}
    assert persons[0].status == UNKNOWN


def test_goggles_class_name():
    # Problem 3: the class is goggles, not safety_glasses.
    persons, _ = evaluate_frame([person(0, 0, 100, 200), item("goggles", 50, 20)], ["goggles"])
    assert persons[0].items == {"goggles": PRESENT}
    persons, _ = evaluate_frame([person(0, 0, 100, 200), item("no_goggle", 50, 20)], ["goggles"])
    assert persons[0].items == {"goggles": VIOLATION}


def test_one_helmet_does_not_clear_everyone():
    # Problem 4: per-person logic.
    boxes = [person(0, 0, 100, 200, 1), person(200, 0, 300, 200, 2), item("helmet", 50, 10)]
    persons, orphans = evaluate_frame(boxes, ["helmet"])
    assert [p.items["helmet"] for p in persons] == [PRESENT, UNKNOWN]
    assert [p.status for p in persons] == [COMPLIANT, UNKNOWN]
    assert orphans == []


def test_all_present_is_compliant():
    boxes = [person(0, 0, 100, 200)] + [item(n, 50, 50) for n in ALL]
    persons, _ = evaluate_frame(boxes, ALL)
    assert persons[0].status == COMPLIANT


def test_negative_wins_and_is_flagged_as_conflict():
    boxes = [person(0, 0, 100, 200), item("gloves", 20, 100), item("no_gloves", 80, 100)]
    persons, _ = evaluate_frame(boxes, ["gloves"])
    assert persons[0].items == {"gloves": VIOLATION}
    assert persons[0].conflicts == ["gloves"]


def test_centre_rule_smallest_person_and_orphans():
    big, small = person(0, 0, 300, 300, 1), person(100, 100, 200, 300, 2)
    boxes = [big, small, item("no_boots", 150, 290), item("helmet", 500, 500)]
    persons, orphans = evaluate_frame(boxes, ["boots"])
    assert persons[0].items == {"boots": UNKNOWN}
    assert persons[1].items == {"boots": VIOLATION}
    assert [o.name for o in orphans] == ["helmet"]


def test_none_class_is_ignored():
    persons, orphans = evaluate_frame([person(0, 0, 100, 200), item("none", 50, 50)], ["helmet"])
    assert persons[0].boxes == [] and orphans == []
