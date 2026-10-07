"""PPE-to-person assignment and per-person compliance.

Rules (Phase 0 of the brief):
  * A PPE or no_* box belongs to the person box that contains its centre.
    If several person boxes contain the centre, the smallest one wins
    (the most specific enclosing person). Boxes with no enclosing person
    are returned as orphans.
  * Classes are matched by exact name, never by substring.
  * Per item, a no_* detection is a violation. A positive detection with
    no no_* detection is "present". Neither is "unknown", not a violation.
  * A person is "violation" if any required item is a violation,
    "compliant" if every required item is present, otherwise "unknown".
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The trained weights name this class "person" (lower case), not "Person" as in
# the brief's class table; matching is exact, so this must follow best.pt.
PERSON_CLASS = "person"

# Positive class -> negative class, exactly as named in Construction-PPE.
# vest has no negative class in the dataset, so it can only be present or unknown.
ITEM_CLASSES: dict[str, str | None] = {
    "helmet": "no_helmet",
    "goggles": "no_goggle",
    "gloves": "no_gloves",
    "boots": "no_boots",
    "vest": None,
}

NEGATIVE_TO_ITEM = {neg: item for item, neg in ITEM_CLASSES.items() if neg}
PPE_CLASSES = set(ITEM_CLASSES) | set(NEGATIVE_TO_ITEM)

PRESENT, VIOLATION, UNKNOWN = "present", "violation", "unknown"
COMPLIANT = "compliant"


@dataclass
class Box:
    """One detection. xyxy in pixels."""

    name: str
    conf: float
    xyxy: tuple[float, float, float, float]
    cls_id: int = -1
    track_id: int | None = None

    @property
    def centre(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    def contains(self, x: float, y: float) -> bool:
        x1, y1, x2, y2 = self.xyxy
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_dict(self) -> dict:
        return {
            "cls_id": self.cls_id,
            "name": self.name,
            "conf": round(float(self.conf), 4),
            "xyxy": [round(float(v), 1) for v in self.xyxy],
            "track_id": self.track_id,
        }


@dataclass
class PersonResult:
    person: Box
    boxes: list[Box] = field(default_factory=list)
    items: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    status: str = UNKNOWN

    def to_dict(self) -> dict:
        return {
            "track_id": self.person.track_id,
            "box": self.person.to_dict(),
            "items": self.items,
            "conflicts": self.conflicts,
            "status": self.status,
            "evidence": [b.to_dict() for b in self.boxes],
        }


def assign(persons: list[Box], ppe: list[Box]) -> tuple[list[list[Box]], list[Box]]:
    """Assign each PPE box to the smallest person box containing its centre.

    Returns (per-person box lists aligned with ``persons``, orphan boxes).
    """
    assigned: list[list[Box]] = [[] for _ in persons]
    orphans: list[Box] = []
    for box in ppe:
        cx, cy = box.centre
        candidates = [i for i, p in enumerate(persons) if p.contains(cx, cy)]
        if not candidates:
            orphans.append(box)
            continue
        best = min(candidates, key=lambda i: persons[i].area)
        assigned[best].append(box)
    return assigned, orphans


def item_states(boxes: list[Box], items: list[str]) -> tuple[dict[str, str], list[str]]:
    """Decide present / violation / unknown for each requested item.

    A no_* box is a violation even when a positive box for the same item is
    also assigned; that case is listed in ``conflicts`` so it can be audited.
    """
    names = {b.name for b in boxes}
    states: dict[str, str] = {}
    conflicts: list[str] = []
    for item in items:
        if item not in ITEM_CLASSES:
            raise ValueError(f"Unknown PPE item {item!r}; expected one of {sorted(ITEM_CLASSES)}")
        neg = ITEM_CLASSES[item]
        has_pos = item in names
        has_neg = neg is not None and neg in names
        if has_neg:
            states[item] = VIOLATION
            if has_pos:
                conflicts.append(item)
        elif has_pos:
            states[item] = PRESENT
        else:
            states[item] = UNKNOWN
    return states, conflicts


def person_status(states: dict[str, str]) -> str:
    values = states.values()
    if VIOLATION in values:
        return VIOLATION
    if values and all(v == PRESENT for v in values):
        return COMPLIANT
    return UNKNOWN


def evaluate_frame(boxes: list[Box], required_items: list[str]) -> tuple[list[PersonResult], list[Box]]:
    """Group a frame's detections by person and decide compliance per person.

    Classes outside Person and the PPE / no_* set (for example ``none``) are
    ignored here; they are still written to the runtime log by the caller.
    """
    persons = [b for b in boxes if b.name == PERSON_CLASS]
    ppe = [b for b in boxes if b.name in PPE_CLASSES]
    assigned, orphans = assign(persons, ppe)
    results = []
    for person, own in zip(persons, assigned):
        states, conflicts = item_states(own, required_items)
        results.append(PersonResult(person, own, states, conflicts, person_status(states)))
    return results, orphans
