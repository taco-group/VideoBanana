"""Turn gate and fallback candidate selection.

The hard gate alone decides whether a turn passes. The composite score is
only used to pick one candidate when none passes, so that the conversation
continues from the least-bad output instead of stopping.

Every intent is scored on the same absolute 0-3 scale. For the edits this
turn had to make (execute) the gate needs the floor. For earlier edits the
entry already showed (preserve) it needs "not damaged": the score may not
drop by ``damage_drop`` levels or more from the entry node's score, so an edit
that was already weak before is not this turn's failure. Observe intents
count towards instruction following in the composite but never gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from .contracts import CandidateReport, TurnResult, TurnSpec


@dataclass(frozen=True)
class Selection:
    candidate: CandidateReport | None
    gate_passed: bool
    primary: float | None
    secondary: float | None


def drop(spec: TurnSpec, candidate: CandidateReport, intent_id: str, floor: int) -> int:
    """Levels an earlier edit lost this turn (0 when it held or improved).

    Unknown scores fail closed: an unknown new score counts as 0, and an entry
    without a recorded score as the floor.
    """
    reference = spec.reference_scores.get(intent_id)
    reference = floor if reference is None else reference
    return max(0, reference - (candidate.retention_scores.get(intent_id) or 0))


def damaged(spec: TurnSpec, candidate: CandidateReport, intent_id: str, floor: int) -> bool:
    return drop(spec, candidate, intent_id, floor) >= spec.damage_drop


def passes_gate(candidate: CandidateReport, spec: TurnSpec, floor: int, item_floor: int | None = None) -> bool:
    """Floor on this turn's edits, no damage to earlier ones; observe intents never gate.

    ``item_floor`` (default ``floor``) applies to the edit scores, ``floor`` to
    non-target content and render quality.
    """
    if not candidate.structural_passed:
        return False
    item_floor = floor if item_floor is None else item_floor
    items = [candidate.instruction_scores.get(item.intent_id) for item in spec.execute]
    if not all(score is not None and score >= item_floor for score in items):
        return False
    if not all(score is not None and score >= floor for score in (candidate.nontarget_score, candidate.quality_score)):
        return False
    return not any(damaged(spec, candidate, item.intent_id, floor) for item in spec.preserve)


def composite(
    *,
    instruction: Sequence[int | None],
    retention: Sequence[tuple[int | None, int | None]],
    nontarget: int | None,
    quality: int | None,
    floor: int,
    damage_drop: int,
) -> tuple[float, float]:
    """(primary, secondary) with equal weight on instruction following, preservation and quality.

    ``retention`` holds (reference, new) scores of earlier edits. Each part is
    averaged over its own items first, so the number of instructions does not
    weight a part. Preservation splits evenly between earlier edits and
    non-target content; an earlier edit scores 1 unless damaged in primary,
    and 1 - drop/3 in secondary. Primary clips every other item at the gate
    floor -- surplus on one item cannot buy back a gap on another, and primary
    equals 1 exactly when the gate passes on every item. Secondary is unclipped
    and only breaks primary ties. Unknown scores count as 0; an earlier edit
    without a reference score counts as reaching the floor.
    """

    def lost(reference: int | None, new: int | None) -> int:
        return max(0, (floor if reference is None else reference) - (new or 0))

    def part(scores: Sequence[int | None], cap: float) -> float | None:
        if not scores:
            return None
        return mean(min(score or 0, cap) / cap for score in scores)

    def kept(clipped: bool) -> float | None:
        if not retention:
            return None
        if clipped:
            return mean(0.0 if lost(*pair) >= damage_drop else 1.0 for pair in retention)
        return mean(1.0 - lost(*pair) / 3.0 for pair in retention)

    def combine(cap: float) -> float:
        preservation = mean(value for value in (kept(clipped=cap < 3.0), part([nontarget], cap)) if value is not None)
        parts = [value for value in (part(instruction, cap), preservation, part([quality], cap)) if value is not None]
        return mean(parts)

    return combine(float(floor) if floor > 0 else 3.0), combine(3.0)


def composite_scores(candidate: CandidateReport, spec: TurnSpec, floor: int) -> tuple[float, float]:
    """:func:`composite` of one candidate.

    Instruction following covers every intent sent to the editor (execute and
    observe); preservation covers the preserve intents against the entry's
    reference scores, plus non-target content.
    """
    return composite(
        instruction=[candidate.instruction_scores.get(item.intent_id) for item in [*spec.execute, *spec.observe]],
        retention=[
            (spec.reference_scores.get(item.intent_id), candidate.retention_scores.get(item.intent_id))
            for item in spec.preserve
        ],
        nontarget=candidate.nontarget_score,
        quality=candidate.quality_score,
        floor=floor,
        damage_drop=spec.damage_drop,
    )


def select_candidate(result: TurnResult, spec: TurnSpec, floor: int, item_floor: int | None = None) -> Selection:
    valid = [item for item in result.candidates if item.structural_passed]
    if not valid:
        return Selection(candidate=None, gate_passed=False, primary=None, secondary=None)
    passing = [item for item in valid if passes_gate(item, spec, floor, item_floor)]
    pool = passing or valid
    # Earlier attempts win ties: ``max`` keeps the first of equal keys.
    best = max(pool, key=lambda item: composite_scores(item, spec, floor))
    primary, secondary = composite_scores(best, spec, floor)
    return Selection(candidate=best, gate_passed=bool(passing), primary=primary, secondary=secondary)
