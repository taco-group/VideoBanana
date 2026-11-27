"""Checkpoint ledger reads/writes and per-turn intent roles.

What a checkpoint contains is decided only by verification. The ledger stores
one bit per (node, intent); a cell that was never judged reads as absent,
which is correct for every intent created after that node and fail-closed for
everything else.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import CandidateReport, ConversationState, TurnSpec


def realized(state: ConversationState, node_id: str, intent_id: str) -> bool:
    return state.ledger.get(node_id, {}).get(intent_id, False)


def exhausted_ids(state: ConversationState) -> list[str]:
    limit = state.config.exhaust_after_failures
    return [
        intent_id
        for intent_id, status in state.lifecycle.items()
        if status == "active" and state.failure_counts.get(intent_id, 0) >= limit
    ]


@dataclass(frozen=True)
class Roles:
    execute: list[str]
    preserve: list[str]
    observe: list[str]
    exhausted: list[str]
    #: Inactive intents still shown by the entry that an execute intent
    #: overwrites (the white casing under "make it dark blue").
    overwritten: list[str]
    #: Inactive intents still shown by the entry that nothing overwrites. An
    #: entry with any of them is never chosen: removing an edit is far harder
    #: for the editor than never having made it.
    unwanted: list[str]


def compute_roles(state: ConversationState, entry_id: str, exhausted: list[str]) -> Roles:
    """Split the goal against one candidate entry.

    Every active, non-exhausted intent lands in exactly one of execute or
    preserve, so the gate covers the same set whichever entry is chosen.
    """
    order = _turn_order(state)
    exhausted_set = set(exhausted)
    active = [intent_id for intent_id in order if state.lifecycle[intent_id] == "active"]
    targets = [intent_id for intent_id in active if intent_id not in exhausted_set]
    execute = [intent_id for intent_id in targets if not realized(state, entry_id, intent_id)]
    preserve = [intent_id for intent_id in targets if realized(state, entry_id, intent_id)]

    # Executing an intent overwrites whatever it replaces on the same element
    # ("make it blue instead" over a white casing is an ordinary edit).
    replaced = {value for intent_id in execute for value in state.intents[intent_id].replaces}
    shown_inactive = [
        intent_id
        for intent_id in order
        if state.lifecycle[intent_id] != "active" and realized(state, entry_id, intent_id)
    ]

    capacity = min(state.config.observe_fill_total, state.config.max_instructions_per_call) - len(execute)
    candidates = sorted(
        (intent_id for intent_id in active if intent_id in exhausted_set),
        key=lambda intent_id: (state.failure_counts.get(intent_id, 0), order.index(intent_id)),
    )
    return Roles(
        execute=execute,
        preserve=preserve,
        observe=candidates[: max(capacity, 0)],
        exhausted=list(exhausted),
        overwritten=[intent_id for intent_id in shown_inactive if intent_id in replaced],
        unwanted=[intent_id for intent_id in shown_inactive if intent_id not in replaced],
    )


def output_ledger(
    state: ConversationState, spec: TurnSpec, candidate: CandidateReport
) -> tuple[dict[str, bool], dict[str, int | None]]:
    """Ledger bits and scores for a committed candidate.

    Judged intents take their new absolute score (preserve intents too: an
    earlier edit that is now weak is absent from this node, whether or not the
    turn is blamed for it); everything else is inherited from the entry.
    """
    floor = state.config.item_floor
    row = dict(state.ledger.get(spec.entry.node_id, {}))
    scores = dict(state.scores.get(spec.entry.node_id, {}))

    def passed(score: int | None) -> bool:
        return score is not None and score >= floor

    for intent in [*spec.execute, *spec.observe]:
        score = candidate.instruction_scores.get(intent.intent_id)
        row[intent.intent_id], scores[intent.intent_id] = passed(score), score
    for intent in spec.preserve:
        score = candidate.retention_scores.get(intent.intent_id)
        row[intent.intent_id], scores[intent.intent_id] = passed(score), score
    # An intent is not present where one that replaces it is.
    for intent_id, intent in state.intents.items():
        if row.get(intent_id):
            for replaced in intent.replaces:
                row[replaced], scores[replaced] = False, None
    return row, scores


def _turn_order(state: ConversationState) -> list[str]:
    return sorted(state.intents, key=lambda intent_id: (state.intents[intent_id].introduced_turn, intent_id))
