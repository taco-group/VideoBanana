"""Apply a full goal statement to the conversation's intent records.

The refiner restates the whole goal every turn: which intents hold now,
reusing earlier ids for goals that are unchanged or return to an earlier state
("undo the blue change" brings the white-casing intent back). Undo,
reactivation and objects that take their dependent edits with them are all
resolved there, so this module only checks that the statement is well formed
and derives the lifecycle from it.
"""

from __future__ import annotations

from .contracts import ContractError, ConversationIntent, GoalUpdate, Lifecycle


def apply_goal_update(
    intents: dict[str, ConversationIntent],
    update: GoalUpdate,
) -> tuple[dict[str, ConversationIntent], dict[str, Lifecycle]]:
    """Return the intents and lifecycle after one turn; the inputs are not modified."""
    _validate(intents, update)
    intents = {**intents, **{item.intent_id: item for item in update.new_intents}}
    active = set(update.active)
    lifecycle: dict[str, Lifecycle] = {
        intent_id: "active" if intent_id in active else "inactive" for intent_id in intents
    }
    return intents, lifecycle


def _validate(intents: dict[str, ConversationIntent], update: GoalUpdate) -> None:
    new_ids = [item.intent_id for item in update.new_intents]
    if len(new_ids) != len(set(new_ids)):
        raise ContractError(f"goal update repeats an intent id: {new_ids}")
    reused = sorted(set(new_ids) & set(intents))
    if reused:
        raise ContractError(f"intent ids already exist and records are immutable: {reused}")
    if len(update.active) != len(set(update.active)):
        raise ContractError(f"active list repeats an intent id: {update.active}")
    unknown = sorted(set(update.active) - set(intents) - set(new_ids))
    if unknown:
        raise ContractError(f"active list names unknown intents: {unknown}")
    missing = sorted(set(new_ids) - set(update.active))
    if missing:
        raise ContractError(f"new intents are missing from the active list: {missing}")
    for intent in update.new_intents:
        unknown = sorted(set(intent.replaces) - set(intents))
        if unknown:
            raise ContractError(f"{intent.intent_id} replaces unknown intents: {unknown}")
        still_active = sorted(set(intent.replaces) & set(update.active))
        if still_active:
            raise ContractError(f"{intent.intent_id} replaces intents that stay active: {still_active}")
