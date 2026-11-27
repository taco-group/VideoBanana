from __future__ import annotations

import json
from typing import Any

import pytest
from fakes import ScriptedClient

from videobanana.conversation import ContractError, ConversationIntent
from videobanana.conversation.refine import IntentRefiner, parse_refine_response

INTENTS = {
    "I1": ConversationIntent(intent_id="I1", introduced_turn=1, text="The cup is white."),
    "I2": ConversationIntent(intent_id="I2", introduced_turn=2, text="A spoon lies on the saucer."),
    "I3": ConversationIntent(intent_id="I3", introduced_turn=3, text="The cup is blue.", replaces=["I1"]),
}
LIFECYCLE = {"I1": "inactive", "I2": "active", "I3": "active"}


def _answer(**fields: Any) -> dict[str, Any]:
    return {"new_intents": [], "deactivate": [], "reactivate": [], "constraints": [], "active": ["I2", "I3"], **fields}


def _parse(payload: Any, turn: int = 4):
    return parse_refine_response(payload, intents=INTENTS, lifecycle=LIFECYCLE, turn=turn)  # type: ignore[arg-type]


def test_undo_reactivates_and_new_intents_get_permanent_ids() -> None:
    update = _parse(
        _answer(
            new_intents=[{"key": "new1", "text": "Steam rises from the cup.", "replaces": []}],
            deactivate=[{"id": "I3", "reason": "user undid the blue change"}],
            reactivate=[{"id": "I1", "reason": "state right before the blue change"}],
            active=["I1", "I2", "new1"],
            constraints=["Keep the cup's shape."],
        )
    )
    assert update.active == ["I1", "I2", "I4"]
    assert [(item.intent_id, item.introduced_turn) for item in update.new_intents] == [("I4", 4)]
    assert update.constraints == ["Keep the cup's shape."]


def test_a_replaced_intent_counts_as_deactivated() -> None:
    update = _parse(_answer(new_intents=[{"key": "a", "text": "The cup is green.", "replaces": ["I3"]}], active=["I2", "a"]))
    assert update.active == ["I2", "I4"]
    assert update.new_intents[0].replaces == ["I3"]


def test_an_empty_replaces_may_be_left_out() -> None:
    update = _parse(_answer(new_intents=[{"key": "a", "text": "Steam rises from the cup."}], active=["I2", "I3", "a"]))
    assert update.new_intents[0].replaces == []


def test_an_answer_wrapped_in_a_one_element_list_is_accepted() -> None:
    assert _parse([_answer()]).active == ["I2", "I3"]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_answer(active=["I2", "I9"]), "unknown ids"),
        (_answer(deactivate=[{"id": "I1"}], active=["I2", "I3"]), "not active"),
        (_answer(reactivate=[{"id": "I2"}]), "not inactive"),
        (_answer(deactivate=[{"id": "I3"}], active=["I2", "I3"]), "does not match"),
        (_answer(new_intents=[{"key": "n", "text": "The cup is red.", "replaces": ["I3"]}], active=["I2", "I3", "n"]), "does not match"),
        (_answer(new_intents=[{"key": "n", "text": "The cup is red.", "replaces": ["I9"]}], active=["I2", "I3", "n"]), "replaces unknown"),
        ({"new_intents": [], "active": ["I2"], "constraints": []}, "fields must be exactly"),
    ],
)
def test_malformed_answers_are_rejected(payload: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        _parse(payload)


def test_refiner_sends_the_error_back_once() -> None:
    bad = _answer(active=["I2", "I9"])
    good = _answer(deactivate=[{"id": "I3", "reason": "white again"}], reactivate=[{"id": "I1", "reason": "white again"}], active=["I1", "I2"])
    client = ScriptedClient([bad, good])
    update = IntentRefiner(client, model="m").refine(["a", "b", "c", "make it white again"], INTENTS, LIFECYCLE)

    assert update.active == ["I1", "I2"]
    first, second = (json.loads(call["prompt"]) for call in client.calls)
    assert "previous_answer_error" not in first
    assert "unknown ids" in second["previous_answer_error"]
    assert first["recorded_intents"][2] == {"id": "I3", "text": "The cup is blue.", "status": "active", "replaces": ["I1"]}
    assert list(first["output_fields"]) == ["new_intents", "deactivate", "reactivate", "constraints", "active"]

    with pytest.raises(ContractError, match="after 2 attempts"):
        IntentRefiner(ScriptedClient([bad, bad]), model="m").refine(["a", "b", "c", "d"], INTENTS, LIFECYCLE)
