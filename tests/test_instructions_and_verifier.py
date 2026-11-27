from __future__ import annotations

import json
from typing import Any

import pytest
from fakes import ScriptedClient

from videobanana.conversation import ContractError, ConversationIntent, Node, TurnSpec
from videobanana.conversation.instructions import InstructionWriter, parse_instructions
from videobanana.conversation.verifier import VERIFY_SYSTEM, MultiTurnVerifier, parse_verdict


def _intent(intent_id: str, turn: int, text: str) -> ConversationIntent:
    return ConversationIntent(intent_id=intent_id, introduced_turn=turn, text=text)


WRITER_SPEC = TurnSpec(
    conversation_id="c",
    turn=3,
    entry=Node(node_id="N0000", parent_id=None, video_path="source.mp4", depth=0, turn=0),
    execute=[_intent("I2", 2, "Red roses stand on the counter."), _intent("I3", 3, "The grinder casing is dark blue.")],
    preserve=[],
    observe=[],
    user_requests={"I2": "Now add red roses.", "I3": "Actually, make the casing dark blue instead."},
    max_attempts=2,
    max_instructions_per_call=5,
)

VERIFY_SPEC = TurnSpec(
    conversation_id="c",
    turn=3,
    entry=Node(node_id="N0002", parent_id="N0001", video_path="entry.mp4", depth=2, turn=2),
    execute=[_intent("I3", 3, "Golden particles drift above the counter.")],
    preserve=[_intent("I1", 1, "The casing is white."), _intent("I2", 2, "Red roses stand on the counter.")],
    observe=[],
    max_attempts=2,
    max_instructions_per_call=5,
)


def test_instructions_must_cover_every_target_once() -> None:
    good = {"instructions": [{"intent_ids": ["I2"], "text": "Add red roses on the counter."}, {"intent_ids": ["I3"], "text": "Change the black casing to dark blue."}]}
    assert [item.intent_ids for item in parse_instructions(good, WRITER_SPEC)] == [["I2"], ["I3"]]
    with pytest.raises(ContractError, match="exactly once"):
        parse_instructions({"instructions": [{"intent_ids": ["I2"], "text": "Add red roses."}]}, WRITER_SPEC)
    with pytest.raises(ContractError, match="exactly once"):
        parse_instructions({"instructions": [{"intent_ids": ["I2", "I3", "I3"], "text": "x"}]}, WRITER_SPEC)


def test_at_most_the_call_limit_of_instructions_is_accepted() -> None:
    spec = WRITER_SPEC.model_copy(update={"max_instructions_per_call": 1})
    two = {"instructions": [{"intent_ids": ["I2"], "text": "a"}, {"intent_ids": ["I3"], "text": "b"}]}
    with pytest.raises(ContractError, match="at most 1"):
        parse_instructions(two, spec)


def test_writer_shows_the_entry_context_and_retries_once() -> None:
    merged = {"instructions": [{"intent_ids": ["I2", "I3"], "text": "Add red roses and change the casing to dark blue."}]}
    client = ScriptedClient([{"instructions": []}, merged])
    assert len(InstructionWriter(client, model="m").write(WRITER_SPEC)) == 1
    first, second = (json.loads(call["prompt"]) for call in client.calls)
    assert first["starting_video"] == "the original source video"
    assert first["targets"][1] == {"id": "I3", "target_state": "The grinder casing is dark blue.", "user_words": "Actually, make the casing dark blue instead."}
    assert first["edits_to_remove"] == []
    assert "exactly once" in second["previous_answer_error"]


def _item(score: int) -> dict[str, Any]:
    return {"score": score, "explanation": "..."}


def _verdict(**overrides: Any) -> dict[str, Any]:
    answer = {
        "survey": "...",
        "instructional_following": {"I3": _item(4), "I1": _item(4), "I2": _item(2)},
        "render_quality": _item(4),
        "edit_exclusivity": _item(3),
    }
    return {**answer, **overrides}


def test_scores_are_stored_on_the_zero_to_three_scale() -> None:
    scores = parse_verdict(_verdict(), VERIFY_SPEC)
    assert scores.instruction_scores == {"I3": 3}
    assert scores.retention_scores == {"I1": 3, "I2": 1}
    assert (scores.nontarget_score, scores.quality_score) == (2, 3)


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (_verdict(instructional_following={"I3": _item(4), "I1": _item(4)}), "must score exactly"),
        (_verdict(instructional_following={"I3": _item(5), "I1": _item(4), "I2": _item(2)}), "score must be 1-4"),
        (_verdict(edit_exclusivity={"explanation": "x"}), "integer score"),
    ],
)
def test_incomplete_or_invalid_verdicts_are_rejected(answer: dict[str, Any], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        parse_verdict(answer, VERIFY_SPEC)


def test_verifier_compares_with_the_entry_and_fails_closed() -> None:
    client = ScriptedClient([_verdict()])
    MultiTurnVerifier(client, model="m").score_pair(VERIFY_SPEC, "entry.mp4", "candidate.mp4")
    call = client.calls[0]
    assert call["media_paths"] == ["entry.mp4", "candidate.mp4"]
    assert call["media_fps"] == 3.92
    prompt = json.loads(call["prompt"])
    # Earlier edits are listed like this turn's targets.
    assert [item["id"] for item in prompt["intents"]] == ["I3", "I1", "I2"]
    assert prompt["undo"] == []
    assert call["system_instruction"].startswith("You are an expert evaluator assessing the quality of AI-generated video edits.")

    broken = MultiTurnVerifier(ScriptedClient([{"survey": "x"}, {"survey": "x"}]), model="m").score_pair(VERIFY_SPEC, "entry.mp4", "candidate.mp4")
    assert broken.instruction_scores == {"I3": None} and broken.retention_scores == {"I1": None, "I2": None}
    assert broken.nontarget_score is None


def test_multiturn_prompt_does_not_call_video_1_the_original() -> None:
    assert "The Original Video (before editing)" not in VERIFY_SYSTEM
    assert "may already contain edits from earlier steps" in VERIFY_SYSTEM.split("---")[0]
    assert "or no change occurred at all" not in VERIFY_SYSTEM
    assert "Whether Video 2 differs from Video 1 does not decide this score." in VERIFY_SYSTEM
