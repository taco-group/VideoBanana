from __future__ import annotations

import json
from pathlib import Path

from fakes import FakeEditor, ScriptedClient
from test_editor import _client, _spec

from videobanana.conversation import InstructionWriter, MultiTurnVerifier
from videobanana.conversation.compiler import COMPILER_SYSTEM, IntentCompiler
from videobanana.editor import EditorTurnExecutor

INSTRUCTIONS = ["Make the cup white.", "Add a spoon on the saucer."]


def _intent(index: int) -> dict:
    return {"instruction_index": index, "text": f"Result {index}.", "target": "t", "operation": "recolor",
            "desired_state": "s", "preservation_constraints": []}


def test_an_answer_covering_every_instruction_is_accepted() -> None:
    client = ScriptedClient([{"status": "ok", "intents": [_intent(0), _intent(1)]}])
    result = IntentCompiler(client, model="agent").compile(INSTRUCTIONS)

    assert result.ok and len(result.intents) == 2
    call = client.calls[0]
    assert call["system_instruction"] == COMPILER_SYSTEM
    prompt = json.loads(call["prompt"])
    assert [item["text"] for item in prompt["instructions"]] == INSTRUCTIONS and prompt["repair_message"] is None


def test_an_answer_that_misses_an_instruction_is_asked_again_with_the_error() -> None:
    client = ScriptedClient([{"status": "ok", "intents": [_intent(0)]}, {"status": "ok", "intents": [_intent(0), _intent(1)]}])
    result = IntentCompiler(client, model="agent").compile(INSTRUCTIONS)

    assert result.ok
    assert "omitted source instruction indices [1]" in json.loads(client.calls[1]["prompt"])["repair_message"]


def test_an_ambiguous_request_is_blocked_and_three_invalid_answers_fail() -> None:
    blocked = IntentCompiler(ScriptedClient([{"status": "blocked_ambiguity", "reason": "which cup?"}]), model="agent")
    assert blocked.compile(INSTRUCTIONS).status == "blocked_ambiguity"

    client = ScriptedClient([{"status": "ok", "intents": []}] * 3)
    assert IntentCompiler(client, model="agent").compile(INSTRUCTIONS).status == "failed"
    assert len(client.calls) == 3


def test_a_blocked_attempt_makes_no_edit_and_the_next_attempt_runs(tiny_video: Path, tmp_path: Path) -> None:
    client, editor = _client([4]), FakeEditor()
    compiler_client = ScriptedClient([{"status": "blocked_ambiguity", "reason": "which cup?"},
                                      {"status": "ok", "intents": [_intent(0), _intent(1)]}])
    executor = EditorTurnExecutor(
        tmp_path / "turns",
        writer=InstructionWriter(client, model="agent"),
        verifier=MultiTurnVerifier(client, model="verifier"),
        editor=editor,
        compiler=IntentCompiler(compiler_client, model="agent"),
    )
    result = executor.run_turn(_spec(tiny_video))

    assert not result.candidates[0].structural_passed and result.candidates[1].structural_passed
    assert [request.seed for request in editor.requests] == [1]
    # The editor gets the written instructions, not the compiled intents.
    assert editor.requests[0].prompts == INSTRUCTIONS
    saved = json.loads((tmp_path / "turns/T01/A1/intent_compiler.json").read_text())
    assert saved["status"] == "blocked_ambiguity" and saved["instructions"] == INSTRUCTIONS
