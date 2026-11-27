from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeEditor, ScriptedClient, write_video

from videobanana.conversation import ConversationIntent, EditorInstruction, InstructionWriter, MultiTurnVerifier, Node, TurnSpec
from videobanana.conversation.instructions import INSTRUCTION_SYSTEM
from videobanana.editor import CoinVESettings, EditorTurnExecutor, EditRequest, coinve_argv, editor_prompts, frame_count


def _spec(entry_video: Path, *, turn: int = 1) -> TurnSpec:
    return TurnSpec(
        conversation_id="c",
        turn=turn,
        entry=Node(node_id="N0000", parent_id=None, video_path=str(entry_video), depth=0, turn=0),
        execute=[ConversationIntent(intent_id="I1", introduced_turn=1, text="The cup is white."),
                 ConversationIntent(intent_id="I2", introduced_turn=1, text="A spoon lies on the saucer.")],
        preserve=[],
        observe=[],
        max_attempts=2,
        max_instructions_per_call=5,
    )


def _client(if_scores: list[int]) -> ScriptedClient:
    """Instructions first, then one verdict per attempt with the given score for every intent."""

    def answer(call: dict[str, Any]) -> Any:
        if call["system_instruction"] == INSTRUCTION_SYSTEM:
            return {"instructions": [{"intent_ids": ["I1"], "text": " Make the cup white. "}, {"intent_ids": ["I2"], "text": "Add a spoon on the saucer."}]}
        score = if_scores.pop(0)
        return {"survey": "", "instructional_following": {"I1": {"score": score}, "I2": {"score": score}},
                "render_quality": {"score": 4}, "edit_exclusivity": {"score": 4}}

    return ScriptedClient(answer=answer)


def _executor(root: Path, client: ScriptedClient, editor: FakeEditor) -> EditorTurnExecutor:
    return EditorTurnExecutor(
        root,
        writer=InstructionWriter(client, model="agent"),
        verifier=MultiTurnVerifier(client, model="verifier"),
        editor=editor,
    )


def test_attempts_resample_with_the_next_seed_until_one_passes(tiny_video: Path, tmp_path: Path) -> None:
    client, editor = _client([2, 4]), FakeEditor()
    result = _executor(tmp_path / "turns", client, editor).run_turn(_spec(tiny_video))

    assert [item.candidate_id for item in result.candidates] == ["T01_A1", "T01_A2"]
    assert [request.seed for request in editor.requests] == [0, 1]
    assert editor.requests[0].prompts == ["Make the cup white.", "Add a spoon on the saucer."]
    assert editor.requests[0].input_video == str(tiny_video) and editor.requests[0].frames == 9
    # The verifier compares the output with the video the editor received.
    verify_calls = [call for call in client.calls if "media_paths" in call]
    assert verify_calls[1]["media_paths"] == [str(tiny_video), str((tmp_path / "turns/T01/A2/output.mp4").resolve())]
    assert result.candidates[0].instruction_scores == {"I1": 1, "I2": 1}
    saved = json.loads((tmp_path / "turns/T01/A2/multiturn_verdict.json").read_text())
    assert saved["verifier_identity"] == {"name": "multiturn-verifier", "model": "verifier", "media_fps": 3.92}


def test_a_failing_editor_is_retried_with_the_same_seed_then_the_attempt_fails(tiny_video: Path, tmp_path: Path) -> None:
    client, editor = _client([4]), FakeEditor(failures=3)
    result = _executor(tmp_path / "turns", client, editor).run_turn(_spec(tiny_video))

    assert [request.seed for request in editor.requests] == [0, 0, 0, 1]
    assert not result.candidates[0].structural_passed and result.candidates[1].structural_passed
    assert json.loads((tmp_path / "turns/T01/A1/edit.json").read_text())["status"] == "failed"


def test_a_resumed_turn_reuses_instructions_videos_and_verdicts(tiny_video: Path, tmp_path: Path) -> None:
    client, editor = _client([4]), FakeEditor()
    executor = _executor(tmp_path / "turns", client, editor)
    first = executor.run_turn(_spec(tiny_video))
    calls = len(client.calls)
    assert executor.run_turn(_spec(tiny_video)) == first
    assert len(client.calls) == calls and len(editor.requests) == 1

    # A turn planned differently does not reuse the old directory.
    changed = _spec(tiny_video).model_copy(update={"execute": _spec(tiny_video).execute[:1]})
    client.answer = lambda call: (
        {"instructions": [{"intent_ids": ["I1"], "text": "Make the cup white."}]}
        if call["system_instruction"] == INSTRUCTION_SYSTEM
        else {"survey": "", "instructional_following": {"I1": {"score": 4}}, "render_quality": {"score": 4}, "edit_exclusivity": {"score": 4}}
    )
    executor.run_turn(changed)
    assert len(list((tmp_path / "turns").glob("T01.superseded-*"))) == 1


def test_editor_prompts_are_each_instruction_once_in_order() -> None:
    instructions = [EditorInstruction(text="Add a hat. ", intent_ids=["I1"]), EditorInstruction(text="Add a hat. ", intent_ids=["I2"]),
                    EditorInstruction(text="Paint the wall red.", intent_ids=["I3"])]
    assert editor_prompts(instructions) == ["Add a hat.", "Paint the wall red."]


@pytest.mark.skipif(shutil.which("ffmpeg") is None and __import__("importlib").util.find_spec("imageio_ffmpeg") is None, reason="needs ffmpeg")
def test_a_long_entry_is_cut_to_its_first_49_frames(tmp_path: Path) -> None:
    long_video = write_video(tmp_path / "long.mp4", frames=60)
    client, editor = _client([4]), FakeEditor()
    _executor(tmp_path / "turns", client, editor).run_turn(_spec(long_video))
    prepared = tmp_path / "turns/T01/A1/normalized_input.mp4"
    assert editor.requests[0].input_video == str(prepared) and editor.requests[0].frames == 49
    assert frame_count(prepared) == 49


def test_coinve_arguments() -> None:
    settings = CoinVESettings(repo_root="/r", python="/py", composite_checkpoint="/c.safetensors", local_model_path="/base", mllm_model="/base/mllm")
    request = EditRequest(input_video="/v/in.mp4", prompts=["Add a hat.", "Paint the wall red."], seed=1, frames=49, output_path="/o/A2/output.mp4")
    assert coinve_argv(settings, request) == [
        "--src_video", "/v/in.mp4",
        "--prompts", "Add a hat.", "Paint the wall red.",
        "--composite_checkpoint", "/c.safetensors",
        "--local_model_path", "/base",
        "--mllm_model", "/base/mllm",
        "--output_dir", "/o/A2",
        "--output_name", "output",
        "--eval_max_pixels", "921600",
        "--eval_max_frame", "49",
        "--src_max_frames", "49",
        "--seed", "1",
        "--num_inference_steps", "50",
        "--no_side_by_side",
        "--save_instruction_masks",
    ]
    assert settings.entrypoint_path == "/r/infer_coinve_single.py"
