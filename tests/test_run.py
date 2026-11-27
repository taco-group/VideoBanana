"""A three-turn session end to end through ``run_session``, with a scripted model and a fake editor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from fakes import FakeEditor, ScriptedClient, write_video

from videobanana.conversation import ConversationConfig
from videobanana.conversation.compiler import COMPILER_SYSTEM
from videobanana.conversation.instructions import INSTRUCTION_SYSTEM
from videobanana.conversation.refine import REFINE_SYSTEM
from videobanana.conversation.verifier import VERIFY_SYSTEM
from videobanana.run import AgentSettings, agent_settings, load_config, run_session

TURNS = ["Put a red hat on the man.", "Add a coffee cup on the table.", "Remove the cup again."]
#: The refiner's answer to each turn.
GOALS = {
    1: {"new_intents": [{"key": "n1", "text": "The man wears a red hat."}], "deactivate": [], "reactivate": [], "constraints": [], "active": ["n1"]},
    2: {"new_intents": [{"key": "n1", "text": "A coffee cup stands on the table."}], "deactivate": [], "reactivate": [], "constraints": [], "active": ["I1", "n1"]},
    3: {"new_intents": [], "deactivate": [{"id": "I2", "reason": "removed"}], "reactivate": [], "constraints": [], "active": ["I1"]},
}


def _answer(call: dict[str, Any]) -> Any:
    prompt = json.loads(call["prompt"])
    if call["system_instruction"] == REFINE_SYSTEM:
        return GOALS[prompt["latest_turn"]]
    if call["system_instruction"] == COMPILER_SYSTEM:
        return {"status": "ok", "intents": [{"instruction_index": item["instruction_index"], "text": item["text"], "target": "t",
                                             "operation": "edit", "desired_state": "s"} for item in prompt["instructions"]]}
    if call["system_instruction"] == INSTRUCTION_SYSTEM:
        return {"instructions": [{"intent_ids": [item["id"]], "text": f"Edit: {item['target_state']}"} for item in prompt["targets"]]}
    assert call["system_instruction"] == VERIFY_SYSTEM
    scores = {item["id"]: {"score": 4, "explanation": ""} for item in prompt["intents"]}
    return {"survey": "", "instructional_following": scores, "render_quality": {"score": 4}, "edit_exclusivity": {"score": 3}}


def _bench(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    write_video(bench / "clips" / "s1.mp4")
    sessions = [{"session_id": "s1", "clip": "clips/s1.mp4", "turns": [{"turn": n, "request": text} for n, text in enumerate(TURNS, start=1)]}]
    (bench / "sessions.json").write_text(json.dumps(sessions), encoding="utf-8")
    return bench


def test_a_session_runs_end_to_end_and_resumes(tmp_path: Path) -> None:
    bench, out = _bench(tmp_path), tmp_path / "runs"
    client, editor = ScriptedClient(answer=_answer), FakeEditor()
    settings = AgentSettings(agent_model="agent", verifier_model="verifier")
    session = json.loads((bench / "sessions.json").read_text())[0]

    assert run_session(session, bench=bench, out=out, client=client, editor=editor, settings=settings)["status"] == "done"

    root = out / "s1"
    state = json.loads((root / "conversation.json").read_text())
    turns = state["turns"]
    # T2 redoes the hat together with the cup from the source; T3 reuses the T1 output.
    assert [(t["entry_id"], t["execute_ids"], t["output_id"], t["selection"]) for t in turns] == [
        ("N0000", ["I1"], "N0001", "gate_pass"),
        ("N0000", ["I1", "I2"], "N0002", "gate_pass"),
        ("N0001", [], "N0001", "reuse_entry"),
    ]
    assert state["ledger"]["N0002"] == {"I1": True, "I2": True}
    assert [request.prompts for request in editor.requests] == [
        ["Edit: The man wears a red hat."],
        ["Edit: The man wears a red hat.", "Edit: A coffee cup stands on the table."],
    ]
    assert all(request.seed == 0 for request in editor.requests)
    for number, node in ((1, "N0001"), (2, "N0002"), (3, "N0001")):
        exported = root / f"turn{number:02d}.mp4"
        assert exported.read_bytes() == Path(state["nodes"][node]["video_path"]).read_bytes()
    assert sorted(path.name for path in (root / "raw").rglob("*.json")) == [
        "T01_00.json", "T01_00.json", "T01_A1_0.json", "T01_A1_001.json", "T02_00.json", "T02_00.json", "T02_A1_0.json",
        "T02_A1_002.json", "T03_00.json",
    ]
    assert json.loads((root / "turns/T01/A1/intent_compiler.json").read_text())["status"] == "ok"
    assert (root / "turns/T02/A1/output.mp4").is_file() and (root / "turns/T02/instructions.json").is_file()

    # A second run finds every turn done and calls nothing.
    calls = len(client.calls)
    run_session(session, bench=bench, out=out, client=client, editor=editor, settings=settings)
    assert len(client.calls) == calls and len(editor.requests) == 2


def test_an_interrupted_turn_reuses_its_saved_refine_answer(tmp_path: Path) -> None:
    bench, out = _bench(tmp_path), tmp_path / "runs"
    session = json.loads((bench / "sessions.json").read_text())[0]
    settings = AgentSettings(agent_model="agent", verifier_model="verifier")

    class Broken(FakeEditor):
        def edit(self, request):  # type: ignore[no-untyped-def]
            raise KeyboardInterrupt

    client = ScriptedClient(answer=_answer)
    try:
        run_session(session, bench=bench, out=out, client=client, editor=Broken(), settings=settings)
    except KeyboardInterrupt:
        pass
    assert json.loads((out / "s1/conversation.json").read_text())["phase"] == "planned"
    refines = sum(call["system_instruction"] == REFINE_SYSTEM for call in client.calls)

    run_session(session, bench=bench, out=out, client=client, editor=FakeEditor(), settings=settings)
    # Turn 1 resumes from its plan; turns 2 and 3 are refined once each.
    assert sum(call["system_instruction"] == REFINE_SYSTEM for call in client.calls) == refines + 2
    assert (out / "s1/turn03.mp4").is_file()


def test_shipped_configs_load(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    configs = Path(__file__).resolve().parents[1] / "configs"
    gemini = load_config(configs / "gemini.yaml")
    assert gemini["gemini"]["project"] == "my-project"
    settings = agent_settings(gemini)
    assert settings.conversation == ConversationConfig() and settings.verify_fps == 3.92
    assert (settings.agent_model, settings.verifier_model) == ("gemini-3.6-flash", "gemini-3.7-flash")
    qwen = agent_settings(yaml.safe_load((configs / "qwen.yaml").read_text()))
    assert qwen.conversation.item_floor == 3
