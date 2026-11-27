from __future__ import annotations

import json
from pathlib import Path

import pytest
from fakes import FakeCandidate, FakeTurnExecutor

from videobanana.conversation import (
    CandidateReport,
    ContractError,
    ConversationConfig,
    ConversationIntent,
    ConversationRunner,
    GoalUpdate,
    TurnRecord,
)
from videobanana.conversation.scoring import composite_scores


def _intent(intent_id: str, turn: int, text: str, *, replaces: tuple[str, ...] = ()) -> ConversationIntent:
    return ConversationIntent(intent_id=intent_id, introduced_turn=turn, text=text, replaces=list(replaces))


def _goal(active: list[str], *new: ConversationIntent, constraints: tuple[str, ...] = ()) -> GoalUpdate:
    return GoalUpdate(new_intents=list(new), active=active, constraints=list(constraints))


def _grow(runner: ConversationRunner, *new: ConversationIntent) -> GoalUpdate:
    """Keep the current goal and add ``new``."""
    kept = [intent_id for intent_id, status in runner.load().lifecycle.items() if status == "active"]
    return _goal([*kept, *(item.intent_id for item in new)], *new)


def _runner(tmp_path: Path, script: dict[int, list[FakeCandidate]] | None = None, **config: object) -> tuple[ConversationRunner, FakeTurnExecutor]:
    executor = FakeTurnExecutor(script)
    runner = ConversationRunner.create(tmp_path / "conversation", tmp_path / "source.mp4", executor, config=ConversationConfig(**config))
    return runner, executor


def _assert_gated_once(runner: ConversationRunner, record: TurnRecord) -> None:
    """Every active, non-exhausted intent is gated exactly once: execute xor preserve."""
    state = runner.load()
    active = {intent_id for intent_id, status in state.lifecycle.items() if status == "active"}
    assert set(record.execute_ids) | set(record.preserve_ids) == active - set(record.exhausted_ids)
    assert not set(record.execute_ids) & set(record.preserve_ids)


def _grinder_dialogue(runner: ConversationRunner) -> list[TurnRecord]:
    """White casing, roses, blue instead, particles, undo blue, restore the original casing."""
    updates = [
        _goal(["I1"], _intent("I1", 1, "The grinder casing is white.")),
        _goal(["I1", "I2"], _intent("I2", 2, "A bouquet of red roses stands on the counter.")),
        _goal(["I2", "I3"], _intent("I3", 3, "The grinder casing is dark blue.", replaces=("I1",))),
        _goal(["I2", "I3", "I4"], _intent("I4", 4, "Golden light particles drift above the counter.")),
        _goal(["I1", "I2", "I4"]),
        _goal(["I2", "I4"]),
    ]
    records = []
    for index, update in enumerate(updates, start=1):
        record = runner.submit_turn(f"user turn {index}", update)
        _assert_gated_once(runner, record)
        records.append(record)
    return records


def test_cross_turn_merge_redoes_earlier_edits_to_keep_outputs_shallow(tmp_path: Path) -> None:
    runner, executor = _runner(tmp_path)
    records = _grinder_dialogue(runner)

    assert [record.entry_id for record in records] == ["N0000", "N0000", "N0000", "N0003", "N0002", "N0000"]
    # T2 redoes the white casing together with the roses in one call from the source.
    assert records[1].execute_ids == ["I1", "I2"]
    # From the source T4 would need three instructions, so it continues from T3.
    assert records[3].execute_ids == ["I4"] and records[3].preserve_ids == ["I2", "I3"]
    # T5 undoes blue: N0002 already shows the white casing and the roses.
    assert records[4].execute_ids == ["I4"] and records[4].preserve_ids == ["I1", "I2"]
    # T6 restores the original casing: only the source shows neither colour.
    assert records[5].execute_ids == ["I2", "I4"]
    assert executor.calls[5].user_requests == {"I2": "user turn 2", "I4": "user turn 4"}
    state = runner.load()
    assert state.lifecycle == {"I1": "inactive", "I2": "active", "I3": "inactive", "I4": "active"}
    assert max(node.depth for node in state.nodes.values()) == 2


def test_a_revision_from_an_entry_that_shows_the_old_version_overwrites_it(tmp_path: Path) -> None:
    runner, executor = _runner(tmp_path)
    runner.submit_turn("white cup", _goal(["I1"], _intent("I1", 1, "The cup is white.")))
    runner.submit_turn("spoon", _grow(runner, _intent("I2", 2, "A spoon lies on the saucer.")))
    runner.submit_turn("table", _grow(runner, _intent("I3", 3, "The table is oak.")))
    record = runner.submit_turn("blue instead", _goal(["I2", "I3", "I4"], _intent("I4", 4, "The cup is blue.", replaces=("I1",))))

    # The shallowest entry within the merge limit shows the white cup, which the blue edit overwrites.
    assert record.entry_id == "N0002" and record.execute_ids == ["I3", "I4"]
    assert [item.intent_id for item in executor.calls[-1].overwritten] == ["I1"]
    # Blue overwrote white in the output, so the ledger does not claim both.
    state = runner.load()
    assert state.ledger[record.output_id]["I4"] is True and state.ledger[record.output_id]["I1"] is False


def test_depth_limit_restarts_from_a_shallower_node(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    for turn in range(1, 7):
        runner.submit_turn(f"add {turn}", _grow(runner, _intent(f"I{turn}", turn, f"Object {turn} is added.")))
    state = runner.load()
    assert [item.entry_id for item in state.turns] == ["N0000", "N0000", "N0002", "N0002", "N0004", "N0004"]
    # Continuing from the depth-3 output of T5 would exceed the depth limit.
    assert state.nodes["N0005"].depth == 3 and state.turns[5].execute_ids == ["I5", "I6"]
    assert all(node.depth <= 3 for node in state.nodes.values())


def test_more_new_edits_than_the_merge_limit_take_the_fewest_instructions(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    update = _goal(["I1", "I2", "I3"], *(_intent(f"I{n}", 1, f"Object {n} is added.") for n in (1, 2, 3)))
    record = runner.submit_turn("three edits", update)
    assert record.entry_id == "N0000" and record.execute_ids == ["I1", "I2", "I3"]
    planned = [json.loads(line) for line in (runner.root / "audit.jsonl").read_text().splitlines()][1]
    assert planned["details"]["reason"] == "fewest instructions within depth and call limits"


def test_merge_limit_cannot_exceed_the_call_limit() -> None:
    with pytest.raises(ValueError, match="merge_max_instructions"):
        ConversationConfig(max_instructions_per_call=2, observe_fill_total=2, merge_max_instructions=3)


def test_defaults_are_the_papers() -> None:
    config = ConversationConfig()
    assert (config.max_depth, config.max_instructions_per_call, config.merge_max_instructions) == (3, 5, 2)
    assert (config.observe_fill_total, config.max_attempts_per_turn, config.gate_floor, config.item_floor) == (3, 2, 2, 2)


def test_bringing_back_a_replaced_object_rebuilds_without_its_edits(tmp_path: Path) -> None:
    """SUV -> white sedan, bike, red instead, mist, undo red, SUV back."""
    runner, _ = _runner(tmp_path)
    lighting = "Keep the scene's twilight lighting."
    runner.submit_turn("replace the SUV with a white sedan", _goal(["I1"], _intent("I1", 1, "A white sedan stands in the SUV's place."), constraints=(lighting,)))
    runner.submit_turn("add a red bicycle", _goal(["I1", "I2"], _intent("I2", 2, "A red-framed bicycle leans on the railing."), constraints=(lighting,)))
    runner.submit_turn("paint the sedan red instead", _goal(["I2", "I3"], _intent("I3", 3, "A red sedan stands in the SUV's place.", replaces=("I1",)), constraints=(lighting,)))
    unobscured = (lighting, "The sedan and the bicycle stay unobscured by the mist.")
    runner.submit_turn("add mist behind the sedan", _goal(["I2", "I3", "I4"], _intent("I4", 4, "Low mist drifts along the road."), constraints=unobscured))
    record = runner.submit_turn("undo the red paint", _goal(["I1", "I2", "I4"], constraints=unobscured))
    # The white-sedan output of T2 is reused as the entry and only the mist is added.
    assert record.entry_id == "N0002" and record.execute_ids == ["I4"]
    record = runner.submit_turn("bring back the original SUV", _goal(["I2", "I4"], constraints=("The bicycle stays unobscured by the mist.",)))

    assert runner.load().lifecycle == {"I1": "inactive", "I2": "active", "I3": "inactive", "I4": "active"}
    assert record.entry_id == "N0000" and record.execute_ids == ["I2", "I4"]
    assert runner.load().constraints == ["The bicycle stays unobscured by the mist."]


def test_returning_to_an_earlier_state_reuses_its_intent(tmp_path: Path) -> None:
    """"make it white again" after blue reuses the white intent; "back to the original red" drops both."""
    runner, _ = _runner(tmp_path)
    runner.submit_turn("make the cup white", _goal(["I1"], _intent("I1", 1, "The cup is white.")))
    runner.submit_turn("add a spoon", _goal(["I1", "I2"], _intent("I2", 2, "A spoon lies on the saucer.")))
    runner.submit_turn("blue instead", _goal(["I2", "I3"], _intent("I3", 3, "The cup is blue.", replaces=("I1",))))

    back_to_white = ConversationRunner(tmp_path / "white", FakeTurnExecutor())
    back_to_white.state_path.parent.mkdir(parents=True)
    back_to_white.state_path.write_text(runner.load().model_dump_json(), encoding="utf-8")
    record = back_to_white.submit_turn("make it white again", _goal(["I1", "I2"]))
    assert back_to_white.load().lifecycle == {"I1": "active", "I2": "active", "I3": "inactive"}
    assert record.entry_id == "N0002" and record.selection == "reuse_entry"

    record = runner.submit_turn("change it back to the original red", _goal(["I2"]))
    assert record.entry_id == "N0000" and record.execute_ids == ["I2"]


def test_repeated_failures_exhaust_an_intent_into_an_observe_item(tmp_path: Path) -> None:
    fail_i1 = FakeCandidate(scores={"I1": 0})
    runner, _ = _runner(tmp_path, script={1: [fail_i1, fail_i1], 2: [fail_i1, fail_i1], 3: [FakeCandidate()]})
    first = runner.submit_turn("hard edit", _grow(runner, _intent("I1", 1, "The sky is purple.")))
    # Two failed attempts in one turn count once.
    assert runner.load().failure_counts == {"I1": 1}
    assert first.selection == "composite" and not first.gate_passed

    runner.submit_turn("easy edit", _grow(runner, _intent("I2", 2, "A vase stands on the table.")))
    assert runner.load().failure_counts == {"I1": 2}

    third = runner.submit_turn("another edit", _grow(runner, _intent("I3", 3, "The wall is green.")))
    assert third.exhausted_ids == ["I1"]
    assert third.observe_ids == ["I1"] and "I1" not in third.execute_ids
    assert third.gate_passed
    state = runner.load()
    assert state.failure_counts == {"I1": 2}
    assert state.ledger[third.output_id]["I1"] is True


def test_failed_gate_still_advances_with_the_best_composite_candidate(tmp_path: Path) -> None:
    # A: perfect new edits but destroys the earlier edit. B: one point short.
    candidate_a = FakeCandidate(scores={"I2": 3, "I3": 3, "I1": 0})
    candidate_b = FakeCandidate(scores={"I2": 2, "I3": 1, "I1": 2}, nontarget=2, quality=2)
    runner, executor = _runner(tmp_path, script={2: [candidate_a, candidate_b]})
    runner.submit_turn("first edit", _grow(runner, _intent("I1", 1, "The hat is yellow.")))
    record = runner.submit_turn("two more edits", _grow(runner, _intent("I2", 2, "The cup is blue."), _intent("I3", 2, "The shirt is striped.")))

    assert not record.gate_passed and record.selection == "composite"
    assert record.selected_candidate_id == "T02_A2"
    assert record.primary_score == pytest.approx(0.9166667)
    state = runner.load()
    assert state.ledger[record.output_id] == {"I1": True, "I2": True, "I3": False}
    assert state.failure_counts == {"I3": 1}
    assert runner.submit_turn("next", _grow(runner, _intent("I4", 3, "The door is red."))).turn == 3
    assert len(executor.calls) == 3


def test_item_gate_floor_applies_to_intent_scores_only(tmp_path: Path) -> None:
    candidate = FakeCandidate(scores={"I1": 2}, nontarget=2, quality=2)
    default, _ = _runner(tmp_path / "default", script={1: [candidate]})
    strict, _ = _runner(tmp_path / "strict", script={1: [candidate]}, item_gate_floor=3)
    update = _goal(["I1"], _intent("I1", 1, "The hat is yellow."))

    assert default.submit_turn("edit", update).gate_passed
    assert default.load().ledger["N0001"]["I1"] is True
    record = strict.submit_turn("edit", update)
    assert not record.gate_passed and record.selection == "composite"
    state = strict.load()
    assert state.ledger[record.output_id]["I1"] is False
    assert state.failure_counts == {"I1": 1}


def test_composite_primary_clips_surplus_above_the_floor(tmp_path: Path) -> None:
    runner, executor = _runner(tmp_path)
    runner.submit_turn("first edit", _grow(runner, _intent("I1", 1, "The hat is yellow.")))
    runner.submit_turn("second", _grow(runner, _intent("I2", 2, "The cup is blue."), _intent("I3", 2, "The shirt is striped.")))
    spec = executor.calls[-1]
    a = CandidateReport(candidate_id="a", video_path="a.mp4", instruction_scores={"I2": 3, "I3": 3}, retention_scores={"I1": 0}, nontarget_score=3, quality_score=3)
    b = CandidateReport(candidate_id="b", video_path="b.mp4", instruction_scores={"I2": 2, "I3": 1}, retention_scores={"I1": 2}, nontarget_score=2, quality_score=2)
    assert composite_scores(a, spec, 2) == pytest.approx((0.8333333, 0.8333333))
    assert composite_scores(b, spec, 2) == pytest.approx((0.9166667, 0.6111111))


def test_no_valid_candidate_keeps_the_previous_output(tmp_path: Path) -> None:
    broken = FakeCandidate(structural=False)
    runner, _ = _runner(tmp_path, script={2: [broken, broken]})
    first = runner.submit_turn("first", _grow(runner, _intent("I1", 1, "The hat is yellow.")))
    record = runner.submit_turn("second", _grow(runner, _intent("I2", 2, "The cup is blue.")))

    assert record.selection == "previous_output" and record.output_id == first.output_id
    state = runner.load()
    assert len(state.nodes) == 2
    # T2 redid the hat from the source together with the cup, so both failed.
    assert state.failure_counts == {"I1": 1, "I2": 1}


def test_entry_that_already_satisfies_the_goal_is_reused_without_editing(tmp_path: Path) -> None:
    runner, executor = _runner(tmp_path)
    runner.submit_turn("add a dog", _goal(["I1"], _intent("I1", 1, "A dog sits on the deck.")))
    runner.submit_turn("add a cat", _goal(["I1", "I2"], _intent("I2", 2, "A cat sleeps on the sofa.")))
    record = runner.submit_turn("remove the cat we added", _goal(["I1"]))

    assert record.selection == "reuse_entry"
    assert record.entry_id == record.output_id == "N0001"
    assert len(executor.calls) == 2
    assert runner.load().turns[-1].gate_passed


def test_resume_reissues_a_planned_turn_after_an_executor_crash(tmp_path: Path) -> None:
    class Crashing(FakeTurnExecutor):
        def run_turn(self, spec):  # type: ignore[no-untyped-def]
            raise RuntimeError("GPU worker died")

    runner = ConversationRunner.create(tmp_path / "conversation", tmp_path / "source.mp4", Crashing())
    with pytest.raises(RuntimeError):
        runner.submit_turn("first", _goal(["I1"], _intent("I1", 1, "The hat is yellow.")))
    assert runner.load().phase == "planned"
    with pytest.raises(ContractError):
        runner.submit_turn("second", _goal(["I1", "I2"], _intent("I2", 2, "The cup is blue.")))

    executor = FakeTurnExecutor()
    record = ConversationRunner(tmp_path / "conversation", executor).resume()
    assert record is not None and record.turn == 1 and record.gate_passed
    assert len(executor.calls) == 1
    assert runner.load().lifecycle == {"I1": "active"}


def test_resume_commits_an_executed_turn_without_rerunning_the_editor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner, executor = _runner(tmp_path)

    def crash(self, state):  # type: ignore[no-untyped-def]
        raise RuntimeError("killed before commit")

    monkeypatch.setattr(ConversationRunner, "_commit", crash)
    with pytest.raises(RuntimeError):
        runner.submit_turn("first", _goal(["I1"], _intent("I1", 1, "The hat is yellow.")))
    assert runner.load().phase == "executed"
    monkeypatch.undo()

    fresh = FakeTurnExecutor()
    record = ConversationRunner(tmp_path / "conversation", fresh).resume()
    assert record is not None and record.output_id == "N0001"
    assert fresh.calls == [] and len(executor.calls) == 1
    assert ConversationRunner(tmp_path / "conversation", fresh).resume() is None


@pytest.mark.parametrize(
    ("update", "message"),
    [
        (_goal(["I1", "I9"]), "unknown intents"),
        (_goal(["I1"], _intent("I1", 2, "The hat is red.")), "already exist"),
        (_goal(["I1"], _intent("I2", 2, "The hat is red.")), "missing from the active list"),
        (_goal(["I2"], _intent("I2", 2, "The hat is red.", replaces=("I9",))), "replaces unknown"),
        (_goal(["I1", "I2"], _intent("I2", 2, "The hat is red.", replaces=("I1",))), "stay active"),
        (_goal(["I1", "I1"]), "repeats"),
        (_goal(["I2"], _intent("I2", 3, "The hat is red.")), "introduced in turn 2"),
    ],
)
def test_invalid_updates_are_rejected_without_changing_state(tmp_path: Path, update: GoalUpdate, message: str) -> None:
    runner, _ = _runner(tmp_path)
    runner.submit_turn("first", _goal(["I1"], _intent("I1", 1, "The hat is yellow.")))
    before = runner.state_path.read_text(encoding="utf-8")
    with pytest.raises(ContractError, match=message):
        runner.submit_turn("second", update)
    assert runner.state_path.read_text(encoding="utf-8") == before


def test_state_and_audit_are_persisted_per_phase(tmp_path: Path) -> None:
    runner, _ = _runner(tmp_path)
    runner.submit_turn("first", _goal(["I1"], _intent("I1", 1, "The hat is yellow.")))
    events = [json.loads(line)["event"] for line in (runner.root / "audit.jsonl").read_text().splitlines()]
    assert events == ["conversation_created", "turn_planned", "turn_executed", "turn_committed"]
    reloaded = ConversationRunner(runner.root, FakeTurnExecutor()).load()
    assert reloaded == runner.load()
    assert reloaded.phase == "awaiting_turn" and reloaded.pending is None


def test_a_weak_earlier_edit_is_not_blamed_on_the_turn_but_leaves_the_ledger(tmp_path: Path) -> None:
    # T1 barely achieves I1 (2). T2 keeps it at 1: one level down is not damage,
    # so the gate passes, but I1 is no longer shown by T2's output, so T3 does
    # not build on that output as if it were.
    runner, _ = _runner(tmp_path, script={1: [FakeCandidate(scores={"I1": 2})], 2: [FakeCandidate(scores={"I1": 1})]})
    runner.submit_turn("hat and cup", _grow(runner, _intent("I1", 1, "The hat is yellow."), _intent("I2", 1, "The cup is blue.")))
    second = runner.submit_turn("shirt", _grow(runner, _intent("I3", 2, "The shirt is striped.")))
    assert second.entry_id == "N0001" and second.preserve_ids == ["I1", "I2"] and second.gate_passed
    state = runner.load()
    assert state.scores[second.output_id]["I1"] == 1 and state.ledger[second.output_id]["I1"] is False
    third = runner.submit_turn("door", _grow(runner, _intent("I4", 3, "The door is red.")))
    assert third.entry_id == "N0001" and third.preserve_ids == ["I1", "I2"] and third.execute_ids == ["I3", "I4"]


def test_dropping_an_earlier_edit_by_two_levels_is_damage(tmp_path: Path) -> None:
    damaged = FakeCandidate(scores={"I1": 1})
    runner, _ = _runner(tmp_path, script={2: [damaged, damaged]})
    runner.submit_turn("hat and cup", _grow(runner, _intent("I1", 1, "The hat is yellow."), _intent("I2", 1, "The cup is blue.")))
    record = runner.submit_turn("shirt", _grow(runner, _intent("I3", 2, "The shirt is striped.")))
    assert record.preserve_ids == ["I1", "I2"] and not record.gate_passed and record.selection == "composite"
    # Damage is not a failure of the damaged intent itself.
    assert runner.load().failure_counts == {}


def test_observe_intents_count_towards_instruction_following(tmp_path: Path) -> None:
    runner, executor = _runner(tmp_path)
    runner.submit_turn("first", _grow(runner, _intent("I1", 1, "The hat is yellow.")))
    spec = executor.calls[-1].model_copy(update={"observe": [_intent("I9", 1, "The sky is purple.")]})
    with_observe = CandidateReport(candidate_id="a", video_path="a.mp4", instruction_scores={"I1": 3, "I9": 0}, nontarget_score=3, quality_score=3)
    assert composite_scores(with_observe, spec, 2)[0] == pytest.approx((0.5 + 1 + 1) / 3)
