"""Multi-turn conversation layer on top of a per-turn executor.

A turn advances through three durable phases, each written atomically to
``conversation.json`` before the next begins:

``planned``   the goal update is applied and the entry and roles are frozen;
``executed``  the executor returned its candidates and verdicts;
committed     the output is chosen, the ledger and failure counts are written,
              and the conversation waits for the next user turn.

A crash in any phase resumes from the last one written: a planned turn is
re-issued to the executor with the same spec, and an executed turn is
committed without calling the executor again.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .contracts import (
    ContractError,
    ConversationConfig,
    ConversationState,
    GoalUpdate,
    Node,
    PendingTurn,
    TurnRecord,
    TurnResult,
    TurnSpec,
)
from .entry import EntryChoice, select_entry
from .goal import apply_goal_update
from .roles import exhausted_ids, output_ledger
from .scoring import select_candidate

STATE_FILE = "conversation.json"
AUDIT_FILE = "audit.jsonl"
SOURCE_NODE_ID = "N0000"


class TurnExecutor(Protocol):
    """Runs one turn from ``spec.entry``: edit, verify, resample, and report every candidate.

    It must be safe to call again for a turn whose previous call was
    interrupted; the conversation layer re-issues the same spec on resume.
    """

    def run_turn(self, spec: TurnSpec) -> TurnResult: ...


class ConversationRunner:
    def __init__(self, root: str | Path, executor: TurnExecutor) -> None:
        self.root = Path(root)
        self.executor = executor

    @classmethod
    def create(
        cls,
        root: str | Path,
        source_video: str | Path,
        executor: TurnExecutor,
        *,
        config: ConversationConfig | None = None,
        conversation_id: str | None = None,
    ) -> "ConversationRunner":
        runner = cls(root, executor)
        if runner.state_path.exists():
            raise ContractError(f"conversation already exists: {runner.state_path}")
        source = Node(node_id=SOURCE_NODE_ID, parent_id=None, video_path=str(source_video), depth=0, turn=0)
        state = ConversationState(
            conversation_id=conversation_id or runner.root.name,
            config=config or ConversationConfig(),
            nodes={source.node_id: source},
            ledger={source.node_id: {}},
            scores={source.node_id: {}},
        )
        runner._save(state)
        runner._audit("conversation_created", source=str(source_video), config=state.config.model_dump(mode="json"))
        return runner

    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    def load(self) -> ConversationState:
        return ConversationState.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def submit_turn(self, user_text: str, update: GoalUpdate) -> TurnRecord:
        """Plan, execute and commit the next user turn."""
        state = self.load()
        if state.phase != "awaiting_turn":
            raise ContractError(f"turn {state.pending.turn if state.pending else '?'} is still in progress; call resume()")
        turn = len(state.turns) + 1
        stale = [item.intent_id for item in update.new_intents if item.introduced_turn != turn]
        if stale:
            raise ContractError(f"new intents must be introduced in turn {turn}: {stale}")
        # Failure counts were settled when the previous turn committed.
        exhausted = exhausted_ids(state)
        intents, lifecycle = apply_goal_update(state.intents, update)
        state = _with(state, intents=intents, lifecycle=lifecycle, constraints=list(update.constraints))
        exhausted = [intent_id for intent_id in exhausted if lifecycle[intent_id] == "active"]
        choice = select_entry(state, exhausted)
        pending = PendingTurn(turn=turn, user_text=user_text, spec=self._spec(state, turn, choice, user_text), exhausted_ids=exhausted)
        self._save(_with(state, phase="planned", pending=pending))
        self._audit(
            "turn_planned",
            turn=turn,
            entry=choice.node_id,
            reason=choice.reason,
            execute=choice.roles.execute,
            preserve=choice.roles.preserve,
            observe=choice.roles.observe,
            exhausted=exhausted,
        )
        record = self.resume()
        assert record is not None
        return record

    def resume(self) -> TurnRecord | None:
        """Finish an interrupted turn; ``None`` when no turn is in progress."""
        state = self.load()
        if state.phase == "awaiting_turn":
            return None
        if state.phase == "planned":
            assert state.pending is not None
            spec = state.pending.spec
            result = TurnResult() if not spec.execute else self.executor.run_turn(spec)
            state = _with(state, phase="executed", pending=state.pending.model_copy(update={"result": result}))
            self._save(state)
            self._audit("turn_executed", turn=spec.turn, candidates=[item.candidate_id for item in result.candidates])
        return self._commit(state)

    def _spec(self, state: ConversationState, turn: int, choice: EntryChoice, user_text: str) -> TurnSpec:
        roles = choice.roles
        intents = state.intents
        user_turns = {record.turn: record.user_text for record in state.turns} | {turn: user_text}
        return TurnSpec(
            conversation_id=state.conversation_id,
            turn=turn,
            entry=state.nodes[choice.node_id],
            execute=[intents[intent_id] for intent_id in roles.execute],
            preserve=[intents[intent_id] for intent_id in roles.preserve],
            observe=[intents[intent_id] for intent_id in roles.observe],
            overwritten=[intents[intent_id] for intent_id in roles.overwritten],
            reference_scores={
                intent_id: state.scores.get(choice.node_id, {}).get(intent_id) for intent_id in roles.preserve
            },
            damage_drop=state.config.damage_drop,
            user_requests={
                intent_id: user_turns[intents[intent_id].introduced_turn]
                for intent_id in [*roles.execute, *roles.observe]
                if intents[intent_id].introduced_turn in user_turns
            },
            max_attempts=state.config.max_attempts_per_turn,
            max_instructions_per_call=state.config.max_instructions_per_call,
        )

    def _commit(self, state: ConversationState) -> TurnRecord:
        pending = state.pending
        assert pending is not None and pending.result is not None
        spec = pending.spec
        nodes = dict(state.nodes)
        ledger = dict(state.ledger)
        scores = dict(state.scores)
        failures = dict(state.failure_counts)
        execute_ids = [item.intent_id for item in spec.execute]
        selected_id = None
        primary = secondary = None

        if not spec.execute:
            # The entry already satisfies the goal: expose it as this turn's output.
            output_id, selection, gate_passed = spec.entry.node_id, "reuse_entry", True
        else:
            chosen = select_candidate(pending.result, spec, state.config.gate_floor, state.config.item_floor)
            if chosen.candidate is None:
                # No usable video: the user keeps seeing the previous output.
                output_id, selection, gate_passed = state.last_output_id, "previous_output", False
                for intent_id in execute_ids:
                    failures[intent_id] = failures.get(intent_id, 0) + 1
            else:
                candidate = chosen.candidate
                output_id = f"N{len(nodes):04d}"
                nodes[output_id] = Node(
                    node_id=output_id,
                    parent_id=spec.entry.node_id,
                    video_path=candidate.video_path,
                    depth=spec.entry.depth + candidate.generations,
                    turn=spec.turn,
                    sub_session_id=pending.result.sub_session_id,
                )
                ledger[output_id], scores[output_id] = output_ledger(state, spec, candidate)
                for intent_id in execute_ids:
                    if not ledger[output_id].get(intent_id, False):
                        failures[intent_id] = failures.get(intent_id, 0) + 1
                selection = "gate_pass" if chosen.gate_passed else "composite"
                gate_passed = chosen.gate_passed
                selected_id = candidate.candidate_id
                primary, secondary = chosen.primary, chosen.secondary

        record = TurnRecord(
            turn=spec.turn,
            user_text=pending.user_text,
            entry_id=spec.entry.node_id,
            execute_ids=execute_ids,
            preserve_ids=[item.intent_id for item in spec.preserve],
            observe_ids=[item.intent_id for item in spec.observe],
            exhausted_ids=pending.exhausted_ids,
            output_id=output_id,
            selection=selection,
            selected_candidate_id=selected_id,
            gate_passed=gate_passed,
            primary_score=primary,
            secondary_score=secondary,
        )
        self._save(
            _with(
                state,
                phase="awaiting_turn",
                pending=None,
                nodes=nodes,
                ledger=ledger,
                scores=scores,
                failure_counts=failures,
                turns=[*state.turns, record],
            )
        )
        self._audit("turn_committed", **record.model_dump(mode="json"))
        return record

    def _save(self, state: ConversationState) -> None:
        atomic_write_text(self.state_path, state.model_dump_json(indent=2) + "\n")

    def _audit(self, event: str, **details: Any) -> None:
        line = json.dumps(
            {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(), "details": details},
            ensure_ascii=False,
        )
        with (self.root / AUDIT_FILE).open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a crash leaves either the old or the new file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temp.open("w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _with(state: ConversationState, **changes: Any) -> ConversationState:
    """Copy with validation (``model_copy`` would skip the phase invariants)."""
    return ConversationState.model_validate({**dict(state), **changes})
