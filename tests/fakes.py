"""Test doubles: scripted model client, scripted turn executor and a fake editor."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from videobanana.conversation.contracts import CandidateReport, TurnResult, TurnSpec
from videobanana.conversation.scoring import passes_gate
from videobanana.editor import EditorError, EditRequest


def write_video(path: Path, frames: int = 9, size: tuple[int, int] = (64, 48), fps: float = 24.0) -> Path:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for index in range(frames):
        writer.write(np.full((size[1], size[0], 3), index * 20 % 255, dtype=np.uint8))
    writer.release()
    return path


@dataclass
class Response:
    payload: Any
    raw_text: str = ""


class ScriptedClient:
    """Answers each call with the next scripted payload, or with ``answer(call)``."""

    def __init__(self, answers: Sequence[Any] = (), answer: Callable[[dict[str, Any]], Any] | None = None) -> None:
        self.answers = list(answers)
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate_json(self, **call: Any) -> Response:
        self.calls.append(call)
        payload = self.answer(call) if self.answer else self.answers.pop(0)
        return Response(payload, raw_text=json.dumps(payload))


class FakeEditor:
    """Copies the input video to the output; fails the first ``failures`` calls."""

    def __init__(self, failures: int = 0) -> None:
        self.failures = failures
        self.requests: list[EditRequest] = []

    def edit(self, request: EditRequest) -> None:
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise EditorError("scripted failure")
        shutil.copyfile(request.input_video, request.output_path)


@dataclass
class FakeCandidate:
    """Scripted verdicts for one attempt. Unlisted intents score 3."""

    scores: dict[str, int | None] = field(default_factory=dict)
    nontarget: int | None = 3
    quality: int | None = 3
    structural: bool = True


class FakeTurnExecutor:
    """Deterministic executor for conversation-layer tests.

    ``script`` maps a turn number to the attempts it produces. Attempts stop at
    the first one that passes the gate or at ``spec.max_attempts``.
    """

    def __init__(self, script: Mapping[int, Sequence[FakeCandidate]] | None = None, *, floor: int = 2) -> None:
        self.script = {turn: list(items) for turn, items in (script or {}).items()}
        self.floor = floor
        self.calls: list[TurnSpec] = []

    def run_turn(self, spec: TurnSpec) -> TurnResult:
        self.calls.append(spec)
        candidates: list[CandidateReport] = []
        for index, attempt in enumerate((self.script.get(spec.turn) or [FakeCandidate()])[: spec.max_attempts], start=1):
            candidate = CandidateReport(
                candidate_id=f"T{spec.turn:02d}_A{index}",
                video_path=str(Path("fake") / spec.conversation_id / f"turn_{spec.turn:02d}_attempt{index}.mp4"),
                structural_passed=attempt.structural,
                instruction_scores={item.intent_id: attempt.scores.get(item.intent_id, 3) for item in [*spec.execute, *spec.observe]},
                retention_scores={item.intent_id: attempt.scores.get(item.intent_id, 3) for item in spec.preserve},
                nontarget_score=attempt.nontarget,
                quality_score=attempt.quality,
            )
            candidates.append(candidate)
            if passes_gate(candidate, spec, self.floor):
                break
        return TurnResult(candidates=candidates, sub_session_id=f"fake-{spec.conversation_id}-T{spec.turn:02d}")
