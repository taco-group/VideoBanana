"""Intent compiler: checks a turn's editor instructions before the editor runs.

Each attempt first turns the written instructions into atomic edit intents. The
compiler must cover every instruction; if a target is ambiguous it returns
``blocked_ambiguity`` and the attempt makes no edit. The editor still receives
the instructions as written: the compiled intents are a check, not a rewrite.
An answer that does not match the schema is asked again with the error
(``repair_message``), at most ``max_repairs`` times.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import parse_json_response

PROMPT_VERSION = "intent-compiler"

#: The splitting and wording rules are the intent refiner's (``refine.py``),
#: minus the ones that only apply to a dialogue (revisions, undo, requests
#: carried across turns), so both define an intent the same way.
COMPILER_SYSTEM = """You turn a video-editing request into edit intents.

The request gives the edits for one source video as a list of instructions; a single instruction may describe several edits. You do not see the video. Each intent is later planned and executed by a video editor and checked on its own by a video judge. The guidance below serves that purpose; where a case is not covered, do what serves it best. Return only JSON matching the supplied schema, keep every instruction index, and do not resolve an ambiguous target by guessing: return blocked_ambiguity instead.

What the user wants
- Do not add edits the user did not ask for.

Splitting into intents
- Aim for one intent per thing the user asked to change, so that each can be judged on its own. Changing an existing wall's color and its texture is two intents; "add a blue brick wall" or "replace the SUV with a white sedan" is one, because the appearance describes the new element. Several elements in one instruction usually mean several intents, and separate instructions should not be merged just to have fewer.
- Write each intent's "text" as a self-contained description of the result, understandable without the request, for example "A table stands to the right of the window and to the left of the lamp."
- Describe what the edited video shows, in the present tense, never the edit itself: "The fish has no dorsal fin.", not "The dorsal fin is removed."
- Carry over the new information the user gave about that edit: where it goes, how it relates to other things (for instance that the particles must not cover the flowers), its size, material or placement.
- Avoid wording that only makes sense relative to the request or the editing, such as "undo", "instead" or "again".

Fields for the planner
- "instruction_index": the instruction the intent comes from. "target": the element it changes, as the source video shows it. "operation": a short verb for the change (restyle, replace, add, remove, recolor, ...). "desired_state": the target's new appearance in a few words.

Constraints
- List in "preservation_constraints" what the instruction only asks to keep as it would stay anyway, such as "keep the lamp's light unchanged" or "leave the other cars alone". Anything that adds new information belongs in an intent instead.
"""


class IntentProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instruction_index: int = Field(ge=0)
    #: The intent as the result it leaves, in the intent refiner's form.
    text: str = Field(min_length=1)
    target: str = Field(min_length=1)
    operation: str = Field(min_length=1)
    desired_state: str = Field(min_length=1)
    preservation_constraints: list[str] = Field(default_factory=list)


class CompilerResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = "ok"
    intents: list[IntentProposal] = Field(default_factory=list)
    reason: str | None = None


@dataclass
class CompileResult:
    """``ok``, ``blocked_ambiguity`` (the compiler refused) or ``failed`` (no valid answer)."""

    status: str
    intents: list[dict[str, Any]] = field(default_factory=list)
    reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def build_compiler_prompt(instructions: list[str], *, request_id: str = "R0001", repair_message: str | None = None) -> str:
    return json.dumps(
        {
            "request_id": request_id,
            "instructions": [{"instruction_index": index, "text": text} for index, text in enumerate(instructions)],
            "output_schema": CompilerResponse.model_json_schema(),
            "repair_message": repair_message,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def check_response(payload: Any, instruction_count: int) -> CompilerResponse:
    """The validated answer; ``ValueError`` (or ``ValidationError``) when it must be asked again."""
    parsed = CompilerResponse.model_validate(payload)
    if parsed.status == "blocked_ambiguity":
        return parsed
    if parsed.status != "ok" or not parsed.intents:
        raise ValueError("compiler returned no intents")
    indexes = {item.instruction_index for item in parsed.intents}
    missing = sorted(set(range(instruction_count)) - indexes)
    if missing:
        raise ValueError(f"compiler omitted source instruction indices {missing}")
    out_of_range = sorted(index for index in indexes if index >= instruction_count)
    if out_of_range:
        raise ValueError(f"source instruction index out of range: {out_of_range[0]}")
    return parsed


class IntentCompiler:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        max_repairs: int = 2,
        save_raw: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_repairs = max_repairs
        self.save_raw = save_raw

    def compile(self, instructions: list[str], *, tag: str = "compile") -> CompileResult:
        error: Exception | None = None
        for attempt in range(self.max_repairs + 1):
            prompt = build_compiler_prompt(instructions, repair_message=str(error) if error else None)
            response = self.client.generate_json(model=self.model, system_instruction=COMPILER_SYSTEM, prompt=prompt)
            if self.save_raw:
                self.save_raw(
                    f"intent_compiler/{tag}_{attempt}.json",
                    {"prompt_version": PROMPT_VERSION, "model": self.model, "prompt": prompt, "raw_text": response.raw_text},
                )
            try:
                parsed = check_response(parse_json_response(response.raw_text), len(instructions))
            except (ValidationError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
                error = exc
                continue
            intents = [item.model_dump() for item in parsed.intents]
            if parsed.status == "blocked_ambiguity":
                return CompileResult("blocked_ambiguity", intents, parsed.reason or "intent compiler reported ambiguity")
            return CompileResult("ok", intents)
        return CompileResult("failed", reason=f"no valid compiler answer after {self.max_repairs + 1} tries: {error}")
