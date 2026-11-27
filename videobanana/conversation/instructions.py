"""Instruction writer: the editor instructions for one turn.

The goal records absolute target states ("The grinder's outer casing is dark
blue."), but the editor receives the entry video and a few instructions, each
handled on its own. One LLM call per turn rewrites the execute and observe
intents as instructions relative to the entry: what to change from what the
entry shows, never how the conversation got there. Intents on the same element
may share one instruction; every intent is covered exactly once, in order, so
the verifier can still judge each one. At most ``max_instructions_per_call``
instructions are accepted, so one editor call carries them all.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from pydantic import Field

from .contracts import ContractError, ImmutableModel, TurnSpec
from .refine import JsonModelClient

PROMPT_VERSION = "instructions"

INSTRUCTION_SYSTEM = """You write instructions for a video editing model.

The editor receives one starting video and a short list of instructions, and applies each instruction on its own, predicting a separate edit region for each. Your input is a list of target states the edited video must show, some notes on what the starting video already shows, and, for context, the user's original words. Turn the target states into instructions for this starting video.

- Write each instruction as one concrete imperative sentence describing the change to make to the starting video, for example "Change the black outer casing of the coffee grinder to dark blue." or "Add a bouquet of red roses on the counter between the amp on the left and the grinder."
- Phrase the change from what the starting video actually shows. If it already shows an earlier version of the same element (listed under "starting_video_shows"), change that version; if the target is an element of the original video, describe it as it appears there.
- Keep every detail of the target state: position, relation to other things (for example that particles must not cover the flowers), size, material, placement. The user's words can help recover a detail the target state left out, but do not add anything beyond the target state; the user's message may also contain requests that are handled elsewhere.
- The editor knows nothing about the conversation, so write only about the video: avoid references such as "again", "instead", "back" or "undo".
- Targets on the same element can share one instruction ("Replace the dark SUV with a red sedan."). Keep separate elements in separate instructions, and cover every target exactly once.
- Do not write instructions for what the starting video already shows correctly.
- Some inputs may list "edits_to_remove": edits the starting video shows that are no longer wanted. Write the change that removes each one, returning that part to how the original video looks unless a target says otherwise; such a removal can share an instruction with a target on the same element ("Change the dark blue casing of the coffee grinder to white.").
"""


class EditorInstruction(ImmutableModel):
    text: str = Field(min_length=1)
    #: The intents this instruction realizes, in goal order.
    intent_ids: list[str] = Field(min_length=1)


def build_instruction_prompt(spec: TurnSpec, *, repair_message: str | None = None) -> str:
    targets = [*spec.execute, *spec.observe]
    payload: dict[str, Any] = {
        "starting_video": "the original source video" if spec.entry.parent_id is None else "an earlier edited version of the source video",
        "starting_video_shows": [item.text for item in [*spec.preserve, *spec.overwritten]],
        "targets": [
            {"id": item.intent_id, "target_state": item.text, "user_words": spec.user_requests.get(item.intent_id)}
            for item in targets
        ],
        # The entry selector never picks an entry that shows an unwanted edit,
        # so there is never anything to remove; the field stays in the prompt.
        "edits_to_remove": [],
        "task": "Return only a JSON object with one field.",
        "output_fields": {
            "instructions": 'list of {"intent_ids": ["I2"], "text": "..."}, in the order of the targets; every target and edit-to-remove id appears exactly once'
        },
    }
    if repair_message:
        payload["previous_answer_error"] = repair_message
    return json.dumps(payload, ensure_ascii=False, indent=1)


def parse_instructions(payload: Any, spec: TurnSpec) -> list[EditorInstruction]:
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        payload = payload[0]
    if not isinstance(payload, dict) or set(payload) != {"instructions"} or not isinstance(payload["instructions"], list):
        raise ContractError('answer must be a JSON object {"instructions": [...]}')
    instructions = [EditorInstruction.model_validate(item) for item in payload["instructions"]]
    wanted = [item.intent_id for item in [*spec.execute, *spec.observe]]
    covered = [intent_id for item in instructions for intent_id in item.intent_ids]
    if sorted(covered) != sorted(wanted):
        raise ContractError(f"instructions must cover each target exactly once: expected {wanted}, got {covered}")
    if len(instructions) > spec.max_instructions_per_call:
        raise ContractError(f"at most {spec.max_instructions_per_call} instructions per call, got {len(instructions)}")
    return instructions


class InstructionWriter:
    def __init__(
        self,
        client: JsonModelClient,
        *,
        model: str,
        max_attempts: int = 2,
        save_raw: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_attempts = max_attempts
        self.save_raw = save_raw

    def write(self, spec: TurnSpec) -> list[EditorInstruction]:
        if not spec.execute and not spec.observe:
            return []
        error: str | None = None
        for attempt in range(self.max_attempts):
            prompt = build_instruction_prompt(spec, repair_message=error)
            response = self.client.generate_json(model=self.model, system_instruction=INSTRUCTION_SYSTEM, prompt=prompt)
            if self.save_raw:
                self.save_raw(
                    f"instructions/T{spec.turn:02d}_{attempt:02d}.json",
                    {"prompt_version": PROMPT_VERSION, "model": self.model, "prompt": prompt, "raw_text": response.raw_text},
                )
            try:
                return parse_instructions(response.payload, spec)
            except (ContractError, ValueError) as exc:
                error = str(exc)
        raise ContractError(f"instruction writing failed after {self.max_attempts} attempts: {error}")
