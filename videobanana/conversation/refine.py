"""Intent refiner: turn a conversation into the goal that should hold after its latest turn.

One LLM call per user turn. The model sees every user turn so far (never the
videos or verification results: what the user wants is decided by the
dialogue alone) and every intent recorded so far. It answers with the new
intents, the existing intents to deactivate or reactivate, and the complete
active list, and the two ways of stating the goal must agree.

New intents carry a local ``key`` and receive their permanent id here, so the
model cannot collide with or recycle ids. A malformed or inconsistent answer
is sent back once with the error; a second failure raises.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

from .contracts import ContractError, ConversationIntent, ConversationState, GoalUpdate, Lifecycle
from .goal import apply_goal_update

PROMPT_VERSION = "refine"
#: Name of the answer format, recorded with every raw answer.
FORMAT = "both"

REFINE_SYSTEM = """You keep track of what a user currently wants in a multi-turn video editing conversation.

The user edits one source video over several turns, and each turn produces a new video. You do not see the videos. After the latest turn, write down the edit intents the next video should satisfy. Each intent is later turned into an editing instruction and checked on its own by a video judge, and the system reuses an earlier video when the intents it needs are already there. The guidance below serves that purpose; where a case is not covered, do what serves it best.

What the user wants
- Requests stay in effect until the user revises or cancels them, even if an earlier video may not have shown them correctly.
- Do not add edits the user did not ask for.

Splitting into intents
- Aim for one intent per thing the user asked to change, so that each can be judged, kept or undone on its own. Changing an existing wall's color and its texture is two intents; "add a blue brick wall" or "replace the SUV with a white sedan" is one, because the appearance describes the new element. Several elements in one request usually mean several intents, and separate requests should not be merged just to have fewer.
- Write each intent as a self-contained description of the result, understandable without the conversation, for example "A table stands to the right of the window and to the left of the lamp."
- Carry over the new information the user gave about that edit: where it goes, how it relates to other things (for instance that the particles must not cover the flowers), its size, material or placement.
- Avoid wording that only makes sense relative to the conversation history, such as "undo", "instead" or "again".

Revisions and undo
- When the user changes something an active intent already sets, including the look of something added earlier, write a new intent with the updated description and list the old one in "replaces".
- Write a new intent only when the user changes that element itself. An intent that mentions another element (mist that must not cover the sedan) stays as it is when that other element changes or disappears; keep its id.
- When the user undoes a change, removes something that was added, or asks for the original back, make the corresponding intents inactive. Undoing a revision brings back the intent it replaced. There is no need for an intent that describes the original video.
- If the goal returns to a state an earlier intent already describes, reuse that intent's id instead of writing a new one: this is what lets the system go back to an earlier video. "Red" after "bright red" is usually the same state unless the user makes clear it is different.
- Intents that only make sense for an element that is gone (removed, replaced, or its addition undone) become inactive as well.

Constraints
- Separately list requests that add no new information and only stress that something should stay as it would anyway, such as "keep the lamp's light unchanged", "keep the pajamas' cut" or "leave the other cars alone". Anything that adds new information belongs in an intent instead.
- Keep this list current and drop items that no longer apply.
"""

OUTPUT_FIELDS: dict[str, str] = {
    "new_intents": 'list of {"key": "new1", "text": "...", "replaces": ["I3"]}; new intents are active',
    "deactivate": 'list of {"id": "I2", "reason": "..."}: existing active intents that stop applying',
    "reactivate": 'list of {"id": "I1", "reason": "..."}: existing inactive intents that apply again',
    "constraints": "list of strings: the preservation-only requests that still apply",
    "active": "list of existing intent ids and new-intent keys: every intent the next output must satisfy"
    "; must equal the previous active list after the changes",
}


class JsonModelClient(Protocol):
    """What the agent needs from a model: a JSON answer (``.payload``) and its text (``.raw_text``)."""

    def generate_json(self, *, model: str, system_instruction: str, prompt: str, **media: Any) -> Any: ...


def build_refine_prompt(
    user_turns: list[str],
    intents: dict[str, ConversationIntent],
    lifecycle: dict[str, Lifecycle],
    *,
    repair_message: str | None = None,
) -> str:
    payload = {
        "conversation": [{"turn": index, "user": text} for index, text in enumerate(user_turns, start=1)],
        "latest_turn": len(user_turns),
        "recorded_intents": [
            {
                "id": intent_id,
                "text": intents[intent_id].text,
                "status": lifecycle[intent_id],
                # Needed to know what "undo that change" returns to.
                "replaces": intents[intent_id].replaces,
            }
            for intent_id in sorted(intents, key=_order_key)
        ],
        "task": "State the goal after the latest turn. Return only a JSON object with exactly these fields.",
        "output_fields": OUTPUT_FIELDS,
    }
    if repair_message:
        payload["previous_answer_error"] = repair_message
    return json.dumps(payload, ensure_ascii=False, indent=1)


def parse_refine_response(
    payload: Any,
    *,
    intents: dict[str, ConversationIntent],
    lifecycle: dict[str, Lifecycle],
    turn: int,
) -> GoalUpdate:
    """Validate the model's answer and turn it into a :class:`GoalUpdate` with permanent ids."""
    # The model occasionally wraps the object in a one-element list.
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ContractError("answer must be a JSON object")
    expected = set(OUTPUT_FIELDS)
    if set(payload) != expected:
        raise ContractError(f"answer fields must be exactly {sorted(expected)}, got {sorted(payload)}")

    active_before = {intent_id for intent_id, status in lifecycle.items() if status == "active"}
    new_ids = _next_ids(intents, len(_list(payload, "new_intents")))
    key_to_id: dict[str, str] = {}
    new_intents = []
    for item, intent_id in zip(_list(payload, "new_intents"), new_ids):
        # An empty "replaces" is often left out; anything else missing or extra is an error.
        if not isinstance(item, dict) or not {"key", "text"} <= set(item) <= {"key", "text", "replaces"}:
            raise ContractError(f"each new intent needs key, text and optionally replaces: {item}")
        key = str(item["key"])
        if key in key_to_id or key in intents:
            raise ContractError(f"new-intent key {key!r} is repeated or collides with an existing id")
        replaces = [str(value) for value in _as_list(item.get("replaces", []), "replaces")]
        unknown = sorted(set(replaces) - set(intents))
        if unknown:
            raise ContractError(f"new intent {key!r} replaces unknown intents {unknown}; only existing ids can be replaced")
        key_to_id[key] = intent_id
        new_intents.append(
            ConversationIntent(intent_id=intent_id, introduced_turn=turn, text=str(item["text"]).strip(), replaces=replaces)
        )
    replaced = {value for intent in new_intents for value in intent.replaces}

    deactivate = _ids(payload, "deactivate")
    reactivate = _ids(payload, "reactivate")
    wrong = sorted(set(deactivate) - active_before)
    if wrong:
        raise ContractError(f"deactivate lists intents that are not active: {wrong}")
    wrong = sorted(set(reactivate) - (set(intents) - active_before))
    if wrong:
        raise ContractError(f"reactivate lists intents that are not inactive: {wrong}")
    # A replaced intent is overwritten, so it is inactive whether or not it was listed.
    derived = (active_before - set(deactivate) - replaced) | set(reactivate) | set(key_to_id.values())

    names = [str(value) for value in _as_list(payload["active"], "active")]
    unknown = sorted(set(names) - set(intents) - set(key_to_id))
    if unknown:
        raise ContractError(f"active lists unknown ids or keys: {unknown}")
    stated = {key_to_id.get(name, name) for name in names}

    if derived != stated:
        raise ContractError(
            "active does not match the previous active list after the changes: "
            f"changes give {sorted(derived, key=_order_key)}, active gives {sorted(stated, key=_order_key)}"
        )
    constraints = [str(value).strip() for value in _as_list(payload["constraints"], "constraints") if str(value).strip()]
    update = GoalUpdate(new_intents=new_intents, active=sorted(stated, key=_order_key), constraints=constraints)
    apply_goal_update(intents, update)  # raises on anything the runner would reject
    return update


class IntentRefiner:
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

    def refine(
        self,
        user_turns: list[str],
        intents: dict[str, ConversationIntent],
        lifecycle: dict[str, Lifecycle],
    ) -> GoalUpdate:
        turn = len(user_turns)
        error: str | None = None
        for attempt in range(self.max_attempts):
            prompt = build_refine_prompt(user_turns, intents, lifecycle, repair_message=error)
            response = self.client.generate_json(model=self.model, system_instruction=REFINE_SYSTEM, prompt=prompt)
            if self.save_raw:
                self.save_raw(
                    f"refine/T{turn:02d}_{attempt:02d}.json",
                    {"prompt_version": PROMPT_VERSION, "format": FORMAT, "model": self.model, "prompt": prompt, "raw_text": response.raw_text},
                )
            try:
                return parse_refine_response(response.payload, intents=intents, lifecycle=lifecycle, turn=turn)
            except (ContractError, ValueError) as exc:
                error = str(exc)
        raise ContractError(f"intent refine failed after {self.max_attempts} attempts: {error}")

    def refine_turn(self, state: ConversationState, user_text: str) -> GoalUpdate:
        """Goal update for the next turn of a running conversation."""
        user_turns = [*(record.user_text for record in state.turns), user_text]
        return self.refine(user_turns, state.intents, state.lifecycle)


def _order_key(intent_id: str) -> tuple[int, str]:
    digits = intent_id[1:] if intent_id[:1] == "I" else ""
    return (int(digits), intent_id) if digits.isdigit() else (10**9, intent_id)


def _next_ids(intents: dict[str, ConversationIntent], count: int) -> list[str]:
    ids: list[str] = []
    number = len(intents)
    while len(ids) < count:
        number += 1
        if f"I{number}" not in intents:
            ids.append(f"I{number}")
    return ids


def _as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be a list")
    return value


def _list(payload: dict[str, Any], field: str) -> list[Any]:
    return _as_list(payload[field], field)


def _ids(payload: dict[str, Any], field: str) -> list[str]:
    ids = []
    for item in _list(payload, field):
        if not isinstance(item, dict) or "id" not in item:
            raise ContractError(f"each {field} entry needs an id: {item}")
        ids.append(str(item["id"]))
    if len(ids) != len(set(ids)):
        raise ContractError(f"{field} repeats an id: {ids}")
    return ids
