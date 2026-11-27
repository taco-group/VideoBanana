"""Records of the multi-turn layer: the goal, the checkpoint tree, turn specs and verdicts.

Scores are stored on a 0--3 scale: the verifier answers on 1--4 and one is
subtracted, so a floor of 2 here is a score of 3 from the verifier.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractError(RuntimeError):
    """A model answer or a goal update that breaks the rules of the conversation layer."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImmutableModel(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Lifecycle = Literal["active", "inactive"]


class ConversationConfig(ImmutableModel):
    """Turn-level settings. The defaults are the ones used in the paper."""

    schema_version: Literal[1] = 1
    #: Deepest generation lineage (D) a turn output may have before the entry
    #: selector restarts from a shallower node.
    max_depth: int = Field(default=3, ge=1)
    #: Instructions per editor call (K); CoinVE-Edit was trained on 2--5.
    max_instructions_per_call: int = Field(default=5, ge=1)
    #: Cross-turn merging (K_merge): the selector prefers the entry with the
    #: shallowest output as long as the call stays within this many
    #: instructions, even if that redoes edits an earlier output already has.
    merge_max_instructions: int = Field(default=2, ge=1)
    #: Total instructions per call after exhausted intents are added as
    #: observe items.
    observe_fill_total: int = Field(default=3, ge=0)
    #: Cross-turn failures after which an intent stops being a gate target.
    exhaust_after_failures: int = Field(default=2, ge=1)
    #: Editor attempts per turn: the first try plus at most one resample.
    max_attempts_per_turn: int = Field(default=2, ge=1)
    #: Gate floor on the 0-3 scale (2 = a verifier score of 3 out of 4).
    gate_floor: int = Field(default=2, ge=0, le=3)
    #: Floor on intent scores alone (this turn's edits in the gate, and what the
    #: ledger counts as shown); render quality and edit exclusivity keep
    #: ``gate_floor``. ``None`` means ``gate_floor``. Set per verifier model.
    item_gate_floor: int | None = Field(default=None, ge=0, le=3)
    #: An earlier edit counts as damaged by a turn when its score drops by at
    #: least this many levels from its score on the entry node.
    damage_drop: int = Field(default=2, ge=1, le=3)

    @model_validator(mode="after")
    def fill_within_call(self) -> "ConversationConfig":
        if self.observe_fill_total > self.max_instructions_per_call:
            raise ValueError("observe_fill_total cannot exceed max_instructions_per_call")
        if self.merge_max_instructions > self.max_instructions_per_call:
            raise ValueError("merge_max_instructions cannot exceed max_instructions_per_call")
        return self

    @property
    def item_floor(self) -> int:
        return self.gate_floor if self.item_gate_floor is None else self.item_gate_floor


class ConversationIntent(ImmutableModel):
    """One atomic edit goal. Records are immutable: a changed desired state is a new intent."""

    intent_id: str = Field(min_length=1)
    introduced_turn: int = Field(ge=1)
    #: Absolute, entry-independent description of what the output shows
    #: ("The grinder's outer casing is dark blue."), never history-relative
    #: wording such as "undo", "instead" or "again".
    text: str = Field(min_length=1)
    #: Earlier intents on the same element and attribute that this one
    #: overwrites ("dark blue instead" over a white casing).
    replaces: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def no_self_reference(self) -> "ConversationIntent":
        if self.intent_id in self.replaces:
            raise ValueError(f"{self.intent_id} cannot replace itself")
        return self


class GoalUpdate(ImmutableModel):
    """The goal after one user turn, stated in full rather than as a delta.

    ``active`` lists every intent that should hold in the turn output, reusing
    earlier ids for goals that are unchanged or return to an earlier state;
    everything else is inactive. ``constraints`` are requests that add no new
    information and only stress that something stays as it is; they are
    recorded but not used.
    """

    new_intents: list[ConversationIntent] = Field(default_factory=list)
    active: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class Node(ImmutableModel):
    """A checkpoint in the conversation tree: the source or one turn output."""

    node_id: str
    parent_id: str | None
    video_path: str
    #: Editor calls between the source and this video.
    depth: int = Field(ge=0)
    turn: int = Field(ge=0)
    sub_session_id: str | None = None


class TurnSpec(ImmutableModel):
    """Everything an executor needs to run one turn from the chosen entry."""

    conversation_id: str
    turn: int = Field(ge=1)
    entry: Node
    execute: list[ConversationIntent]
    preserve: list[ConversationIntent]
    observe: list[ConversationIntent]
    #: Inactive edits the entry still shows that an execute intent overwrites;
    #: instruction writing needs them to phrase the change from the entry.
    overwritten: list[ConversationIntent] = Field(default_factory=list)
    #: Scores (0-3) of the preserve intents on the entry node. A turn damages
    #: an earlier edit only when its new score drops by ``damage_drop`` levels
    #: from this reference.
    reference_scores: dict[str, int | None] = Field(default_factory=dict)
    damage_drop: int = Field(default=2, ge=1, le=3)
    #: The user turn that introduced each execute / observe intent, so that
    #: instruction writing can recover details the intent text dropped.
    user_requests: dict[str, str] = Field(default_factory=dict)
    max_attempts: int = Field(ge=1)
    max_instructions_per_call: int = Field(ge=1)


class CandidateReport(ImmutableModel):
    """Verdicts for one candidate video. Scores are 0--3; ``None`` means unknown."""

    candidate_id: str
    video_path: str
    #: Editor calls in this candidate's lineage from the entry.
    generations: int = Field(default=1, ge=1)
    #: False when the editor produced no usable video.
    structural_passed: bool = True
    #: Scores of the execute and observe intents.
    instruction_scores: dict[str, int | None] = Field(default_factory=dict)
    #: Scores of the preserve intents, on the same absolute scale; damage is
    #: judged against ``TurnSpec.reference_scores``.
    retention_scores: dict[str, int | None] = Field(default_factory=dict)
    nontarget_score: int | None = None
    quality_score: int | None = None


class TurnResult(ImmutableModel):
    candidates: list[CandidateReport] = Field(default_factory=list)
    sub_session_id: str | None = None


class PendingTurn(ImmutableModel):
    """A turn whose goal update and roles are fixed but that is not committed yet."""

    turn: int = Field(ge=1)
    user_text: str
    spec: TurnSpec
    exhausted_ids: list[str]
    result: TurnResult | None = None


class TurnRecord(ImmutableModel):
    turn: int = Field(ge=1)
    user_text: str
    entry_id: str
    execute_ids: list[str]
    preserve_ids: list[str]
    observe_ids: list[str]
    exhausted_ids: list[str]
    output_id: str
    selection: Literal["gate_pass", "composite", "previous_output", "reuse_entry"]
    selected_candidate_id: str | None = None
    gate_passed: bool
    primary_score: float | None = None
    secondary_score: float | None = None


class ConversationState(StrictModel):
    schema_version: Literal[1] = 1
    conversation_id: str
    config: ConversationConfig
    phase: Literal["awaiting_turn", "planned", "executed"] = "awaiting_turn"
    intents: dict[str, ConversationIntent] = Field(default_factory=dict)
    lifecycle: dict[str, Lifecycle] = Field(default_factory=dict)
    #: Preservation-only requests from the latest goal update (recorded only).
    constraints: list[str] = Field(default_factory=list)
    failure_counts: dict[str, int] = Field(default_factory=dict)
    nodes: dict[str, Node]
    #: node_id -> intent_id -> shown. Written only from verification; a
    #: missing cell means "never judged" and reads as absent.
    ledger: dict[str, dict[str, bool]] = Field(default_factory=dict)
    #: node_id -> intent_id -> score (0-3) behind the ledger bit: judged when
    #: the node was committed, or inherited from its entry.
    scores: dict[str, dict[str, int | None]] = Field(default_factory=dict)
    turns: list[TurnRecord] = Field(default_factory=list)
    pending: PendingTurn | None = None

    @model_validator(mode="after")
    def pending_matches_phase(self) -> "ConversationState":
        if (self.phase == "awaiting_turn") != (self.pending is None):
            raise ValueError("pending turn must exist exactly when a turn is in progress")
        if self.phase == "executed" and (self.pending is None or self.pending.result is None):
            raise ValueError("executed phase requires a recorded turn result")
        return self

    @property
    def source_id(self) -> str:
        return next(node.node_id for node in self.nodes.values() if node.parent_id is None)

    @property
    def last_output_id(self) -> str:
        return self.turns[-1].output_id if self.turns else self.source_id
