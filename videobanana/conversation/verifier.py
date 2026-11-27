"""Verifier: score one candidate video against the whole goal.

One VLM call per candidate scores every intent the goal currently holds --
the ones sent to the editor this turn and the earlier edits the entry already
showed -- the same way: as an absolute state on Video 2. The model is not
told which is which. Whether a turn damaged an earlier edit is decided in code
(``scoring.damaged``) by comparing its new score with its score on the entry
node, which separates damage from an edit that was never really there. Render
quality and edit exclusivity are scored once per candidate.

Video 1 is the video the editor received (the entry), not the source. Scores
come back on a 1--4 scale and are stored as 0--3.

The rubric of the three dimensions follows the VEFX-Bench reward prompt
(Apache License 2.0), with the intents passed in the request and instruction
following scored per intent.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import ContractError, TurnSpec
from .refine import JsonModelClient

PROMPT_VERSION = "multiturn-verifier"
SCORE_MIN, SCORE_MAX = 1, 4
#: Video sampling rate sent with each video: 8 of the 49 frames of a 24 fps clip.
DEFAULT_MEDIA_FPS = 3.92

#: The single-turn verifier prompt the multi-turn prompt is derived from.
VERIFIER_SYSTEM = """You are an expert evaluator assessing the quality of AI-generated video edits. You will be provided with two videos:
- **Video 1**: The Original Video (before editing)
- **Video 2**: The Edited Video (after editing)

The editing intents are listed in the request. Your task is to evaluate the Edited Video across THREE independent dimensions. Each dimension is scored on a 1–4 integer scale. **Scores across dimensions are independent** — a failure in one dimension must NOT affect scores in another.

---

## Step 1: Survey the differences before you score anything
Write the survey first. Walk the frame region by region and list every place where Video 2 differs from Video 1: the requested edits, and everything else. Hunt for differences; do not summarise, and do not describe what stayed the same. Small, local and short-lived differences count and are the ones most often missed: leftover fragments of a removed object, invented content in blurred areas, a shift in sharpness or depth of field, colour or saturation drift on an object nobody asked you to change, flicker confined to a few frames. For each difference say where it is, when it happens, and whether an instruction asked for it.

---

## Dimension 1: Instructional Following (IF)
**Core question:** Does the edited video accurately reflect the semantic requirements of the editing instruction?

Score every intent separately. Do not let success on one intent compensate for failure on another.

Evaluation criteria:
- Object replacement: If the instruction says "replace apple with orange," did the model actually generate an orange (not a lemon or tomato)?
- Action/attribute changes: If the instruction involves motion or attribute changes (e.g., "make it night"), was this correctly executed?
- Completeness: Were ALL parts of the instruction addressed, not just partial execution?

Scoring rubric:
- **4 (Perfect):** The edit precisely and completely executes all instructions. Object categories, attributes (color, shape), actions, and styles all match the instruction with no ambiguity.
- **3 (High):** The main instruction was executed, but minor details deviate. E.g., instruction asks for "red sports car" but a "red truck" was generated — the main concept "red car" is correct.
- **2 (Low):** The main instruction was partially executed but with significant deviations, or completely irrelevant modifications were made.
- **1 (Failed):** The edit has no relation to the instruction. E.g., instruction asks for "night scene" but the video remains daytime, or no change occurred at all.

**Important notes:**
- If the edit instruction asks for a camera perspective change (e.g., "shift to high angle") and the video shows no actual perspective change, score 1.
- If the instruction asks for adding/increasing objects and no new objects appear, score 1.
- A video that looks identical to the original (no edit happened) always scores 1 for IF.

---

## Dimension 2: Render Quality (RQ)
**Core question:** What is the visual and temporal quality of the edited video?

Evaluation criteria:
- Naturalness and clarity: Are all parts of the video natural and sharp? Any blurriness, noise, or artifacts?
- Physical plausibility: Does object motion obey physics? Any flickering, jittering, objects disappearing/morphing unexpectedly?
- Temporal consistency: Is the video smooth frame-to-frame? Any sudden jumps, abrupt texture/color changes between frames?

Scoring rubric:
- **4 (Excellent):** Video clarity is very high with no visible defects, or only extremely minor artifacts detectable on very close inspection. Object motion fully obeys physics, smooth and natural. Visual quality is on par with or better than the original.
- **3 (Medium):** Some quality degradation exists (e.g., slight blurring, localized flickering), but all objects remain clearly identifiable. The video's overall structure is intact despite imperfections.
- **2 (Poor):** Significant quality degradation with obvious artifacts, distortion, or frame-to-frame inconsistency. Some object outlines deform, motion appears unnatural, affecting viewing experience.
- **1 (Unusable):** Video quality completely breaks down. Objects are severely deformed or unrecognizable, serious physics violations (e.g., person walking through walls, objects shattering spontaneously), heavy noise or complete blur.

**Important notes:**
- A sudden scene transition mid-video (e.g., white background abruptly becoming a construction site) counts as a physics/consistency violation — score ≤ 3.
- If the edit did NOT happen (original is preserved), RQ can still be high if the video itself looks fine — evaluate the video's visual quality independently.
- Evaluate temporal artifacts carefully: a single frame of flickering is minor (score 3), persistent warping or morphing is severe (score 1-2).

---

## Dimension 3: Edit Exclusivity (EE)
**Core question:** Did the model ONLY perform the specified edit, without making unintended changes to other parts of the video?

Evaluation criteria:
- Over-editing: When editing a foreground object, did the background, lighting, or other unrelated objects change?
- Scene consistency: Are pixels, textures, and structures in non-edited regions preserved?
- Camera trajectory: Was the original camera movement preserved? (Changing camera motion when not instructed is over-editing.)
- Identity preservation: Do unedited people maintain their facial features, expressions, and body movements?

Scoring rubric:
- **4 (Perfect):** Strict exclusivity maintained. Only the target region specified by the instruction changed. All other regions (background, unrelated objects) remain identical to the original. Tiny pixel-level differences invisible to the eye are acceptable.
- **3 (Medium):** Visible over-editing occurred. Non-target areas show noticeable changes, but overall scene layout and unrelated object consistency are still preserved. E.g., replaced a cup on a table but the table style also changed, or a background window disappeared.
- **2 (Poor):** The overall scene or multiple unrelated objects changed significantly.
- **1 (Complete failure):** No exclusivity at all. The entire video looks like a completely new video. The surrounding scene changed drastically, or more than three unrelated objects underwent serious alterations.

**Important notes:**
- Camera trajectory changes (when not instructed) are over-editing — if the original video had camera motion and the edited video is static (or vice versa), penalize EE.
- If the edit failed (IF=1) but the rest of the video also changed, EE should still be scored low.

---

## Explanations
4 is the full score. Every score needs an evidence-based explanation. A score of 4 means no defect; say briefly why. Any score below 4 must name every concrete defect that cost points: what is wrong, where in the frame, and when it happens. For Edit Exclusivity, also say where each unintended change sits relative to the requested edits, so it can be traced back."""

MULTITURN_ADDENDUM = """

---

## Multi-turn session
This edit is one step of a multi-turn editing session. Video 1 is exactly the video the editor received for this step and may already contain edits from earlier steps; it is not necessarily the original footage.

### Instructional Following for listed intents
Every listed intent is a target state the edited video should show. Some were requested for this step; others were achieved in earlier steps and should still hold. Score every one the same way, as a state shown in Video 2:
- Judge the state, not whether anything changed. If Video 2 shows the state, it counts even when Video 1 already showed it.
- Score accuracy as described in Dimension 1, and also how long the state holds: 4 it holds for the whole clip; 3 it holds for more than half of the clip, or drops out only in a few frames; 2 it holds for half of the clip or less; 1 it is not shown. Give the lower of the two.

### Undo statements (only if listed)
Each describes an earlier edit that is no longer wanted: it should hold in Video 1 and must no longer hold in Video 2. Score under Instructional Following: 4 it no longer holds and the area looks like the original scene (or like a listed intent on the same element); 3 it no longer holds apart from minor residue; 2 it partly still holds; 1 it still holds.

### Edit Exclusivity
Judge only content that no listed intent or undo statement is about: it must stay as in Video 1. Changes to what the listed intents describe are judged under Instructional Following, not here."""

#: Lines of the single-turn rubric that assume Video 1 is the untouched
#: source. Here Video 1 is the entry, which may already show earlier edits;
#: left as is, the model reads "unchanged from Video 1" as "never edited" and
#: scores kept edits 1.
_ENTRY_REWRITES = (
    (
        "- **Video 1**: The Original Video (before editing)",
        "- **Video 1**: The video the editor received for this step (it may already contain edits from earlier steps)",
    ),
    (
        '- **1 (Failed):** The edit has no relation to the instruction. E.g., instruction asks for "night scene" but the video remains daytime, or no change occurred at all.',
        '- **1 (Failed):** Video 2 does not show the requested state at all, or shows something unrelated to it. E.g., the instruction asks for a "night scene" but Video 2 shows daytime. Whether Video 2 differs from Video 1 does not decide this score.',
    ),
)


def _entry_system(single_turn: str) -> str:
    for old, new in _ENTRY_REWRITES:
        single_turn = single_turn.replace(old, new)
    return single_turn


VERIFY_SYSTEM = _entry_system(VERIFIER_SYSTEM) + MULTITURN_ADDENDUM


@dataclass(frozen=True)
class CandidateScores:
    instruction_scores: dict[str, int | None] = field(default_factory=dict)
    retention_scores: dict[str, int | None] = field(default_factory=dict)
    nontarget_score: int | None = None
    quality_score: int | None = None


def build_verify_prompt(spec: TurnSpec, *, repair_message: str | None = None) -> str:
    scored = {"score": f"integer {SCORE_MIN}-{SCORE_MAX}", "explanation": "..."}
    payload: dict[str, Any] = {
        "video_order": ["Video 1: the video the editor received for this step", "Video 2: the edited video"],
        "intents": [
            {"id": intent.intent_id, "target_state": intent.text}
            for intent in [*spec.execute, *spec.observe, *spec.preserve]
        ],
        # Entries never show unwanted edits, so there are no undo statements;
        # the field stays in the prompt.
        "undo": [],
        "task": "Return only a JSON object with exactly these fields.",
        "output_fields": {
            "survey": "string: the region-by-region list of differences",
            "instructional_following": {"<intent or undo id>": scored},
            "render_quality": scored,
            "edit_exclusivity": scored,
        },
    }
    if repair_message:
        payload["previous_answer_error"] = repair_message
    return json.dumps(payload, ensure_ascii=False, indent=1)


def _score(value: Any, where: str) -> int:
    if not isinstance(value, dict) or not isinstance(value.get("score"), int) or isinstance(value.get("score"), bool):
        raise ContractError(f"{where} needs an integer score")
    score = value["score"]
    if not SCORE_MIN <= score <= SCORE_MAX:
        raise ContractError(f"{where} score must be {SCORE_MIN}-{SCORE_MAX}, got {score}")
    return score - SCORE_MIN  # stored on the 0-3 scale


def parse_verdict(payload: Any, spec: TurnSpec) -> CandidateScores:
    if isinstance(payload, list) and len(payload) == 1:
        payload = payload[0]
    expected = {"survey", "instructional_following", "render_quality", "edit_exclusivity"}
    if not isinstance(payload, dict) or not expected <= set(payload):
        raise ContractError(f"answer must be a JSON object with fields {sorted(expected)}")
    answer = payload["instructional_following"] if isinstance(payload["instructional_following"], dict) else {}
    judged = [intent.intent_id for intent in [*spec.execute, *spec.observe, *spec.preserve]]
    if sorted(answer) != sorted(judged):
        raise ContractError(f"instructional_following must score exactly {judged}, got {sorted(answer)}")
    scores = {intent_id: _score(answer[intent_id], f"instructional_following.{intent_id}") for intent_id in judged}
    keep = {intent.intent_id for intent in spec.preserve}
    return CandidateScores(
        instruction_scores={key: value for key, value in scores.items() if key not in keep},
        retention_scores={key: value for key, value in scores.items() if key in keep},
        nontarget_score=_score(payload["edit_exclusivity"], "edit_exclusivity"),
        quality_score=_score(payload["render_quality"], "render_quality"),
    )


class MultiTurnVerifier:
    name = PROMPT_VERSION

    def __init__(
        self,
        client: JsonModelClient,
        *,
        model: str,
        media_fps: float = DEFAULT_MEDIA_FPS,
        max_attempts: int = 2,
        save_raw: Callable[[str, Any], Any] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.media_fps = media_fps
        self.max_attempts = max_attempts
        self.save_raw = save_raw
        self.calls = 0

    def identity(self) -> dict[str, object]:
        """What a saved verdict must match to be reused: prompt version, model and video sampling."""
        return {"name": self.name, "model": self.model, "media_fps": self.media_fps}

    def score_pair(self, spec: TurnSpec, entry_video: str, candidate_video: str, *, tag: str | None = None) -> CandidateScores:
        """Score ``candidate_video`` against ``entry_video`` for the roles in ``spec``."""
        error: str | None = None
        for _ in range(self.max_attempts):
            self.calls += 1
            prompt = build_verify_prompt(spec, repair_message=error)
            response = self.client.generate_json(
                model=self.model,
                system_instruction=VERIFY_SYSTEM,
                prompt=prompt,
                media_paths=[entry_video, candidate_video],
                media_fps=self.media_fps,
            )
            if self.save_raw:
                self.save_raw(
                    f"verify/{tag or f'T{spec.turn:02d}'}_{self.calls:03d}.json",
                    {
                        "prompt_version": PROMPT_VERSION,
                        "model": self.model,
                        "videos": [entry_video, candidate_video],
                        "media_fps": self.media_fps,
                        "prompt": prompt,
                        "raw_text": response.raw_text,
                    },
                )
            try:
                return parse_verdict(response.payload, spec)
            except (ContractError, ValueError) as exc:
                error = str(exc)
        # Unknown everywhere: the gate fails closed and the composite counts zeros.
        return CandidateScores(
            instruction_scores={intent.intent_id: None for intent in [*spec.execute, *spec.observe]},
            retention_scores={intent.intent_id: None for intent in spec.preserve},
        )
