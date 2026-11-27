"""Checklist judging: build one request per (turn, question type), parse the answers, and score them.

The request format, the four system prompts in ``prompts/``, the answer parsing, and the scoring rules are
adapted from the CoinVE-Bench evaluator (https://huggingface.co/datasets/FireCRT/CoinVE-Bench, CC BY-SA 4.0).
See ``NOTICE``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

QUESTION_TYPES = ("Single-TF", "AB-MCQ", "Dual-TF", "Score-MCQ")
# Dual-TF and Score-MCQ questions compare the source clip (Video A) with the output (Video B);
# the other two types see only the output.
TWO_VIDEO_TYPES = {"Dual-TF", "Score-MCQ"}
PROMPT_FILES = {"Single-TF": "Single-TF.txt", "AB-MCQ": "AB-MCQ.txt", "Dual-TF": "Dual-TF.txt",
                "Score-MCQ": "Score-MCQ-0-10.txt"}
YES = {"yes", "true", "1", "correct"}
NO = {"no", "false", "0", "incorrect"}


def load_prompts() -> dict[str, str]:
    here = Path(__file__).resolve().parent / "prompts"
    return {t: (here / f).read_text(encoding="utf-8") for t, f in PROMPT_FILES.items()}


def build_request(questions: list[dict], q_type: str, video_a: Path, video_b: Path) -> list[tuple[str, Any]]:
    """The user message of one request, as ("text", str) and ("video", path) parts.

    All questions of one type at one turn go into one request as a JSON array. Each question gets an
    opaque id (i0, i1, ...), because question ids repeat across the question groups of a turn.
    The user's request is not sent, so the judge cannot be biased by the target state.
    """
    items = []
    for k, q in enumerate(questions):
        item = {"id": f"i{k}", "question": q["question"]}
        if q.get("options"):  # the two choices of an AB-MCQ question (and of some others)
            item["options"] = q["options"]
        items.append(item)
    parts: list[tuple[str, Any]] = []
    if q_type in TWO_VIDEO_TYPES:
        parts += [("text", "Video A (Original Source):\n"), ("video", video_a),
                  ("text", "\nVideo B (Edited/Generated):\n"), ("video", video_b)]
    else:
        parts += [("text", "Video for evaluation (this is Video B):\n"), ("video", video_b)]
    n = len(items)
    parts.append(("text",
        f"\nBelow is a list of {n} independent questions you need to evaluate based "
        "on the video(s). Each question has a unique `id` (an opaque token like "
        "`i0`, `i1`, ...). Treat each question completely independently — do NOT "
        "let one answer bias another.\n\n"
        f"Questions:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        "CRITICAL OUTPUT REQUIREMENTS:\n"
        f"1. Output STRICTLY as a valid JSON array of EXACTLY {n} objects, "
        "one per input question, in the SAME order as the input.\n"
        "2. Every output object MUST carry the same `id` (verbatim, e.g. `i0`) as "
        "its corresponding input question so we can match them.\n"
        "3. Every output object MUST contain the fields required by the system prompt "
        "(e.g. `reasoning` and `final_answer` / `final_score`).\n"
        "4. Do NOT include any markdown formatting, prose, or explanations outside the JSON array."))
    return parts


def _json_payload(text: str) -> str:
    text = (text or "").strip()
    for pattern in (r"```(?:json)?\s*(.*?)\s*```", r"(\[.*\])", r"(\{.*\})"):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return text


def parse_answers(text: str, n: int) -> dict[int, dict]:
    """Answers by question index (0..n-1). Questions whose answer cannot be read are left out."""
    parsed = json.loads(_json_payload(text))
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    by_id, unlabeled = {}, []
    for item in parsed:
        if isinstance(item, dict):
            if item.get("id") is None:
                unlabeled.append(item)
            else:
                by_id[str(item["id"]).strip()] = item
    out = {}
    for k in range(n):
        item = by_id.get(f"i{k}") or (unlabeled[k] if k < len(unlabeled) else None)
        if item is None:
            continue
        answer = next((item[key] for key in ("final_answer", "final_score", "answer", "score") if item.get(key) is not None), None)
        if answer is not None:
            out[k] = {"answer": answer, "reasoning": item.get("reasoning", "")}
    return out


def _yes_no(value: Any) -> str:
    s = str(value).strip().lower()
    return "yes" if s in YES else "no" if s in NO else s


def _ab(value: Any) -> str:
    s = re.sub(r"\s+", " ", str(value).strip().upper())
    return "A AND B" if s in {"A AND B", "B AND A", "A, B", "B, A", "A,B", "B,A", "A&B", "B&A", "BOTH"} else s


def score(question: dict, answer: Any) -> float | None:
    """1/0 for the true-false and A/B questions; 0-10 for Score-MCQ (content preservation); None if unreadable."""
    if answer is None:
        return None
    if question["type"] == "Score-MCQ":
        try:
            return max(0.0, min(10.0, float(answer)))
        except (TypeError, ValueError):
            return None
    if question["type"] in ("Single-TF", "Dual-TF"):
        return float(_yes_no(answer) == _yes_no(question["expected_answer"]))
    if question["type"] == "AB-MCQ":
        return float(_ab(answer) == _ab(question["expected_answer"]))
    return None
