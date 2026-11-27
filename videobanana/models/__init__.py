"""Model clients. Each answers ``generate_json(model=, system_instruction=, prompt=, media_paths=, media_fps=)``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class JsonResponse:
    raw_text: str
    payload: Any
    model: str


def parse_json_response(raw_text: str) -> Any:
    """The JSON value in a model answer, tolerating code fences and text around it."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_start, object_end = text.find("{"), text.rfind("}")
        array_start, array_end = text.find("["), text.rfind("]")
        candidates = []
        if object_start >= 0 and object_end > object_start:
            candidates.append(text[object_start : object_end + 1])
        if array_start >= 0 and array_end > array_start:
            candidates.append(text[array_start : array_end + 1])
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        raise
