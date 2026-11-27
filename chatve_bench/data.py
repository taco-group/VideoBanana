"""Load ChatVE-Bench as released on Hugging Face (sessions.json, checklists.json, clips/)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


def load_benchmark(root: str | Path) -> tuple[dict[str, dict], list[dict]]:
    """Return the sessions by id and the per-turn checklists of a downloaded ChatVE-Bench folder."""
    root = Path(root)
    sessions = {s["session_id"]: s for s in json.loads((root / "sessions.json").read_text(encoding="utf-8"))}
    checklists = json.loads((root / "checklists.json").read_text(encoding="utf-8"))
    return sessions, checklists


def source_clip(root: str | Path, session: dict) -> Path:
    """The source clip of a session (Video A of every question that compares two videos)."""
    return Path(root) / session["clip"]


def output_video(template: str, session_id: str, turn: int) -> Path:
    """A system's output after one turn, from a path template such as ``outputs/{session_id}/turn{turn:02d}.mp4``."""
    return Path(template.format(session_id=session_id, turn=turn))


def frame_count(path: str | Path) -> int:
    """Number of decodable frames (counted by reading, not from the container header)."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    try:
        while cap.grab():
            n += 1
    finally:
        cap.release()
    return n
