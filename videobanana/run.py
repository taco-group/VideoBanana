"""Run VideoBanana on ChatVE-Bench sessions.

    python -m videobanana.run --bench ChatVE-Bench --out runs/videobanana \\
        --config configs/gemini.yaml [--sessions mt_0001 mt_0002] [--workers 8]

Per session and turn: intent refiner (LLM) -> entry selection over the
checkpoint tree -> instruction writer (LLM) -> intent compiler (LLM check) ->
CoinVE-Edit, verified after each attempt (VLM), at most two attempts -> commit. Sessions run in parallel
threads; the turns of one session run in order. Each session writes its state
to ``<out>/<session_id>/`` and its output after every turn to
``<out>/<session_id>/turn{NN}.mp4``, ready for ``chatve_bench.evaluate``.
Re-running the same command resumes interrupted sessions and skips finished
turns.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .conversation import ContractError, ConversationConfig, ConversationRunner, InstructionWriter, IntentRefiner, MultiTurnVerifier
from .conversation import compiler as compiler_module
from .conversation import instructions as instructions_module
from .conversation import refine as refine_module
from .conversation import verifier as verifier_module
from .conversation.compiler import IntentCompiler
from .conversation.refine import JsonModelClient, parse_refine_response
from .editor import CoinVEEditor, CoinVESettings, Editor, EditorTurnExecutor
from .models import parse_json_response

_LOCK = threading.Lock()


@dataclass(frozen=True)
class AgentSettings:
    """Models and turn-level settings of the agent."""

    agent_model: str
    verifier_model: str
    verify_fps: float = verifier_module.DEFAULT_MEDIA_FPS
    conversation: ConversationConfig = field(default_factory=ConversationConfig)


def load_config(path: str | Path) -> dict[str, Any]:
    """The YAML config with ``${VAR}`` references replaced by environment variables."""
    text = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    return yaml.safe_load(text) or {}


CONFIG_SECTIONS = {"models", "gemini", "openai", "editor", "conversation"}
MODEL_KEYS = {"client", "agent_model", "verifier_model", "verify_fps"}
PLACEHOLDERS = ("${", "YOUR_", "/path/to/")


def check_config(config: dict[str, Any]) -> list[str]:
    """Problems a typo or an unfilled template value would otherwise hide until the first call."""
    errors = [f"unknown config section: {key}" for key in config if key not in CONFIG_SECTIONS]
    errors += [f"unknown models key: {key}" for key in (config.get("models") or {}) if key not in MODEL_KEYS]

    def walk(value: Any, where: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{where}.{key}" if where else str(key))
        elif isinstance(value, list):
            for item in value:
                walk(item, where)
        elif isinstance(value, str) and any(mark in value for mark in PLACEHOLDERS):
            errors.append(f"{where} is not set: {value}")

    walk({key: value for key, value in config.items() if key in CONFIG_SECTIONS}, "")
    return errors


def agent_settings(config: dict[str, Any]) -> AgentSettings:
    models = config["models"]
    return AgentSettings(
        agent_model=models["agent_model"],
        verifier_model=models["verifier_model"],
        verify_fps=float(models.get("verify_fps", verifier_module.DEFAULT_MEDIA_FPS)),
        conversation=ConversationConfig(**(config.get("conversation") or {})),
    )


def build_client(config: dict[str, Any]) -> Any:
    """The model client named by ``models.client``: ``gemini`` or ``openai``."""
    kind = config["models"]["client"]
    if kind == "gemini":
        from .models.gemini import GeminiClient, GeminiSettings

        return GeminiClient(GeminiSettings(**(config.get("gemini") or {})))
    if kind == "openai":
        from .models.openai_compat import OpenAICompatClient

        options = dict(config.get("openai") or {})
        return OpenAICompatClient(options.pop("base_urls"), **options)
    raise ValueError(f"unknown models.client: {kind!r}")


def build_editor(config: dict[str, Any], log_dir: Path) -> CoinVEEditor:
    options = dict(config["editor"])
    options["gpus"] = tuple(options.get("gpus", (0,)))
    return CoinVEEditor(CoinVESettings(**options), log_dir)


def load_sessions(bench: Path) -> list[dict[str, Any]]:
    return json.loads((bench / "sessions.json").read_text(encoding="utf-8"))


def raw_saver(root: Path):
    """Saves every prompt and raw answer under ``<root>/raw/``."""

    def save(name: str, payload: Any) -> None:
        path = root / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    return save


def cached_update(root: Path, intents: dict[str, Any], lifecycle: dict[str, Any], turn: int, users: list[str]) -> Any | None:
    """The goal update an interrupted run already got from the refiner for this turn, if any.

    Reused only if it was refined from exactly these user messages with the
    same prompt version, so a resumed turn keeps its goal and its saved
    instructions and videos stay valid.
    """
    for path in sorted((root / "raw" / "refine").glob(f"T{turn:02d}_*.json"), reverse=True):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("prompt_version") != refine_module.PROMPT_VERSION:
            continue
        try:
            if [item["user"] for item in json.loads(raw["prompt"])["conversation"]] != users:
                continue
            return parse_refine_response(parse_json_response(raw["raw_text"]), intents=intents, lifecycle=lifecycle, turn=turn)
        except (ContractError, ValueError, KeyError):
            continue
    return None


def export_turn_videos(runner: ConversationRunner) -> None:
    """Copy each committed turn's output to ``turn{NN}.mp4`` next to the conversation state."""
    state = runner.load()
    for record in state.turns:
        target = runner.root / f"turn{record.turn:02d}.mp4"
        if target.is_file():
            continue
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copyfile(state.nodes[record.output_id].video_path, temporary)
        os.replace(temporary, target)


def run_session(
    session: dict[str, Any],
    *,
    bench: Path,
    out: Path,
    client: JsonModelClient,
    editor: Editor,
    settings: AgentSettings,
) -> dict[str, Any]:
    """Run (or resume) every turn of one ChatVE-Bench session."""
    session_id = session["session_id"]
    root = out / session_id
    save_raw = raw_saver(root)
    config = settings.conversation
    executor = EditorTurnExecutor(
        root / "turns",
        writer=InstructionWriter(client, model=settings.agent_model, save_raw=save_raw),
        verifier=MultiTurnVerifier(client, model=settings.verifier_model, media_fps=settings.verify_fps, save_raw=save_raw),
        editor=editor,
        compiler=IntentCompiler(client, model=settings.agent_model, save_raw=save_raw),
        gate_floor=config.gate_floor,
        item_gate_floor=config.item_gate_floor,
    )
    if (root / "conversation.json").is_file():
        runner = ConversationRunner(root, executor)
    else:
        source = (bench / session["clip"]).resolve()
        runner = ConversationRunner.create(root, source, executor, config=config, conversation_id=session_id)
    refiner = IntentRefiner(client, model=settings.agent_model, save_raw=save_raw)
    users = [turn["request"] for turn in session["turns"]]

    runner.resume()
    for number in range(len(runner.load().turns) + 1, len(users) + 1):
        state = runner.load()
        update = cached_update(root, state.intents, state.lifecycle, number, users[:number])
        if update is None:
            update = refiner.refine_turn(state, users[number - 1])
        record = runner.submit_turn(users[number - 1], update)
        export_turn_videos(runner)
        _log({"session": session_id, "turn": record.turn, "entry": record.entry_id, "execute": record.execute_ids,
              "preserve": record.preserve_ids, "output": record.output_id, "selection": record.selection})
    export_turn_videos(runner)
    return {"session": session_id, "status": "done", "turns": len(runner.load().turns)}


def _log(row: dict[str, Any]) -> None:
    with _LOCK:
        print(json.dumps(row, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", type=Path, required=True, help="ChatVE-Bench folder (sessions.json, clips/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="YAML config, e.g. configs/gemini.yaml")
    parser.add_argument("--sessions", nargs="*", default=None, help="session ids to run (default: all)")
    parser.add_argument("--workers", type=int, default=8, help="sessions run in parallel")
    args = parser.parse_args()

    config = load_config(args.config)
    problems = check_config(config)
    if problems:
        raise SystemExit("config: " + "; ".join(problems))
    settings = agent_settings(config)
    client = build_client(config)
    editor = build_editor(config, args.out / "logs")
    errors = client.preflight() + editor.preflight()
    if errors:
        raise SystemExit("; ".join(errors))
    all_sessions = load_sessions(args.bench)
    unknown = sorted(set(args.sessions or ()) - {item["session_id"] for item in all_sessions})
    if unknown:
        raise SystemExit(f"unknown session ids: {' '.join(unknown)}")
    sessions = [item for item in all_sessions if not args.sessions or item["session_id"] in args.sessions]
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "manifest.json"
    if not manifest.exists():
        manifest.write_text(json.dumps({
            "videobanana": __version__,
            "agent_model": settings.agent_model,
            "verifier_model": settings.verifier_model,
            "verify_fps": settings.verify_fps,
            "conversation_config": settings.conversation.model_dump(mode="json"),
            "prompts": [refine_module.PROMPT_VERSION, instructions_module.PROMPT_VERSION, compiler_module.PROMPT_VERSION,
                        verifier_module.PROMPT_VERSION],
        }, indent=2) + "\n", encoding="utf-8")

    def guarded(session: dict[str, Any]) -> dict[str, Any]:
        try:
            return run_session(session, bench=args.bench, out=args.out, client=client, editor=editor, settings=settings)
        except Exception:  # noqa: BLE001 - one broken session must not stop the others
            message = traceback.format_exc()
            print(f"{session['session_id']} FAILED\n{message}", flush=True)
            with _LOCK, (args.out / "failures.log").open("a", encoding="utf-8") as stream:
                stream.write(f"== {session['session_id']}\n{message}\n")
            return {"session": session["session_id"], "status": "failed"}

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            statuses = list(pool.map(guarded, sessions))
    finally:
        editor.close()
        getattr(client, "close", lambda: None)()
    counts = Counter(item["status"] for item in statuses)
    print(json.dumps(dict(counts)), flush=True)
    if counts.get("failed"):
        raise SystemExit(f"{counts['failed']} session(s) failed; see {args.out / 'failures.log'} and re-run to resume")


if __name__ == "__main__":
    main()
