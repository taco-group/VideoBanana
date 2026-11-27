"""Score a system's outputs on ChatVE-Bench.

The system's output after every turn is one video with the source clip's frame count (49 frames; one short clip has
41), found by a
path template with {session_id} and {turn}. Every turn's checklist is judged; answers are saved as they arrive,
so an interrupted run resumes where it stopped.

  python -m chatve_bench.evaluate --bench ChatVE-Bench --outputs "outputs/{session_id}/turn{turn:02d}.mp4" \\
      --out results/my_system

Writes <out>/answers.jsonl (one judged question per line), <out>/summary.json and <out>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import judge, metrics
from .backends import GeminiBackend, OpenAIBackend
from .data import frame_count, load_benchmark, output_video, source_clip


def question_key(sid: str, turn: int, group: dict, q: dict) -> str:
    return f"{sid}|{turn}|{group['intent_id'] or ''}|{group['set']}|{q['id']}"


def build_tasks(bench: Path, sessions: dict, checklists: list, template: str, allow_missing: bool):
    """One task per (turn, question type); turns whose output is missing are skipped if allowed."""
    tasks, missing = [], []
    src_frames = {}
    for case in checklists:
        sid, turn = case["session_id"], case["turn"]
        video_a, video_b = source_clip(bench, sessions[sid]), output_video(template, sid, turn)
        if not video_b.is_file():
            if not allow_missing:
                raise SystemExit(f"missing output: {video_b} (use --allow-missing to score a partial run)")
            missing.append((sid, turn))
            continue
        if sid not in src_frames:
            src_frames[sid] = frame_count(video_a)
        n = frame_count(video_b)
        if n != src_frames[sid]:
            raise SystemExit(f"{video_b}: {n} frames, the source clip has {src_frames[sid]}; cut outputs to the clip")
        by_type = defaultdict(list)
        for g in case["question_groups"]:
            for q in g["questions"]:
                by_type[q["type"]].append((question_key(sid, turn, g, q), g, q))
        for q_type, items in by_type.items():
            tasks.append({"session_id": sid, "turn": turn, "type": q_type, "video_a": video_a, "video_b": video_b,
                          "items": items})
    return tasks, missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, type=Path, help="the downloaded ChatVE-Bench folder")
    ap.add_argument("--outputs", required=True, help="path template of the outputs, with {session_id} and {turn}")
    ap.add_argument("--out", required=True, type=Path, help="folder for answers and summary")
    ap.add_argument("--name", default=None, help="system name in the report (default: the --out folder name)")
    ap.add_argument("--backend", choices=("gemini", "openai"), default="gemini")
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="openai backend: server address")
    ap.add_argument("--frames", type=int, default=8, help="frames per clip shown to the judge (paper: 8)")
    ap.add_argument("--workers", type=int, default=8, help="requests in parallel")
    ap.add_argument("--retries", type=int, default=2,
                    help="times a question without a usable answer is asked again (same turn, same video)")
    ap.add_argument("--sessions", default=None, help="comma-separated session ids to score (default: all)")
    ap.add_argument("--allow-missing", action="store_true", help="score the turns whose output exists")
    args = ap.parse_args()

    sessions, checklists = load_benchmark(args.bench)
    if args.sessions:
        keep = {s.strip() for s in args.sessions.split(",") if s.strip()}
        if keep - set(sessions):
            raise SystemExit(f"unknown session ids: {sorted(keep - set(sessions))}")
        checklists = [c for c in checklists if c["session_id"] in keep]
    tasks, missing = build_tasks(args.bench, sessions, checklists, args.outputs, args.allow_missing)
    args.out.mkdir(parents=True, exist_ok=True)
    # answers are reused only by a run with the same outputs and the same judge
    config = {"outputs": str(Path(args.outputs).resolve()), "backend": args.backend, "model": args.model,
              "frames": args.frames}
    config_path = args.out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise SystemExit(f"{args.out} holds answers of another run ({config_path}); use another --out")
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    answers_path = args.out / "answers.jsonl"
    # an answer counts once it can be scored; anything else is asked again in the next retry
    usable = lambda rec: judge.score({"type": rec["type"], "expected_answer": rec["expected_answer"]}, rec["answer"]) is not None
    done = {}
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:  # a line cut off when an earlier run was stopped
                continue
            if usable(rec):
                done[rec["key"]] = rec
    backend = (GeminiBackend(args.model, args.frames) if args.backend == "gemini"
               else OpenAIBackend(args.model, args.base_url, args.frames))
    prompts, lock = judge.load_prompts(), threading.Lock()

    def run(task: dict) -> int:
        items = [it for it in task["items"] if it[0] not in done]
        if not items:
            return 0
        parts = judge.build_request([q for _, _, q in items], task["type"], task["video_a"], task["video_b"])
        parsed = judge.parse_answers(backend.generate(prompts[task["type"]], parts), len(items))
        with lock, answers_path.open("a", encoding="utf-8") as f:
            for k, (key, g, q) in enumerate(items):
                if k in parsed and judge.score(q, parsed[k]["answer"]) is not None:
                    rec = {"key": key, "session_id": task["session_id"], "turn": task["turn"],
                           "intent_id": g["intent_id"], "set": g["set"], "question_id": q["id"], "type": q["type"],
                           "metric": q["metric"], "expected_answer": q["expected_answer"],
                           "answer": parsed[k]["answer"], "reasoning": parsed[k]["reasoning"]}
                    done[key] = rec
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(parsed)

    for attempt in range(args.retries + 1):
        pending = [t for t in tasks if any(it[0] not in done for it in t["items"])]
        if not pending:
            break
        print(f"{'first pass' if attempt == 0 else f'retry {attempt}'}: {len(pending)} requests")
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run, t) for t in pending]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001 - unanswered questions are asked again in the next retry
                    print(f"  request failed: {e}")

    final_turn = {sid: s["turns"][-1]["turn"] for sid, s in sessions.items()}
    rows = []
    for t in tasks:
        for key, g, q in t["items"]:
            rec = done.get(key)
            rows.append({"session_id": t["session_id"], "turn": t["turn"], "metric": q["metric"],
                         "score": judge.score(q, rec["answer"]) if rec else None})
    summary = metrics.report(rows, final_turn)
    summary["missing_turns"] = [f"{sid} T{turn}" for sid, turn in missing]
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = metrics.markdown(summary, args.name or args.out.name)
    (args.out / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
