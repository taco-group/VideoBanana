"""Video quality of a system's outputs on ChatVE-Bench: TQ, VQR, PC, AES, MSM, and their change from the source.

Every output (and every source clip, as turn 0) is scored on its own, following CoVEBench. The metrics come from
their original implementations, which are not part of this repository (their licenses differ); install them
first, as described in the README:

  TQ   DOVER++ technical quality (0-100)       CoVEBench  metrics/vq/dover/eval_dover.py
  VQR  VisualQuality-R1, 10 frames (1-5)       CoVEBench  metrics/vq/vqr/eval_vqr.py
  AES  aesthetic predictor v2.5, 10 frames     CoVEBench  metrics/vq/aes/eval_aes.py
  MSM  motion smoothness, 10 frames (0-1)      CoVEBench  metrics/vq/msm/eval_msm.py
  PC   VideoCon-Physics, P(yes) (0-1)          VideoPhy   videocon/training/pipeline_video/entailment_inference.py

  python -m chatve_bench.vq --bench ChatVE-Bench --outputs "outputs/{session_id}/turn{turn:02d}.mp4" \\
      --out results/my_system --covebench /path/to/CoVEBench --videophy /path/to/videophy

Writes <out>/vq_scores.csv (one row per video; scored videos are kept, so a run resumes), <out>/vq_summary.json
and <out>/vq_summary.md. TQ and VQR sample at random, so repeated runs differ slightly.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
import uuid
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from .data import load_benchmark, output_video, source_clip

METRICS = ("tq", "vqr", "pc", "aes", "msm")
PC_PROMPT = ("The following is a conversation between a curious human and AI assistant. "
             "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
             "Human: <|video|>\nHuman: Does this video follow the physical laws?\nAI: ")


def _covebench_csv(args, script: str, extra: list[str], videos: list[Path], env: dict) -> dict[Path, float]:
    """Run one CoVEBench metric script on ``videos``; it needs numeric file names, so they are linked as 000000.mp4..."""
    with tempfile.TemporaryDirectory(prefix="chatve_vq_") as tmp:
        vdir = Path(tmp) / f"videos_{uuid.uuid4().hex[:8]}"
        vdir.mkdir()
        for i, v in enumerate(videos):
            (vdir / f"{i:06d}.mp4").symlink_to(v.resolve())
        out_csv = Path(tmp) / "scores.csv"
        cmd = [args.covebench_python, str(args.covebench / script), "--video-dir", str(vdir), "--output", str(out_csv), *extra]
        try:
            subprocess.run(cmd, check=True, cwd=args.covebench, env=env)
        finally:  # eval_dover.py leaves a config named after the video folder in the DOVER repository
            for cfg in args.covebench.glob(f"external/**/DOVER/dover_batch_{vdir.name}.yml"):
                cfg.unlink()
        scores = {}
        with out_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("score") or "").strip():
                    scores[videos[int(row["task_id"])]] = float(row["score"])
        return scores


def score_tq(args, videos):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    return _covebench_csv(args, "metrics/vq/dover/eval_dover.py", ["--device", f"cuda:{args.gpu}"], videos, env)


def score_vqr(args, videos):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    return _covebench_csv(args, "metrics/vq/vqr/eval_vqr.py", ["--frames", "10"], videos, env)


def score_aes(args, videos):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
    return _covebench_csv(args, "metrics/vq/aes/eval_aes.py", ["--device", "cuda:0", "--frames", "10"], videos, env)


def score_msm(args, videos):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump([str(v.resolve()) for v in videos], f)
    try:
        helper = Path(__file__).with_name("msm_helper.py")
        out = subprocess.run([args.covebench_python, str(helper), str(args.covebench), f.name],
                             check=True, stdout=subprocess.PIPE, text=True).stdout  # errors go to the terminal
    finally:
        os.unlink(f.name)
    by_path = json.loads(out.strip().splitlines()[-1])
    return {v: by_path[str(v.resolve())] for v in videos if str(v.resolve()) in by_path}


def score_pc(args, videos):
    script = args.videophy / "videocon/training/pipeline_video/entailment_inference.py"
    with tempfile.TemporaryDirectory(prefix="chatve_pc_") as tmp:
        in_csv, out_csv = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
        with in_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["videopath", "caption"])
            w.writerows([str(v.resolve()), PC_PROMPT] for v in videos)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
        # a local checkpoint folder is given relative to where the user runs this; otherwise a Hugging Face id
        ckpt = str(Path(args.videocon_checkpoint).resolve()) if Path(args.videocon_checkpoint).exists() else args.videocon_checkpoint
        subprocess.run([args.videophy_python, str(script), "--input_csv", str(in_csv), "--output_csv", str(out_csv),
                        "--checkpoint", ckpt, "--batch_size", "4"],
                       check=True, cwd=args.videophy.parent, env=env)
        by_path = {}
        with out_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                try:
                    by_path[str(Path(row[0]).resolve())] = float(row[-1])
                except (ValueError, IndexError):
                    continue  # header
    return {v: by_path[str(v.resolve())] for v in videos if str(v.resolve()) in by_path}


SCORERS = {"tq": score_tq, "vqr": score_vqr, "pc": score_pc, "aes": score_aes, "msm": score_msm}


def summarize(rows: list[dict], final_turn: dict[str, int]) -> dict:
    """Means over the final turns and over all turns, and the change of TQ and VQR from each session's source (%).
    Outputs of a session whose source clip has no complete scores are left out."""
    src = {r["session_id"]: r for r in rows if int(r["turn"]) == 0}
    rows = [r for r in rows if r["session_id"] in src]
    out = {"source": {m: sum(float(r[m]) for r in src.values()) / len(src) for m in METRICS} if src else {}}
    for name, keep in (("final_turn", lambda r: int(r["turn"]) == final_turn[r["session_id"]]),
                       ("all_turns", lambda r: int(r["turn"]) > 0)):
        sel = [r for r in rows if keep(r)]
        if not sel:
            continue
        s = {"videos": len(sel), **{m: sum(float(r[m]) for r in sel) / len(sel) for m in METRICS}}
        for m in ("tq", "vqr"):
            s[f"d{m}"] = 100 * sum(float(r[m]) / float(src[r["session_id"]][m]) - 1 for r in sel) / len(sel)
        out[name] = s
    return out


def markdown(summary: dict, name: str) -> str:
    def half_up(x, digits):
        return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))

    def pct(x):
        t = half_up(x, 0)
        return "0%" if t in ("0", "-0") else (t if t.startswith("-") else "+" + t) + "%"

    def line(label, s, delta=True):
        d = lambda k: pct(s[k]) if delta else "–"
        return (f"| {label} | {half_up(s['tq'], 2)} | {d('dtq')} | {half_up(s['vqr'], 2)} | {d('dvqr')} "
                f"| {half_up(s['pc'], 3)} | {half_up(s['aes'], 2)} | {half_up(s['msm'], 3)} |")
    lines = [f"# {name}: video quality on ChatVE-Bench", "",
             "| Slice | TQ | ΔTQ | VQR | ΔVQR | PC | AES | MSM |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if summary.get("source"):
        lines.append(line("Source clip", summary["source"], delta=False))
    lines += [line(label, summary[k]) for k, label in (("final_turn", "Final turn"), ("all_turns", "All turns")) if k in summary]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, type=Path, help="the downloaded ChatVE-Bench folder")
    ap.add_argument("--outputs", required=True, help="path template of the outputs, with {session_id} and {turn}")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--covebench", type=Path, help="a clone of CoVEBench with its weights (for TQ, VQR, AES, MSM)")
    ap.add_argument("--covebench-python", default=None, help="Python of CoVEBench's environment (default: its .venv)")
    ap.add_argument("--videophy", type=Path, help="a clone of VideoPhy (for PC)")
    ap.add_argument("--videophy-python", default="python", help="Python of VideoPhy's environment")
    ap.add_argument("--videocon-checkpoint", default="videophysics/videocon_physics")
    ap.add_argument("--metrics", default=",".join(METRICS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sessions", default=None, help="comma-separated session ids (default: all)")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args()
    wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if set(wanted) - set(METRICS):
        raise SystemExit(f"unknown metrics: {sorted(set(wanted) - set(METRICS))}; choose from {', '.join(METRICS)}")
    if set(wanted) & {"tq", "vqr", "aes", "msm"} and not args.covebench:
        raise SystemExit("--covebench is needed for TQ, VQR, AES and MSM")
    if "pc" in wanted and not args.videophy:
        raise SystemExit("--videophy is needed for PC")
    if args.covebench:
        args.covebench_python = args.covebench_python or str(args.covebench / ".venv/bin/python")

    sessions, _ = load_benchmark(args.bench)
    if args.sessions:
        keep = {s.strip() for s in args.sessions.split(",") if s.strip()}
        if keep - set(sessions):
            raise SystemExit(f"unknown session ids: {sorted(keep - set(sessions))}")
        sessions = {k: v for k, v in sessions.items() if k in keep}
    videos = {}  # (session_id, turn) -> path; turn 0 is the source clip
    for sid, s in sessions.items():
        videos[(sid, 0)] = source_clip(args.bench, s)
        for t in s["turns"]:
            p = output_video(args.outputs, sid, t["turn"])
            if p.is_file():
                videos[(sid, t["turn"])] = p
            elif not args.allow_missing:
                raise SystemExit(f"missing output: {p} (use --allow-missing to score a partial run)")

    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "vq_scores.csv"
    rows = {}
    if table.exists():
        with table.open(newline="", encoding="utf-8") as f:
            rows = {(r["session_id"], int(r["turn"])): r for r in csv.DictReader(f)}
    for k, v in videos.items():  # a video that changed since it was scored is scored again
        if k in rows and rows[k].get("video") != str(v):
            rows[k] = {"session_id": k[0], "turn": k[1], "video": str(v)}
    for m in wanted:
        todo = [k for k in videos if not (rows.get(k) or {}).get(m)]
        if not todo:
            continue
        print(f"{m}: {len(todo)} videos")
        scores = SCORERS[m](args, [videos[k] for k in todo])
        for k in todo:
            if videos[k] in scores:
                rows.setdefault(k, {"session_id": k[0], "turn": k[1], "video": str(videos[k])})[m] = scores[videos[k]]
        with table.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["session_id", "turn", *METRICS, "video"])
            w.writeheader()
            w.writerows(rows[k] for k in sorted(rows))

    # the summary covers this run's videos that have all five scores
    complete = [rows[k] for k in videos if k in rows and all(rows[k].get(m) not in (None, "") for m in METRICS)]
    final_turn = {sid: s["turns"][-1]["turn"] for sid, s in sessions.items()}
    summary = summarize(complete, final_turn)
    (args.out / "vq_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = markdown(summary, args.name or args.out.name)
    (args.out / "vq_summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
