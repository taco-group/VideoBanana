"""Paper metrics from scored questions: SA, SPA, EP, their mean EA (editing accuracy), and CP (content preservation).

Within a slice every metric pools its questions over all turns of the slice: an accuracy metric is the share of
correct answers (x100), CP is the mean 0-10 score (x10). EA is the mean of SA, SPA and EP. Unanswered questions
are left out. The rounding follows the CoinVE-Bench evaluator (4 decimals) so the paper's numbers reproduce.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import ROUND_HALF_UP, Decimal

ACCURACY = {"Semantic Accuracy": "SA", "Scope Accuracy": "SPA", "Edit Persistence": "EP"}
PRESERVATION = {"Content Preservation": "CP"}


def summarize(rows: list[dict]) -> dict:
    """Metrics of one slice. ``rows``: scored questions with keys ``metric`` and ``score`` (None if unanswered)."""
    scores = defaultdict(list)
    total = 0
    for row in rows:
        if row["metric"] not in ACCURACY and row["metric"] not in PRESERVATION:
            continue
        total += 1
        if row["score"] is not None:
            scores[row["metric"]].append(row["score"])
    out = {"questions": total, "answered": sum(len(v) for v in scores.values())}
    for metric, key in ACCURACY.items():
        v = scores[metric]
        out[key] = round(sum(s == 1.0 for s in v) / len(v) * 100.0, 4) if v else None
    v = scores["Content Preservation"]
    out["CP"] = round(sum(v) / len(v) * 10.0, 4) if v else None
    parts = [out[k] for k in ("SA", "SPA", "EP") if out[k] is not None]
    out["EA"] = round(sum(parts) / len(parts), 4) if parts else None
    return out


def report(rows: list[dict], final_turn: dict[str, int]) -> dict:
    """The two slices of the benchmark: the final turn of every session, and all turns."""
    return {
        "final_turn": summarize([r for r in rows if r["turn"] == final_turn[r["session_id"]]]),
        "all_turns": summarize(rows),
    }


def markdown(summary: dict, name: str) -> str:
    def line(label: str, m: dict) -> str:
        cell = lambda x: "–" if x is None else str(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        return (f"| {label} | {cell(m['SA'])} | {cell(m['SPA'])} | {cell(m['EP'])} | {cell(m['EA'])} | {cell(m['CP'])} "
                f"| {m['answered']} / {m['questions']} |")
    head = ["| Slice | SA | SPA | EP | EA | CP | Answered |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines = [f"# {name} on ChatVE-Bench", "", *head, line("Final turn", summary["final_turn"]),
             line("All turns", summary["all_turns"])]
    if summary.get("missing_turns"):
        lines += ["", f"Not scored (no output): {', '.join(summary['missing_turns'])}"]
    return "\n".join(lines) + "\n"
