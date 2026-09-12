"""The cold layer: an append-only decision log.

Nothing here rewrites history. The hot checkpoint is a view of *now* and is
overwritten constantly; the cold log is the record of how the project arrived
there, and a record you are willing to edit is not a record.

Only the tail is ever injected, so these functions are about writing a stable
entry format and choosing a safe tail.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Sequence

from .models import Decision
from .transcript import is_compaction_summary

__all__ = ["render_entry", "render_decision_entry", "select_tail"]


def render_entry(
    *,
    timestamp: datetime,
    kind: str,
    lines: Sequence[str],
    trigger: str = "",
) -> str:
    """Format one append-only block.

    Dated heading plus bullet lines - greppable by date, diff-friendly, and
    unambiguous about where one compaction's worth of intent ends and the next
    begins.
    """
    stamp = timestamp.isoformat(timespec="minutes")
    suffix = f" trigger={trigger}" if trigger else ""
    body = [f"\n## {stamp} {kind}{suffix}"]
    for line in lines:
        text = " ".join(str(line).split())
        if text:
            body.append(f"- {text}")
    return "\n".join(body) + "\n"


def render_decision_entry(decision: Decision, *, timestamp: datetime) -> str:
    """Append the distilled decision as one line, rejected path included."""
    parts = []
    for label, value in (
        ("DECISION", decision.decision),
        ("TARGET", decision.target),
        ("WHY", decision.why),
        ("REJECTED", decision.discarded),
        ("NEXT", decision.next_step),
    ):
        value = (value or "").strip()
        if value:
            parts.append(f"{label}: {value}")
    if not parts:
        return ""
    kind = f"decision (source={decision.source})" if decision.source else "decision"
    return render_entry(timestamp=timestamp, kind=kind, lines=[" | ".join(parts)])


def select_tail(
    lines: Iterable[str],
    *,
    max_lines: int,
    drop: Callable[[str], bool] = is_compaction_summary,
    dedupe: bool = True,
) -> list[str]:
    """Choose the tail of the cold log that is safe to inject.

    Filtering happens *before* the tail is taken, otherwise a run of junk lines
    silently evicts the real history from the window.

    ``dedupe`` keeps the most recent occurrence of a repeated line. A log that
    has accumulated the same block hundreds of times - which is what happens
    when a summary-recording loop goes unnoticed - would otherwise fill the
    entire injection budget with one sentence.
    """
    kept = [line for line in lines if not drop(line)]
    if dedupe:
        seen: set[str] = set()
        unique: list[str] = []
        for line in reversed(kept):
            key = line.strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            unique.append(line)
        kept = list(reversed(unique))
    if max_lines <= 0:
        return []
    return kept[-max_lines:]
