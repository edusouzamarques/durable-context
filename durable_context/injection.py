"""Assembling what a new session is told, as a pure function.

This is the payoff step: the moment where the durable state becomes context the
next session actually reads. It takes values in and returns text out - no files,
no clock - so the ordering and truncation rules can be pinned by tests.

Two rules matter here:

* **The active decision goes first.** It is the smallest block and the one that
  prevents the specific failure this package exists for: an agent re-proposing a
  path the user already rejected.
* **Truncation drops from the bottom.** When the budget is tight the append-only
  history is sacrificed before the live decision, never the other way round.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Sequence

from .documents import format_age
from .models import Checkpoint, Decision, InjectionPayload, Staleness

__all__ = [
    "build_injection",
    "DECISION_HEADER",
    "CHECKPOINT_HEADER",
    "LOG_HEADER",
]

DECISION_HEADER = (
    "=== ACTIVE DECISION (distilled before the last compaction) ===\n"
    "This is the target currently locked in. Honour it: do not re-open a choice "
    "recorded below as REJECTED, and do not change direction without a fresh "
    "instruction from the user. If live state contradicts this, live state wins - "
    "but say so rather than silently re-deciding."
)

CHECKPOINT_HEADER = (
    "=== SESSION CHECKPOINT (state that compaction does not preserve) ===\n"
    "Read this before acting. Do not ask the user for anything already recorded here."
)

LOG_HEADER = (
    "=== DECISION LOG, most recent entries (append-only history) ===\n"
    "Background on how the project reached its current position."
)

_STALE_NOTE = (
    "WARNING: this checkpoint is {age} old and may no longer describe the live "
    "world. Verify against the real source before asserting anything from it."
)

_SKEW_NOTE = (
    "WARNING: this checkpoint is stamped {age} in the future (clock skew), so "
    "its age cannot be judged. Treat it as unverified and check the live source."
)


def _decision_block(decision: Decision, age: timedelta | None) -> str:
    lines = [DECISION_HEADER]
    if age is not None and age.total_seconds() >= 0:
        lines.append(f"(distilled {format_age(age)} ago, source={decision.source or 'unknown'})")
    for label, value in (
        ("DECISION", decision.decision),
        ("TARGET", decision.target),
        ("WHY", decision.why),
        ("REJECTED", decision.discarded),
        ("OPEN", decision.open_thread),
        ("NEXT", decision.next_step),
    ):
        text = (value or "").strip()
        if text:
            lines.append(f"- {label}: {text}")
    return "\n".join(lines)


def _checkpoint_block(checkpoint: Checkpoint, staleness: Staleness | None) -> str:
    lines = [CHECKPOINT_HEADER]
    if staleness is not None:
        if staleness.is_stale and staleness.age is not None:
            lines.append(_STALE_NOTE.format(age=format_age(staleness.age)))
        elif staleness.is_future and staleness.age is not None:
            lines.append(_SKEW_NOTE.format(age=format_age(-staleness.age)))
        elif staleness.is_unknown:
            lines.append(
                "NOTE: this checkpoint carries no timestamp, so its age is unknown. "
                "Treat it as unverified."
            )
    for name, body in checkpoint.sections.items():
        body = (body or "").strip()
        if not body:
            continue
        lines.append(f"## {name}")
        lines.append(body)
    return "\n".join(lines)


def _log_block(tail: Sequence[str]) -> str:
    body = "\n".join(line.rstrip() for line in tail if line.strip())
    return f"{LOG_HEADER}\n{body}" if body else ""


def build_injection(
    *,
    decision: Decision | None = None,
    decision_age: timedelta | None = None,
    checkpoint: Checkpoint | None = None,
    staleness: Staleness | None = None,
    log_tail: Sequence[str] = (),
    max_chars: int = 12000,
) -> InjectionPayload:
    """Render the session-start context, in priority order, within a budget.

    Blocks are emitted highest-priority first and the first block that does not
    fit ends the payload. The alternative - skipping an oversized block and
    letting a smaller, less important one through - would silently reorder
    importance at exactly the moment the reader can least afford it.

    An empty result is a legitimate outcome (fresh install, nothing recorded
    yet) and callers are expected to inject nothing rather than an empty header.
    """
    blocks: list[str] = []
    if decision is not None and not decision.is_empty():
        blocks.append(_decision_block(decision, decision_age))
    if checkpoint is not None and not checkpoint.is_empty():
        block = _checkpoint_block(checkpoint, staleness)
        if block.strip():
            blocks.append(block)
    log_block = _log_block(log_tail)
    if log_block:
        blocks.append(log_block)

    if max_chars <= 0:
        return InjectionPayload(text="", truncated=bool(blocks), blocks=())

    kept: list[str] = []
    used = 0
    truncated = False
    for block in blocks:
        cost = len(block) + (2 if kept else 0)
        if used + cost <= max_chars:
            kept.append(block)
            used += cost
            continue
        truncated = True
        if not kept:
            # Nothing fits, not even the top block: keep a hard-truncated head
            # rather than returning nothing, because a partial active decision
            # still names the locked target.
            kept.append(block[:max_chars])
        break

    return InjectionPayload(text="\n\n".join(kept), truncated=truncated, blocks=tuple(kept))
