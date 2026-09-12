"""The orchestrator: wires pure rules to a store, a clock and a distiller.

Everything difficult already happened in :mod:`transcript`, :mod:`distill`,
:mod:`merge`, :mod:`documents` and :mod:`injection`. This class only sequences
them, which is why it takes its store, clock and distiller as constructor
arguments: the entire engine runs in memory, at a fixed instant, in tests.

The sequencing itself encodes the central claim of the package - that durability
must be *reactive*. :meth:`DurableContext.pre_compact` is driven by the host's
pre-compaction event, not by the agent remembering to save. An agent that
reliably remembered to write down its own decisions would not need this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Sequence

from .coldlog import render_decision_entry, render_entry, select_tail
from .config import Config
from .distill import Distiller, HeuristicDistiller
from .documents import (
    parse_checkpoint,
    parse_decision,
    render_checkpoint,
    render_decision,
    staleness,
)
from .injection import build_injection
from .merge import merge_checkpoint, merge_decision
from .models import Checkpoint, Decision, InjectionPayload, Message, Staleness
from .store import FileStore, Store
from .transcript import parse_transcript, user_messages

__all__ = ["DurableContext", "PreCompactResult"]


@dataclass(frozen=True)
class PreCompactResult:
    """What one pre-compaction pass actually preserved."""

    decision: Decision
    intent: tuple[str, ...]
    messages_seen: int
    distiller: str
    changed: bool

    def summary(self) -> str:
        if not self.changed:
            return f"nothing to preserve ({self.messages_seen} usable messages)"
        target = self.decision.target or self.decision.decision or "(unnamed)"
        return f"preserved decision on {target!r} via {self.distiller or 'none'}"


@dataclass
class DurableContext:
    """Read and write the durable layers for one state directory."""

    config: Config
    store: Store = field(default_factory=FileStore)
    distiller: Distiller = field(default_factory=HeuristicDistiller)
    clock: Callable[[], datetime] = datetime.now

    # -- reading ----------------------------------------------------------

    def load_checkpoint(self) -> Checkpoint:
        return parse_checkpoint(self.store.read(self.config.checkpoint_path))

    def load_decision(self) -> tuple[Decision, datetime | None]:
        return parse_decision(self.store.read(self.config.decision_path))

    #: Bytes of cold log read per line of tail requested. Log entries are one
    #: line each and rarely near this long, so the window comfortably contains
    #: the requested tail while keeping the read bounded.
    TAIL_BYTES_PER_LINE = 512

    def load_log_tail(self, max_lines: int | None = None) -> list[str]:
        """Read the tail of the append-only log, and only the tail.

        The log grows forever by design. Reading all of it to return twenty-five
        lines would put an unbounded cost in the session-start hook, so the file
        is read back-to-front within a byte budget derived from the line count.
        """
        limit = self.config.tail_lines if max_lines is None else max_lines
        if limit <= 0:
            return []
        budget = max(8192, limit * self.TAIL_BYTES_PER_LINE)
        read_tail = getattr(self.store, "read_tail", None)
        raw = read_tail(self.config.log_path, budget) if read_tail else self.store.read(
            self.config.log_path
        )
        if not raw:
            return []
        return select_tail(raw.splitlines(), max_lines=limit)

    def staleness(self, checkpoint: Checkpoint | None = None) -> Staleness:
        target = self.load_checkpoint() if checkpoint is None else checkpoint
        return staleness(target, now=self.clock(), stale_after=self.config.stale_after)

    # -- writing ----------------------------------------------------------

    def save_checkpoint(
        self,
        update: Checkpoint,
        *,
        merge: bool = True,
        allow_clear: Sequence[str] = (),
    ) -> Checkpoint:
        """Write the hot checkpoint.

        Merging is the default because the hot layer is overwritten in place: a
        caller that only knows about one section would otherwise delete every
        section it did not mention.
        """
        previous = self.load_checkpoint() if merge else None
        merged = merge_checkpoint(previous, update, allow_clear=allow_clear) if merge else update
        self.store.write(self.config.checkpoint_path, render_checkpoint(merged))
        return merged

    def save_decision(self, decision: Decision, *, merge: bool = True) -> Decision:
        """Write the active decision, keeping the rejected path sticky."""
        previous, _ = self.load_decision() if merge else (None, None)
        merged = merge_decision(previous, decision) if merge else decision
        now = self.clock()
        self.store.write(self.config.decision_path, render_decision(merged, distilled_at=now))
        entry = render_decision_entry(merged, timestamp=now)
        if entry:
            self.store.append(self.config.log_path, entry)
        return merged

    def append_intent(self, lines: Sequence[str], *, trigger: str = "") -> None:
        """Record raw human intent in the append-only log."""
        if not lines:
            return
        self.store.append(
            self.config.log_path,
            render_entry(
                timestamp=self.clock(), kind="intent", lines=lines, trigger=trigger
            ),
        )

    # -- the reactive entry point ----------------------------------------

    def pre_compact(
        self,
        transcript_lines: Iterable[str],
        *,
        trigger: str = "auto",
    ) -> PreCompactResult:
        """Preserve state immediately before the host compacts the context.

        Order matters. Raw intent is appended to the cold log *first*, so that
        even if distillation produces nothing usable the session's own words
        survive. Only then is the decision distilled, merged with what was
        already on record, and written to the hot layer.

        The checkpoint is stamped at the end whatever happens, so the next
        session can judge how old this state is instead of trusting it blindly.
        """
        messages: list[Message] = parse_transcript(
            transcript_lines, max_chars=self.config.max_message_chars
        )
        intent = [m.text for m in user_messages(messages)][-self.config.max_intent_lines :]
        self.append_intent(intent, trigger=trigger)

        distilled: Decision | None = None
        if messages:
            try:
                distilled = self.distiller.distill(messages)
            except Exception:
                # A distiller must never break compaction. Losing the distilled
                # decision costs nuance; raising here costs the whole session.
                distilled = None

        changed = False
        decision = Decision()
        if distilled and not distilled.is_empty():
            decision = self.save_decision(distilled)
            changed = True
        else:
            decision, _ = self.load_decision()

        now = self.clock()
        self.save_checkpoint(Checkpoint(sections={}, last_compact=now, trigger=trigger))

        return PreCompactResult(
            decision=decision,
            intent=tuple(intent),
            messages_seen=len(messages),
            distiller=getattr(self.distiller, "name", "") if changed else "",
            changed=changed or bool(intent),
        )

    # -- the session-start entry point -----------------------------------

    def session_start(self, *, max_chars: int | None = None) -> InjectionPayload:
        """Build the context block handed to a starting session."""
        checkpoint = self.load_checkpoint()
        decision, distilled_at = self.load_decision()
        age: timedelta | None = None
        if distilled_at is not None:
            age = self.clock() - distilled_at
            if age.total_seconds() < 0:
                age = None
        return build_injection(
            decision=decision,
            decision_age=age,
            checkpoint=checkpoint,
            staleness=self.staleness(checkpoint),
            log_tail=self.load_log_tail(),
            max_chars=self.config.max_injection_chars if max_chars is None else max_chars,
        )

    # -- introspection ----------------------------------------------------

    def status(self) -> dict:
        """A plain snapshot for the CLI and for humans debugging a hook."""
        checkpoint = self.load_checkpoint()
        decision, distilled_at = self.load_decision()
        verdict = self.staleness(checkpoint)
        return {
            "home": str(self.config.home),
            "checkpoint_sections": list(checkpoint.sections),
            "last_compact": checkpoint.last_compact.isoformat() if checkpoint.last_compact else None,
            "trigger": checkpoint.trigger,
            "stale": verdict.is_stale,
            "staleness_reason": verdict.reason,
            "decision": decision.to_dict(),
            "decision_distilled_at": distilled_at.isoformat() if distilled_at else None,
            # The tail that would be injected, not the size of the whole log:
            # counting every line of an append-only file that grows forever is
            # exactly the unbounded read this layer exists to avoid.
            "log_tail_lines": len(self.load_log_tail()),
            "distiller": getattr(self.distiller, "name", "unknown"),
        }
