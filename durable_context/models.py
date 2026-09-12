"""Immutable value types shared by every layer.

These are deliberately plain: no I/O, no clock, no logging. Everything that
matters about durable context is expressed as a transformation between these
values, which is what makes the interesting behaviour testable.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Mapping

__all__ = [
    "Message",
    "Decision",
    "Checkpoint",
    "Staleness",
    "InjectionPayload",
    "normalise_target",
]


_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalise_target(value: str) -> str:
    """Collapse a target string to a comparable key.

    Used to decide whether two decisions are about the *same* thing. Humans
    write "Postgres", "postgres.", and "  PostgreSQL " - only the first two
    should compare equal, so this is casefold + punctuation strip, nothing
    cleverer (no stemming, no fuzzy matching: a false "same target" silently
    merges two unrelated decisions, which is worse than a false "different").
    """
    return _NON_ALNUM.sub(" ", (value or "").strip().lower()).strip()


@dataclass(frozen=True)
class Message:
    """One real conversational turn, already stripped of transport noise."""

    role: str
    text: str

    @property
    def is_user(self) -> bool:
        return self.role == "user"

    def render(self) -> str:
        return f"{self.role.upper()}: {self.text}"


@dataclass(frozen=True)
class Decision:
    """The active decision: what is locked in right now, and what is not.

    ``discarded`` is the field that earns this whole package its keep. A
    summary that records "we are using X" lets a later session helpfully
    re-propose Y. A record that also says "Y was rejected" does not.
    """

    decision: str = ""
    target: str = ""
    why: str = ""
    discarded: str = ""
    open_thread: str = ""
    next_step: str = ""
    source: str = ""

    #: Fields that carry meaning; ``source`` is provenance, not content.
    CONTENT_FIELDS = ("decision", "target", "why", "discarded", "open_thread", "next_step")

    def is_empty(self) -> bool:
        return not any((getattr(self, name) or "").strip() for name in self.CONTENT_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Decision":
        """Build from loosely-shaped data (an LLM's JSON, a config file).

        Unknown keys are ignored and missing keys default to empty rather than
        raising: a distiller that returns five of six fields is still useful,
        and a hard failure here would cost the whole checkpoint.
        """
        if not data:
            return cls()
        clean: dict[str, str] = {}
        for name in (*cls.CONTENT_FIELDS, "source"):
            value = data.get(name)
            if value is None:
                continue
            clean[name] = " ".join(str(value).split())
        return cls(**clean)

    def with_source(self, source: str) -> "Decision":
        return replace(self, source=source)


@dataclass(frozen=True)
class Checkpoint:
    """The hot layer: overwritten every time, re-injected every session.

    ``sections`` is an ordered mapping of heading -> body. Headings are free
    text so a project can carry its own vocabulary; ordering is preserved so a
    render/parse round trip is stable.
    """

    sections: dict[str, str] = field(default_factory=dict)
    last_compact: datetime | None = None
    trigger: str = ""

    def is_empty(self) -> bool:
        return not any(body.strip() for body in self.sections.values())

    def section(self, name: str) -> str:
        return self.sections.get(name, "")


@dataclass(frozen=True)
class Staleness:
    """Verdict about how much the hot layer should be trusted."""

    is_stale: bool
    age: timedelta | None
    reason: str

    @property
    def is_unknown(self) -> bool:
        return self.age is None

    @property
    def is_future(self) -> bool:
        """Stamped ahead of the reading clock, so its age means nothing.

        A wrong system date, or state synced from a host whose clock is ahead,
        produces this. It is neither stale nor unknown, and a reader told
        nothing at all would treat it as fresh forever - the negative age
        short-circuits the staleness comparison permanently.
        """
        return self.age is not None and self.age.total_seconds() < 0


@dataclass(frozen=True)
class InjectionPayload:
    """What a session-start adapter hands to the host."""

    text: str
    truncated: bool = False
    blocks: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text.strip())
