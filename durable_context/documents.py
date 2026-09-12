"""Rendering and parsing for the two hot documents, plus staleness.

Both documents are Markdown because a human has to be able to read and hand-fix
them at 2am. Both are also machine-parsed, so every renderer here has a matching
parser and the pair is tested for round-trip equality: a document this package
wrote must be a document this package can read back without loss.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from .models import Checkpoint, Decision, Staleness

__all__ = [
    "render_checkpoint",
    "parse_checkpoint",
    "render_decision",
    "parse_decision",
    "staleness",
    "format_age",
    "DEFAULT_SECTIONS",
]

#: Suggested section order. Not enforced - projects carry their own vocabulary
#: and unknown sections round-trip untouched - but it is the shape that earned
#: its place in production: what is happening, what closed, what is still open,
#: what was decided and why, where the artefacts are, what will bite you.
DEFAULT_SECTIONS: tuple[str, ...] = (
    "NOW",
    "DONE THIS ARC",
    "OPEN THREADS",
    "DECISIONS AND WHY",
    "KEY ARTIFACTS",
    "GOTCHAS",
)

_HEADER = re.compile(r"^<!--\s*durable-context:(?P<body>.*?)-->\s*$", re.MULTILINE)
_KEYVAL = re.compile(r"(\w+)=([^\s]+)")
_SECTION = re.compile(r"^##\s+(?P<name>.+?)\s*$", re.MULTILINE)

#: A section body is free text, and free text contains Markdown. A body line
#: beginning ``## `` used to be read back as a *new* top-level section, quietly
#: re-attributing half of GOTCHAS to a phantom heading that ``--clear`` could
#: not reach. Render escapes such lines with a backslash and parse removes one,
#: which is lossless because the escape itself is escaped.
_NEEDS_ESCAPE = re.compile(r"^(\\*##\s)", re.MULTILINE)
_ESCAPED = re.compile(r"^\\(\\*##\s)", re.MULTILINE)

#: Header values live inside an HTML comment and are read back with a
#: whitespace-delimited key=value scan, so anything containing whitespace (or a
#: newline, or a comment terminator) would silently disable staleness detection.
_UNSAFE_HEADER_VALUE = re.compile(r"[^\w:.+-]+")

_DECISION_LABELS: tuple[tuple[str, str], ...] = (
    ("decision", "DECISION"),
    ("target", "TARGET"),
    ("why", "WHY"),
    ("discarded", "REJECTED"),
    ("open_thread", "OPEN"),
    ("next_step", "NEXT"),
)
_DECISION_LINE = re.compile(r"^-\s+\*\*(?P<label>[A-Z ]+):\*\*\s*(?P<value>.*)$", re.MULTILINE)

CHECKPOINT_TITLE = "# Session checkpoint"
DECISION_TITLE = "# Active decision"

DECISION_BANNER = (
    "<!-- Honour this. Do not re-open a choice recorded as REJECTED without a "
    "fresh instruction from the user. -->"
)


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _header_values(text: str) -> dict[str, str]:
    match = _HEADER.search(text or "")
    if not match:
        return {}
    return dict(_KEYVAL.findall(match.group("body")))


def _clean_header_value(value: str) -> str:
    """Make a value survive the header's key=value scan, whatever it contains.

    The trigger string comes from the host, unvalidated. One containing a
    newline broke the header regex outright and the checkpoint then read as
    undated - written seconds ago, injected as "age unknown, unverified".
    """
    cleaned = _UNSAFE_HEADER_VALUE.sub("-", str(value).strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned)
    return cleaned.strip("-")


def _render_header(**values: str) -> str:
    pairs = " ".join(
        f"{key}={_clean_header_value(value)}"
        for key, value in values.items()
        if value and _clean_header_value(value)
    )
    return f"<!-- durable-context: {pairs} -->"


def _escape_body(body: str) -> str:
    return _NEEDS_ESCAPE.sub(r"\\\1", body)


def _unescape_body(body: str) -> str:
    return _ESCAPED.sub(r"\1", body)


# --------------------------------------------------------------------------
# Checkpoint
# --------------------------------------------------------------------------


def render_checkpoint(checkpoint: Checkpoint) -> str:
    """Serialise the hot checkpoint, stamped so staleness can be judged later."""
    stamp = checkpoint.last_compact.isoformat(timespec="minutes") if checkpoint.last_compact else ""
    lines = [
        _render_header(last_compact=stamp, trigger=checkpoint.trigger or ""),
        CHECKPOINT_TITLE,
        "",
    ]
    for name, body in checkpoint.sections.items():
        lines.append(f"## {name}")
        body = _escape_body((body or "").strip("\n"))
        if body:
            lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def parse_checkpoint(text: str) -> Checkpoint:
    """Read a checkpoint back, preserving section order and unknown sections."""
    if not text or not text.strip():
        return Checkpoint()
    header = _header_values(text)
    sections: dict[str, str] = {}
    matches = list(_SECTION.finditer(text))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group("name")] = _unescape_body(text[start:end].strip("\n").strip())
    return Checkpoint(
        sections=sections,
        last_compact=_parse_timestamp(header.get("last_compact")),
        trigger=header.get("trigger", ""),
    )


# --------------------------------------------------------------------------
# Active decision
# --------------------------------------------------------------------------


def render_decision(decision: Decision, *, distilled_at: datetime | None = None) -> str:
    """Serialise the active decision, rejected path included and labelled."""
    stamp = distilled_at.isoformat(timespec="minutes") if distilled_at else ""
    lines = [
        _render_header(distilled=stamp, source=decision.source or ""),
        DECISION_BANNER,
        DECISION_TITLE,
        "",
    ]
    for attribute, label in _DECISION_LABELS:
        # One line per field, so a value spanning lines has to be folded into
        # one here. Written whole it parses back truncated at the first newline,
        # which loses the tail of the field without telling anyone.
        value = " ".join((getattr(decision, attribute) or "").split())
        if value:
            lines.append(f"- **{label}:** {value}")
    return "\n".join(lines).rstrip("\n") + "\n"


def parse_decision(text: str) -> tuple[Decision, datetime | None]:
    """Read the active decision back, with the time it was distilled."""
    if not text or not text.strip():
        return Decision(), None
    header = _header_values(text)
    label_to_attribute = {label: attribute for attribute, label in _DECISION_LABELS}
    values: dict[str, str] = {}
    for match in _DECISION_LINE.finditer(text):
        attribute = label_to_attribute.get(match.group("label").strip())
        if attribute:
            values[attribute] = match.group("value").strip()
    values["source"] = header.get("source", "")
    return Decision(**values), _parse_timestamp(header.get("distilled"))


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


def format_age(age: timedelta) -> str:
    total = int(age.total_seconds())
    if total < 60:
        return f"{total}s"
    if total < 3600:
        return f"{total // 60}m"
    if total < 86400:
        return f"{total // 3600}h{(total % 3600) // 60:02d}m"
    return f"{total // 86400}d{(total % 86400) // 3600:02d}h"


def staleness(checkpoint: Checkpoint, *, now: datetime, stale_after: timedelta) -> Staleness:
    """Judge whether the hot layer still describes the live world.

    Injected state that is silently old is worse than no state: the reader
    trusts it. So an unstamped or aged checkpoint is reported rather than
    quietly passed on, and a timestamp from the future is called out as clock
    skew instead of being treated as eternally fresh.
    """
    if checkpoint.last_compact is None:
        return Staleness(False, None, "no compaction timestamp recorded")
    age = now - checkpoint.last_compact
    if age.total_seconds() < 0:
        return Staleness(False, age, "timestamp is in the future (clock skew)")
    if age > stale_after:
        return Staleness(True, age, f"last compaction was {format_age(age)} ago")
    return Staleness(False, age, f"last compaction was {format_age(age)} ago")
