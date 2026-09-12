"""Pure transcript parsing.

A session transcript is JSON Lines. Each line is an envelope; the part we care
about is ``message.role`` and ``message.content``, where content is either a
string or a list of typed blocks. Everything in this module takes raw lines and
returns :class:`~durable_context.models.Message` values - no file handles, no
paths, no encoding guesses. The filesystem edge lives in ``store.py``.

The filters here are not cosmetic. Two of them exist because their absence
caused production incidents, documented inline.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Sequence

from .models import Message

__all__ = [
    "COMPACTION_SUMMARY_MARKERS",
    "NOISE_PREFIXES",
    "flatten_content",
    "is_compaction_summary",
    "is_noise",
    "parse_transcript",
    "user_messages",
]


#: Phrases that identify a *compaction summary* - the text a host injects as a
#: user-role message when it resumes a compacted session.
#:
#: INCIDENT: without this filter the summary is recorded as fresh user intent on
#: every compaction. The next session start injects it, the session is therefore
#: born large, compacts sooner, and records the summary again. Observed in
#: production as several hundred stacked copies of the same block and a session
#: that compacted every few minutes. Never record the output of compaction as
#: input to compaction.
COMPACTION_SUMMARY_MARKERS: tuple[str, ...] = (
    "this session is being continued from a previous conversation",
    "the summary below covers the earlier portion",
    "continue the conversation from where it left off",
    "this conversation was summarized",
)

#: Line prefixes emitted by tooling rather than by a person.
NOISE_PREFIXES: tuple[str, ...] = (
    "stop hook feedback",
    "<command-",
    "<system-reminder",
    "<local-command",
    "[system",
    "tool_result",
    "caveat: the messages below",
)

# There is deliberately no "substring anywhere" noise filter.
#
# An earlier version dropped any turn *containing* "<system-reminder>" or
# "hookSpecificOutput". Those are exactly the words a person types when they are
# working on hooks - the users of this package - so the filter silently deleted
# the instruction it was meant to preserve. Noise is identified by how a turn
# begins, because that is the only signal that distinguishes machine scaffolding
# from a human talking about machine scaffolding.

_WHITESPACE = re.compile(r"\s+")


def flatten_content(content: Any) -> str:
    """Reduce a message body to plain text.

    Handles the three shapes seen in the wild: a bare string, a list of typed
    blocks (only ``text`` blocks carry intent - tool calls and results are
    transport), and anything else, which flattens to empty rather than raising.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Sequence):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(parts).strip()
    return ""


def is_compaction_summary(text: str) -> bool:
    """True when the text is a compaction artefact rather than human intent."""
    lowered = (text or "").lower()
    return any(marker in lowered for marker in COMPACTION_SUMMARY_MARKERS)


def is_noise(text: str) -> bool:
    """True for tool chatter, hook feedback and host-injected scaffolding."""
    stripped = (text or "").strip()
    if not stripped:
        return True
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    # A turn that is *entirely* an XML-ish tag is scaffolding. A turn that
    # merely mentions one ("wrap it in <system-reminder>") is real intent, so
    # only the leading character is tested.
    return stripped.startswith("<")


def parse_transcript(
    lines: Iterable[str],
    *,
    min_length: int = 8,
    max_chars: int = 400,
    roles: Sequence[str] = ("user", "assistant"),
    drop_summaries: bool = True,
) -> list[Message]:
    """Parse JSONL transcript lines into clean messages, oldest first.

    Malformed lines are skipped rather than fatal: a transcript is an append-only
    log written by another process and may be torn mid-write at exactly the
    moment a pre-compaction hook reads it. Losing one line is acceptable;
    losing the checkpoint because of one line is not.

    ``min_length`` drops acknowledgements ("ok", "yes") that add length without
    adding intent. ``max_chars`` bounds each message so a single pasted file
    cannot dominate the distillation window.
    """
    out: list[Message] = []
    for line in lines:
        if not line or not line.strip():
            continue
        try:
            envelope = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(envelope, dict):
            continue
        message = envelope.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in roles:
            continue
        text = flatten_content(message.get("content"))
        if len(text) < min_length:
            continue
        if is_noise(text):
            continue
        if drop_summaries and is_compaction_summary(text):
            continue
        collapsed = _WHITESPACE.sub(" ", text).strip()
        out.append(Message(role=role, text=collapsed[:max_chars]))
    return out


def user_messages(messages: Sequence[Message]) -> list[Message]:
    """Only the human turns - the authoritative statement of intent."""
    return [m for m in messages if m.is_user]
