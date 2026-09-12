"""Agent-host hook adapter - the thin edge, and nothing else.

A host runs a hook as a subprocess: JSON on stdin, JSON on stdout. The two
events that matter are "I am about to compact" and "a session is starting".

Everything here is glue. The rule that keeps it honest: a hook must not fail
loudly. If this package cannot do its job, the user should lose durable context
and nothing more - never their compaction, never their session start. So the
public functions swallow their own errors and return an empty envelope.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any, Mapping

from .session import DurableContext

__all__ = [
    "read_hook_input",
    "pre_compact_hook",
    "session_start_hook",
    "SESSION_START_EVENT",
]

SESSION_START_EVENT = "SessionStart"


def read_hook_input(stream: IO[str] | None) -> dict:
    """Parse a hook payload from stdin, tolerating absence and garbage.

    Hosts differ, versions differ, and a hook invoked by hand has no stdin at
    all. None of those is worth an exception in a hook.
    """
    if stream is None:
        return {}
    try:
        raw = stream.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _transcript_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.readlines()
    except OSError:
        return []


def pre_compact_hook(
    payload: Mapping[str, Any],
    context: DurableContext,
    *,
    read_lines=_transcript_lines,
) -> dict:
    """Handle a pre-compaction event. Always returns an empty envelope.

    The return value is empty on purpose: this hook exists to write state, not
    to influence the host. Anything it printed would be injected into the
    context it is trying to protect.
    """
    try:
        transcript = str(payload.get("transcript_path") or "")
        trigger = str(payload.get("trigger") or "auto")
        if not transcript or not Path(transcript).exists():
            return {}
        context.pre_compact(read_lines(transcript), trigger=trigger)
    except Exception:
        pass
    return {}


def session_start_hook(context: DurableContext) -> dict:
    """Handle a session-start event, returning the context to inject.

    Returns ``{}`` when there is nothing to say. An empty ``additionalContext``
    would still cost the session a header's worth of tokens and teach the reader
    that these blocks are usually noise.
    """
    try:
        payload = context.session_start()
    except Exception:
        return {}
    if not payload:
        return {}
    return {
        "hookSpecificOutput": {
            "hookEventName": SESSION_START_EVENT,
            "additionalContext": payload.text,
        }
    }
