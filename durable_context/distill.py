"""Distillers: turn a slice of conversation into one :class:`Decision`.

Three implementations, all interchangeable:

* :class:`HeuristicDistiller` - pure, deterministic, no model, no network. This
  is the default on purpose. The distiller runs inside a pre-compaction hook,
  where a hang is worse than a mediocre answer.
* :class:`CommandDistiller` - shells out to any command that reads a prompt on
  stdin and prints JSON. Vendor-neutral by construction: you supply the argv.
* :class:`FallbackDistiller` - tries distillers in order and keeps the first
  usable answer, so an LLM can be an *upgrade* rather than a dependency.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence, runtime_checkable

from .models import Decision, Message
from .transcript import user_messages

__all__ = [
    "Distiller",
    "HeuristicDistiller",
    "CommandDistiller",
    "FallbackDistiller",
    "DEFAULT_PROMPT",
    "extract_json",
    "build_prompt",
]


@runtime_checkable
class Distiller(Protocol):
    """Anything that can propose the current decision from conversation."""

    name: str

    def distill(self, messages: Sequence[Message]) -> Decision | None:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------
# Heuristic (default)
# --------------------------------------------------------------------------

DECISION_CUES: tuple[str, ...] = (
    "let's go with",
    "lets go with",
    "we're going with",
    "we are going with",
    "going with",
    "we'll use",
    "we will use",
    "decided on",
    "decided to",
    "decision:",
    "switching to",
    "switch to",
    "pivot to",
    "pivoting to",
    "locked in",
    "lock in",
    "final answer:",
    "final:",
    "go with",
    "stick with",
    "use ",
)

DISCARD_CUES: tuple[str, ...] = (
    "instead of",
    "rather than",
    "no longer",
    "dropping",
    "drop the",
    "abandon",
    "scrap the",
    "ditch the",
    "forget the",
    "not the",
    " not ",
)

WHY_CUES: tuple[str, ...] = (
    "because",
    "the reason is",
    "reason:",
    "so that",
    "in order to",
    "since ",
)

OPEN_CUES: tuple[str, ...] = (
    "still need",
    "still open",
    "still unclear",
    "pending",
    "waiting on",
    "waiting for",
    "blocked on",
    "blocked by",
    "open question",
    "unresolved",
    "tbd",
    "not yet",
)

NEXT_CUES: tuple[str, ...] = (
    "next step",
    "next:",
    "after that",
    "then do",
    "proceed to",
    "start with",
    "follow up with",
    "now do",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.;!?])\s+")
_CLAUSE_STOP = re.compile(
    r"[.;!?,]|\s+because\b|\s+since\b|\s+so that\b|\s+instead\b|\s+rather than\b|\s+but\b|\s+-\s+|\s+—\s+",
    re.IGNORECASE,
)
_CLAUSE_START = re.compile(r"[.;!?,]")

#: A prohibition is not a decision. "Do not **use** X" contains the ``use`` cue,
#: and without this guard the thing the user forbade was recorded as the locked
#: TARGET - the exact inverse of what was said. Matched against the clause
#: leading up to a cue, allowing a few words in between ("not sure we should
#: use ..."), and never across a clause boundary, so "we are not using Mongo,
#: so let's go with Postgres" still reads as a decision.
_NEGATION_BEFORE = re.compile(
    r"\b(?:not|never|no|nor|neither|avoid|avoiding|stop|stopped|stopping|without|"
    r"dont|doesnt|didnt|wont|cant|shouldnt|wouldnt|isnt|arent|"
    r"don't|doesn't|didn't|won't|can't|shouldn't|wouldn't|isn't|aren't|cannot)\b"
    r"(?:\s+\w+){0,3}\s*$"
)

#: "I am not sure we should use Redis" is hedging, not a rejection. Without this
#: the ``not`` discard cue recorded "sure we should use Redis here" as the path
#: ruled out.
_HEDGE_AFTER = re.compile(
    r"^\s*(?:sure|certain|convinced|clear|sold|keen|entirely|quite|really|totally|"
    r"necessarily|always|yet|just)\b"
)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _is_negated(lowered: str, index: int) -> bool:
    """Is this cue occurrence inside a prohibition rather than a decision?"""
    clause = lowered[:index]
    boundary = max((m.end() for m in _CLAUSE_START.finditer(clause)), default=0)
    return bool(_NEGATION_BEFORE.search(clause[boundary:]))


def _is_hedged(lowered: str, index: int) -> bool:
    """Does the clause after this cue hedge ("not *sure*") instead of reject?"""
    return bool(_HEDGE_AFTER.match(lowered[index:]))


def _find_cue(
    text: str,
    cues: Sequence[str],
    *,
    accept: "Callable[[str, str, int], bool] | None" = None,
) -> tuple[str, int] | None:
    """First *acceptable* cue present, in caller-supplied priority order.

    ``accept`` filters occurrences rather than cues, so a cue that appears twice
    - once negated, once not - is still found at the position that means it.
    """
    lowered = text.lower()
    for cue in cues:
        start = 0
        while True:
            index = lowered.find(cue, start)
            if index < 0:
                break
            if accept is None or accept(lowered, cue, index):
                return cue, index
            start = index + 1
    return None


def _accept_decision_cue(lowered: str, cue: str, index: int) -> bool:
    return not _is_negated(lowered, index)


def _accept_discard_cue(lowered: str, cue: str, index: int) -> bool:
    return not _is_hedged(lowered, index + len(cue))


def _clause_after(text: str, start: int, limit: int) -> str:
    """The clause following an offset, cut at the first boundary token."""
    tail = text[start:].strip()
    match = _CLAUSE_STOP.search(tail)
    if match and match.start() > 0:
        tail = tail[: match.start()]
    return " ".join(tail.split())[:limit].strip(" -—:")


def _sentence_with(text: str, cues: Sequence[str], limit: int) -> str:
    for sentence in _sentences(text):
        if _find_cue(sentence, cues):
            return " ".join(sentence.split())[:limit]
    return ""


@dataclass(frozen=True)
class HeuristicDistiller:
    """Deterministic extraction from the human turns. No model, no clock.

    It reads newest-first and keeps the first hit per field, because in a long
    session the most recent statement of a decision is the one in force.

    It is not trying to be an LLM. It is trying to guarantee that *something
    structured and honest* is on disk before anything slower is attempted.
    """

    decision_cues: Sequence[str] = DECISION_CUES
    discard_cues: Sequence[str] = DISCARD_CUES
    why_cues: Sequence[str] = WHY_CUES
    open_cues: Sequence[str] = OPEN_CUES
    next_cues: Sequence[str] = NEXT_CUES
    max_field_chars: int = 240
    name: str = "heuristic"

    def distill(self, messages: Sequence[Message]) -> Decision | None:
        users = user_messages(messages)
        if not users:
            return None
        newest_first = list(reversed(users))

        decision_text = ""
        target = ""
        discarded = ""
        for message in newest_first:
            hit = _find_cue(message.text, self.decision_cues, accept=_accept_decision_cue)
            if not hit:
                continue
            cue, index = hit
            decision_text = " ".join(message.text.split())[: self.max_field_chars]
            target = _clause_after(message.text, index + len(cue), self.max_field_chars)
            discarded = self._discarded_from(message.text)
            break

        if not decision_text:
            # No explicit decision language anywhere: fall back to the latest
            # human turn. A weak record of intent still beats an empty one.
            decision_text = " ".join(newest_first[0].text.split())[: self.max_field_chars]

        if not discarded:
            for message in newest_first:
                discarded = self._discarded_from(message.text)
                if discarded:
                    break

        why = ""
        for message in newest_first:
            hit = _find_cue(message.text, self.why_cues)
            if hit:
                cue, index = hit
                why = _clause_after(message.text, index + len(cue), self.max_field_chars)
                if why:
                    break

        open_thread = ""
        for message in newest_first:
            open_thread = _sentence_with(message.text, self.open_cues, self.max_field_chars)
            if open_thread:
                break

        next_step = ""
        for message in newest_first:
            next_step = _sentence_with(message.text, self.next_cues, self.max_field_chars)
            if next_step:
                break

        decision = Decision(
            decision=decision_text,
            target=target,
            why=why,
            discarded=discarded,
            open_thread=open_thread,
            next_step=next_step,
            source=self.name,
        )
        return None if decision.is_empty() else decision

    def _discarded_from(self, text: str) -> str:
        hit = _find_cue(text, self.discard_cues, accept=_accept_discard_cue)
        if not hit:
            return ""
        cue, index = hit
        return _clause_after(text, index + len(cue), self.max_field_chars)


# --------------------------------------------------------------------------
# Command-backed (optional upgrade)
# --------------------------------------------------------------------------

DEFAULT_PROMPT = """\
Read the conversation below and report ONLY the decision state that is in force
right now, so it survives a context compaction.

If two paths are discussed, say which one is locked in and which was rejected.
Do not invent anything that is not in the conversation; leave a field empty
instead.

CONVERSATION:
{conversation}

Reply with one JSON object and nothing else:
{{"decision": "what is locked in now, one unambiguous sentence",
 "target": "the exact object of the decision",
 "why": "the real reason",
 "discarded": "the path explicitly rejected, or empty",
 "open_thread": "what is still open",
 "next_step": "the concrete next action"}}
"""

_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*|```")


def build_prompt(messages: Sequence[Message], *, template: str = DEFAULT_PROMPT, max_chars: int = 9000) -> str:
    """Render the conversation into a prompt, keeping the most recent text.

    Truncation takes the *tail*: in a session the recent turns carry the live
    decision, and an opening turn that has already been superseded is the least
    valuable thing in the window.
    """
    conversation = "\n".join(message.render() for message in messages)
    if len(conversation) > max_chars:
        conversation = conversation[-max_chars:]
    return template.format(conversation=conversation)


def extract_json(text: str) -> dict | None:
    """Pull the decision object out of noisy model output.

    Models wrap JSON in prose, in code fences, and behind a preamble object of
    their own. This scans every ``{`` with a streaming decoder and accepts the
    first object that actually carries decision fields, so a leading
    ``{"thinking": "..."}`` does not win.
    """
    if not text:
        return None
    cleaned = _FENCE.sub("", text)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[index:])
        except ValueError:
            continue
        if isinstance(candidate, dict) and any(key in candidate for key in Decision.CONTENT_FIELDS):
            return candidate
    return None


def subprocess_runner(command: Sequence[str], prompt: str, timeout: float) -> str:
    """Default runner: prompt on stdin, text on stdout."""
    completed = subprocess.run(
        list(command),
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return completed.stdout or ""


@dataclass(frozen=True)
class CommandDistiller:
    """Distil by shelling out to a user-supplied command.

    The command contract is deliberately minimal - read a prompt on stdin,
    print JSON on stdout - so any model runner, local or hosted, can be dropped
    in without this package knowing a vendor name.

    Every failure mode (missing binary, non-zero exit, timeout, prose instead of
    JSON) returns ``None``. A distiller that raises inside a pre-compaction hook
    would cost the user the very context it exists to protect.
    """

    command: Sequence[str]
    timeout: float = 45.0
    prompt_template: str = DEFAULT_PROMPT
    max_prompt_chars: int = 9000
    runner: Callable[[Sequence[str], str, float], str] = subprocess_runner
    name: str = "command"

    def distill(self, messages: Sequence[Message]) -> Decision | None:
        if not self.command or not messages:
            return None
        prompt = build_prompt(messages, template=self.prompt_template, max_chars=self.max_prompt_chars)
        try:
            output = self.runner(self.command, prompt, self.timeout)
        except Exception:
            return None
        payload = extract_json(output or "")
        if not payload:
            return None
        decision = Decision.from_dict(payload).with_source(self.name)
        return None if decision.is_empty() else decision


@dataclass(frozen=True)
class FallbackDistiller:
    """Try distillers in order; keep the first usable answer.

    This is how an optional LLM stays optional. Put the command distiller first
    and the heuristic second: when the model answers you get its quality, and
    when it is slow, broken or absent you still get a record.
    """

    distillers: Sequence[Distiller] = field(default_factory=tuple)
    name: str = "fallback"

    def distill(self, messages: Sequence[Message]) -> Decision | None:
        for distiller in self.distillers:
            try:
                result = distiller.distill(messages)
            except Exception:
                continue
            if result and not result.is_empty():
                return result
        return None
