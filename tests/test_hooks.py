"""The hook adapter. Thin by design, and forbidden from failing loudly."""

from __future__ import annotations

import io
import json
from datetime import timedelta
from pathlib import Path

from durable_context import Checkpoint, Config, DurableContext, HeuristicDistiller, MemoryStore
from durable_context.hooks import pre_compact_hook, read_hook_input, session_start_hook

from conftest import FROZEN, transcript


def _context():
    return DurableContext(
        config=Config(home=Path("/virtual"), stale_after=timedelta(hours=6)),
        store=MemoryStore(),
        distiller=HeuristicDistiller(),
        clock=lambda: FROZEN,
    )


def test_hook_input_tolerates_absence_and_garbage():
    """Hosts differ, versions differ, and a hand-run hook has no stdin at all.

    None of those is worth an exception in a process the host spawned to protect
    the user's context.
    """
    assert read_hook_input(None) == {}
    assert read_hook_input(io.StringIO("")) == {}
    assert read_hook_input(io.StringIO("not json at all")) == {}
    assert read_hook_input(io.StringIO('["a list, not an object"]')) == {}
    assert read_hook_input(io.StringIO('{"trigger": "auto"}')) == {"trigger": "auto"}


def test_a_missing_transcript_makes_the_hook_decline_quietly():
    """A hook fires on the host's schedule, not on ours.

    The transcript path can be absent, rotated or on a filesystem that is not
    mounted yet. Declining without writing is correct; writing a checkpoint
    derived from nothing would overwrite good state with an empty record.
    """
    context = _context()
    envelope = pre_compact_hook(
        {"transcript_path": "/definitely/not/here.jsonl", "trigger": "auto"}, context
    )
    assert envelope == {}
    assert not context.store.exists(context.config.decision_path)


def test_the_pre_compact_hook_stays_silent_on_success():
    """Whatever this hook printed would land in the context it is protecting.

    Its job is to write to disk. An empty envelope is the correct output, and a
    chatty one would be actively counterproductive.
    """
    context = _context()
    lines = transcript(("user", "Let's go with the streaming parser."))
    envelope = pre_compact_hook(
        {"transcript_path": __file__, "trigger": "auto"},
        context,
        read_lines=lambda path: lines,
    )
    assert envelope == {}
    stored, _ = context.load_decision()
    assert "streaming parser" in stored.decision


def test_the_pre_compact_hook_writes_when_the_transcript_is_real(tmp_path):
    """The only real dependency is a readable transcript file."""
    transcript_file = tmp_path / "transcript.jsonl"
    transcript_file.write_text(
        "".join(transcript(("user", "Let's go with Postgres instead of MongoDB."))),
        encoding="utf-8",
    )
    context = _context()
    assert pre_compact_hook(
        {"transcript_path": str(transcript_file), "trigger": "auto"}, context
    ) == {}
    stored, _ = context.load_decision()
    assert stored.discarded == "MongoDB"


def test_a_broken_context_never_propagates_out_of_a_hook():
    """Losing durable context is acceptable. Breaking the host is not.

    This is the guarantee that lets a user install the hook without auditing it:
    the worst case is that it does nothing.
    """

    class Broken(DurableContext):
        def pre_compact(self, *args, **kwargs):
            raise RuntimeError("disk on fire")

        def session_start(self, **kwargs):
            raise RuntimeError("disk still on fire")

    broken = Broken(
        config=Config(home=Path("/virtual")),
        store=MemoryStore(),
        distiller=HeuristicDistiller(),
        clock=lambda: FROZEN,
    )
    assert pre_compact_hook({"transcript_path": __file__}, broken) == {}
    assert session_start_hook(broken) == {}


def test_the_session_start_hook_says_nothing_when_there_is_nothing_to_say():
    """An empty additionalContext still costs tokens and trains the reader to skip."""
    assert session_start_hook(_context()) == {}


def test_the_session_start_hook_emits_the_host_envelope():
    """The envelope shape is the host's contract; the content is ours."""
    context = _context()
    context.save_checkpoint(Checkpoint(sections={"NOW": "porting the uploader"}))
    envelope = session_start_hook(context)
    specific = envelope["hookSpecificOutput"]
    assert specific["hookEventName"] == "SessionStart"
    assert "porting the uploader" in specific["additionalContext"]
    json.dumps(envelope)  # must be serialisable for the host to read it
