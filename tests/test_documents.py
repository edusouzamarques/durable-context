"""Documents are written for humans and read back by machine. Both must hold."""

from __future__ import annotations

from datetime import datetime, timedelta

from durable_context import (
    Checkpoint,
    Decision,
    parse_checkpoint,
    parse_decision,
    render_checkpoint,
    render_decision,
    staleness,
)

NOW = datetime(2026, 3, 14, 9, 30)


def test_checkpoint_round_trips_without_loss():
    """Anything this package writes, it must be able to read back.

    These files get hand-edited at 2am when something is on fire. If a human fix
    is unparseable, or if the parser quietly drops a section it did not expect,
    the durable layer decays into a text file nobody trusts.
    """
    original = Checkpoint(
        sections={
            "NOW": "porting the uploader",
            "GOTCHAS": "the CDN caches 404s for an hour",
            "TEAM VOCAB": "an arbitrary project-specific section",
        },
        last_compact=NOW,
        trigger="auto",
    )
    restored = parse_checkpoint(render_checkpoint(original))
    assert restored.sections == original.sections
    assert list(restored.sections) == list(original.sections)
    assert restored.last_compact == NOW
    assert restored.trigger == "auto"


def test_decision_round_trips_with_the_rejected_path():
    """The rejected path is the field with the shortest half-life.

    It is the first thing a summariser drops and the one whose absence causes
    the relitigation this package exists to stop, so its survival across a
    render/parse cycle is worth a dedicated test.
    """
    original = Decision(
        decision="store events in the column store",
        target="ClickHouse",
        why="event volume is 40x reads",
        discarded="sharded Postgres",
        open_thread="retention policy undecided",
        next_step="write the ingestion adapter",
        source="heuristic",
    )
    restored, distilled_at = parse_decision(render_decision(original, distilled_at=NOW))
    assert restored == original
    assert distilled_at == NOW


def test_an_empty_decision_renders_a_readable_stub():
    """Rendering must never fail on the empty case.

    A fresh install has nothing recorded, and a crash there would break the very
    first hook invocation a user ever sees.
    """
    text = render_decision(Decision())
    assert "Active decision" in text
    assert parse_decision(text)[0].is_empty()


def test_an_undated_checkpoint_is_reported_as_unknown_not_fresh():
    """Unknown age must not be silently treated as "recent".

    Injected state carries authority. Claiming freshness that was never measured
    is how a reader ends up confidently asserting something that stopped being
    true hours ago.
    """
    verdict = staleness(Checkpoint(sections={"NOW": "x"}), now=NOW, stale_after=timedelta(hours=6))
    assert verdict.is_unknown
    assert not verdict.is_stale
    assert "no compaction timestamp" in verdict.reason


def test_an_aged_checkpoint_is_flagged_with_its_age():
    """The reader needs the number, not just a boolean.

    "Stale" alone invites the reader to ignore it; "9 hours old" tells them what
    to go and verify.
    """
    verdict = staleness(
        Checkpoint(sections={"NOW": "x"}, last_compact=NOW - timedelta(hours=9)),
        now=NOW,
        stale_after=timedelta(hours=6),
    )
    assert verdict.is_stale
    assert "9h" in verdict.reason


def test_a_future_timestamp_is_called_out_as_clock_skew():
    """A clock that ran backwards must not produce eternal freshness.

    Container restarts, NTP corrections and cross-machine state all produce
    future stamps. Treating them as valid would disable staleness detection
    permanently and silently.
    """
    verdict = staleness(
        Checkpoint(sections={"NOW": "x"}, last_compact=NOW + timedelta(hours=2)),
        now=NOW,
        stale_after=timedelta(hours=6),
    )
    assert not verdict.is_stale
    assert "clock skew" in verdict.reason


def test_a_corrupt_timestamp_does_not_break_the_parse():
    """One bad header byte must not cost the whole checkpoint.

    Degrading to "age unknown" keeps every section readable; raising would throw
    away the content to protect the metadata.
    """
    text = "<!-- durable-context: last_compact=not-a-date -->\n# Session checkpoint\n\n## NOW\nporting\n"
    restored = parse_checkpoint(text)
    assert restored.last_compact is None
    assert restored.sections["NOW"] == "porting"


def test_a_body_containing_a_markdown_heading_round_trips():
    """Section bodies are free text, and free text contains Markdown.

    A body line beginning "## " used to be read back as a *new* top-level
    section: half of GOTCHAS silently became a phantom heading that ``--clear``
    could not reach and that every later merge carried forward.
    """
    original = Checkpoint(
        sections={
            "GOTCHAS": "the CDN caches 404s\n## deploy notes\nrun the migration first",
            "NOW": "porting the uploader",
        }
    )
    restored = parse_checkpoint(render_checkpoint(original))
    assert list(restored.sections) == ["GOTCHAS", "NOW"]
    assert restored.sections == original.sections


def test_an_escaped_heading_in_a_body_round_trips_too():
    """The escape is escaped, so the round trip stays lossless at any depth."""
    original = Checkpoint(sections={"GOTCHAS": "first line\n" + r"\## already escaped, keep it"})
    assert parse_checkpoint(render_checkpoint(original)).sections == original.sections


def test_a_trigger_with_whitespace_cannot_break_the_header():
    """The trigger comes from the host, unvalidated.

    One containing a newline broke the header regex outright, so a checkpoint
    written seconds earlier read back undated and was injected as "age unknown,
    treat as unverified" - staleness detection disabled by a host's string.
    """
    for trigger in ("auto\nbogus", "auto compact", "auto --> x"):
        restored = parse_checkpoint(
            render_checkpoint(Checkpoint(sections={"NOW": "x"}, last_compact=NOW, trigger=trigger))
        )
        assert restored.last_compact == NOW, trigger
        assert restored.trigger, trigger
        assert "\n" not in restored.trigger


def test_a_multi_line_decision_field_is_folded_not_truncated():
    """The decision document is one line per field.

    Written whole, a value carrying a newline parsed back truncated at that
    newline - dropping the reason, which is the field the package exists for -
    while every caller was told the write succeeded.
    """
    original = Decision(decision="use ClickHouse\nbecause reads are 40x writes")
    restored, _ = parse_decision(render_decision(original))
    assert restored.decision == "use ClickHouse because reads are 40x writes"
