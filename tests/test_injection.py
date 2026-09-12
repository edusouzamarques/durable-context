"""What the next session is actually told, and in what order."""

from __future__ import annotations

from datetime import timedelta

from durable_context import Checkpoint, Decision, Staleness, build_injection

DECISION = Decision(
    decision="store events in the column store",
    target="ClickHouse",
    why="event volume is 40x reads",
    discarded="sharded Postgres",
)


def test_the_active_decision_comes_first():
    """Position is the cheapest way to make something read.

    The decision block is the smallest and the only one that prevents the
    specific failure this package exists for. Burying it under a checkpoint and
    a history tail means it competes for attention with background.
    """
    payload = build_injection(
        decision=DECISION,
        checkpoint=Checkpoint(sections={"NOW": "porting the uploader"}),
        log_tail=["- earlier entry"],
    )
    text = payload.text
    assert text.index("ACTIVE DECISION") < text.index("SESSION CHECKPOINT") < text.index("DECISION LOG")


def test_the_rejected_path_is_stated_in_the_injected_text():
    """An agent cannot honour a constraint it was never shown.

    Recording the rejection on disk is only half the job; it has to reach the
    context window with a clear instruction not to re-open it.
    """
    payload = build_injection(decision=DECISION)
    assert "REJECTED: sharded Postgres" in payload.text
    assert "do not re-open" in payload.text.lower()


def test_a_tight_budget_sacrifices_history_not_the_decision():
    """Truncation order is a correctness property, not a formatting detail.

    Under pressure the append-only history is background and the live decision
    is the payload. Dropping them in the other order would deliver a session
    that knows how the project got here but not where it is going.
    """
    payload = build_injection(
        decision=DECISION,
        checkpoint=Checkpoint(sections={"NOW": "x" * 3000}),
        log_tail=["- history " + "y" * 3000],
        max_chars=700,
    )
    assert "ACTIVE DECISION" in payload.text
    assert "DECISION LOG" not in payload.text
    assert payload.truncated
    assert len(payload.text) <= 700


def test_an_oversized_decision_is_truncated_rather_than_dropped():
    """A partial decision still names the locked target.

    Returning nothing because the top block does not fit would turn a budget
    problem into total context loss.
    """
    payload = build_injection(
        decision=Decision(decision="z" * 5000, target="T"), max_chars=200
    )
    assert payload.text
    assert len(payload.text) <= 200
    assert payload.truncated


def test_a_stale_checkpoint_is_labelled_in_the_text():
    """Silently injecting old state is worse than injecting none.

    The reader assumes injected context describes the live world. If it does
    not, the warning has to travel with it.
    """
    payload = build_injection(
        checkpoint=Checkpoint(sections={"NOW": "porting"}),
        staleness=Staleness(True, timedelta(hours=9), "last compaction was 9h00m ago"),
    )
    assert "WARNING" in payload.text
    assert "9h" in payload.text


def test_an_undated_checkpoint_is_marked_unverified():
    """Unknown age is its own category and must not read as fresh."""
    payload = build_injection(
        checkpoint=Checkpoint(sections={"NOW": "porting"}),
        staleness=Staleness(False, None, "no compaction timestamp recorded"),
    )
    assert "age is unknown" in payload.text


def test_nothing_recorded_means_nothing_injected():
    """A fresh install must cost the session zero tokens.

    Emitting an empty header block would spend context to say nothing, and would
    teach the reader that these blocks are usually noise - which is exactly how
    a real warning later gets skipped.
    """
    payload = build_injection(decision=Decision(), checkpoint=Checkpoint(), log_tail=[])
    assert not payload
    assert payload.text == ""


def test_blank_log_lines_do_not_create_an_empty_history_block():
    """Whitespace in the log must not manufacture a header with no content."""
    payload = build_injection(decision=DECISION, log_tail=["", "   ", "\n"])
    assert "DECISION LOG" not in payload.text


def test_the_decision_block_reports_its_own_age():
    """The reader has to be able to weigh how old the locked target is."""
    payload = build_injection(decision=DECISION, decision_age=timedelta(hours=3))
    assert "3h00m ago" in payload.text


def test_a_future_stamped_checkpoint_is_labelled_as_clock_skew():
    """Neither stale nor unknown, and silently injected as if it were fresh.

    A wrong system date - or state synced from a host whose clock is ahead -
    produces a negative age, which short-circuits the staleness comparison
    permanently. The verdict already said "clock skew"; the reader was never
    told, so the checkpoint arrived with no warning and no age, and could never
    be flagged again.
    """
    verdict = Staleness(False, timedelta(hours=-2), "timestamp is in the future (clock skew)")
    payload = build_injection(
        checkpoint=Checkpoint(sections={"NOW": "porting the uploader"}),
        staleness=verdict,
    )
    assert "clock skew" in payload.text
    assert "2h" in payload.text
    assert "unverified" in payload.text
