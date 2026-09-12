"""The append-only layer, and the rules for choosing a safe tail of it."""

from __future__ import annotations

from datetime import datetime

from durable_context import Decision, render_decision_entry, render_entry, select_tail

NOW = datetime(2026, 3, 14, 9, 30)


def test_entries_are_dated_and_greppable():
    """The cold log is read by a person with grep, months later.

    A dated heading per block is what makes "when did we decide this" answerable
    without a database.
    """
    entry = render_entry(timestamp=NOW, kind="intent", lines=["ship the parser"], trigger="auto")
    assert "2026-03-14T09:30 intent trigger=auto" in entry
    assert "- ship the parser" in entry


def test_a_decision_entry_carries_the_rejection_into_history():
    """History has to record the road not taken, not just the road taken.

    Six weeks later the useful question is "why aren't we on the other thing",
    and only the rejected field answers it.
    """
    entry = render_decision_entry(
        Decision(decision="use ClickHouse", target="ClickHouse", discarded="sharded Postgres"),
        timestamp=NOW,
    )
    assert "REJECTED: sharded Postgres" in entry


def test_an_empty_decision_writes_no_entry():
    """Append-only means every write is permanent, so empty writes are litter."""
    assert render_decision_entry(Decision(), timestamp=NOW) == ""


def test_filtering_happens_before_the_tail_is_taken():
    """Otherwise a run of junk silently evicts the real history.

    Taking the last N lines and *then* filtering can return nothing at all, even
    when the log is full of useful entries a few lines further back.
    """
    lines = ["This session is being continued from a previous conversation."] * 5
    lines += ["- chose the column store", "- rejected sharded Postgres"]
    tail = select_tail(lines, max_lines=3)
    assert tail == ["- chose the column store", "- rejected sharded Postgres"]


def test_a_repeated_block_cannot_fill_the_injection_window():
    """Logs accumulate duplicates when an upstream loop goes unnoticed.

    Without deduplication a single sentence repeated hundreds of times consumes
    the entire tail budget, and the recent, useful entries never get injected.
    """
    lines = ["- the same stale note"] * 200 + ["- today's actual decision"]
    tail = select_tail(lines, max_lines=5)
    assert tail == ["- the same stale note", "- today's actual decision"]


def test_deduplication_keeps_the_most_recent_occurrence():
    """Order must stay chronological so the tail still reads as history."""
    tail = select_tail(["- a", "- b", "- a", "- c"], max_lines=10)
    assert tail == ["- b", "- a", "- c"]


def test_a_zero_budget_returns_nothing_rather_than_everything():
    """An off-by-one here would inject the entire log into a session."""
    assert select_tail(["- a", "- b"], max_lines=0) == []
