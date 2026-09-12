"""End-to-end behaviour of the reactive mechanism, in memory, at a fixed instant."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from durable_context import Checkpoint, Decision, DurableContext, HeuristicDistiller

from conftest import FROZEN, transcript


@dataclass
class NullDistiller:
    """A distiller that never produces anything - the model-is-down case."""

    name: str = "null"

    def distill(self, messages):
        return None


@dataclass
class ExplodingDistiller:
    """A distiller that raises - the model-is-broken case."""

    name: str = "exploding"

    def distill(self, messages):
        raise RuntimeError("model runner segfaulted")


def test_raw_intent_survives_even_when_distillation_produces_nothing():
    """Intent is written to the cold log *before* distillation is attempted.

    Distillation is the part that can fail: models time out, binaries go
    missing, heuristics find no cue. Ordering the write first guarantees that
    the user's own words survive the compaction regardless.
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=NullDistiller(), clock=lambda: FROZEN
    )
    result = context.pre_compact(
        transcript(("user", "Rewrite the uploader to stream instead of buffering.")),
        trigger="auto",
    )
    log = context.store.read(context.config.log_path)
    assert "stream instead of buffering" in log
    assert result.intent


def test_a_distiller_that_raises_cannot_break_a_compaction():
    """The host is mid-compaction; an exception here costs the whole session.

    Failing soft is not politeness, it is the difference between "we lost some
    nuance" and "we lost the context we were trying to save".
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=ExplodingDistiller(), clock=lambda: FROZEN
    )
    result = context.pre_compact(transcript(("user", "Pin the dependency at 2.4 for now.")))
    assert result.messages_seen == 1
    assert context.load_checkpoint().last_compact == FROZEN


def test_a_pivot_across_two_compactions_is_carried_into_the_next_session():
    """The whole point, exercised end to end.

    Compaction one records MongoDB. Compaction two records Postgres. The session
    that starts afterwards must be told both - that Postgres is locked in *and*
    that MongoDB was rejected - or it will helpfully re-propose MongoDB and the
    user will have the same argument twice.
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=HeuristicDistiller(), clock=lambda: FROZEN
    )
    context.pre_compact(transcript(("user", "Let's go with MongoDB for the event store.")))
    context.pre_compact(transcript(("user", "Let's go with Postgres for the event store.")))

    injected = context.session_start().text
    assert "Postgres" in injected
    assert "REJECTED: MongoDB" in injected


def test_checkpoint_sections_survive_the_compaction_stamp():
    """Stamping the time must not be a covert overwrite.

    The pre-compaction pass touches the checkpoint only to record when it ran.
    If that write dropped the sections, every compaction would quietly erase the
    hot layer it exists to maintain.
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=NullDistiller(), clock=lambda: FROZEN
    )
    context.save_checkpoint(Checkpoint(sections={"GOTCHAS": "the CDN caches 404s"}))
    context.pre_compact(transcript(("user", "Carry on with the uploader work.")))
    assert context.load_checkpoint().sections["GOTCHAS"] == "the CDN caches 404s"


def test_a_fresh_install_injects_nothing():
    """First run must be silent, not a header saying there is no state."""
    context = DurableContext(
        config=_config(), store=_store(), distiller=HeuristicDistiller(), clock=lambda: FROZEN
    )
    assert not context.session_start()


def test_an_old_checkpoint_is_injected_with_its_warning():
    """Age is computed at read time from the recorded stamp.

    A session that starts long after the last compaction is the exact case where
    injected state misleads, so the warning has to be produced by the reader, not
    baked in by the writer.
    """
    early = datetime(2026, 3, 14, 0, 0)
    context = DurableContext(
        config=_config(), store=_store(), distiller=NullDistiller(), clock=lambda: early
    )
    context.save_checkpoint(Checkpoint(sections={"NOW": "porting"}, last_compact=early))

    later = DurableContext(
        config=context.config,
        store=context.store,
        distiller=NullDistiller(),
        clock=lambda: early + timedelta(hours=12),
    )
    assert "WARNING" in later.session_start().text


def test_a_manual_decision_merges_instead_of_replacing():
    """Correcting one field by hand must not wipe the rest.

    Hand-fixing the durable layer is expected. If setting NEXT erased WHY, the
    first correction a user makes would destroy the record.
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=HeuristicDistiller(), clock=lambda: FROZEN
    )
    context.save_decision(
        Decision(decision="use ClickHouse", target="ClickHouse", why="40x read volume")
    )
    context.save_decision(Decision(target="ClickHouse", next_step="write the adapter"))
    stored, _ = context.load_decision()
    assert stored.why == "40x read volume"
    assert stored.next_step == "write the adapter"


def test_every_compaction_appends_and_never_rewrites():
    """The cold log is the record of how the project got here.

    A log you are willing to rewrite is not a record. This pins that repeated
    compactions grow the file rather than replacing its contents.
    """
    context = DurableContext(
        config=_config(), store=_store(), distiller=HeuristicDistiller(), clock=lambda: FROZEN
    )
    context.pre_compact(transcript(("user", "First we ship the parser rewrite.")))
    first = context.store.read(context.config.log_path)
    context.pre_compact(transcript(("user", "Then we ship the uploader rewrite.")))
    second = context.store.read(context.config.log_path)
    assert second.startswith(first)
    assert "parser rewrite" in second and "uploader rewrite" in second


def test_status_reports_what_is_preserved():
    """Operators debugging a hook need one command that shows the whole state."""
    context = DurableContext(
        config=_config(), store=_store(), distiller=HeuristicDistiller(), clock=lambda: FROZEN
    )
    context.pre_compact(transcript(("user", "Let's go with the streaming parser.")), trigger="auto")
    snapshot = context.status()
    assert snapshot["trigger"] == "auto"
    assert snapshot["stale"] is False
    assert "streaming parser" in snapshot["decision"]["decision"]


# -- helpers ---------------------------------------------------------------


def _config():
    from pathlib import Path

    from durable_context import Config

    return Config(home=Path("/virtual/state"), stale_after=timedelta(hours=6))


def _store():
    from durable_context import MemoryStore

    return MemoryStore()


def test_the_log_tail_is_read_from_the_tail_not_the_whole_file(tmp_path):
    """The cold log grows forever, and the hook that tails it blocks a session.

    Reading the entire file to return twenty-five lines puts an unbounded cost
    in the session-start path: after a year the hook reads tens of megabytes and
    hashes every line to return a fixed-size window.
    """
    from durable_context import Config, FileStore

    config = Config(home=tmp_path / "state", tail_lines=5)
    path = config.log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"- entry {i}\n" for i in range(200_000)), encoding="utf-8")

    reads: list[int] = []

    class CountingStore(FileStore):
        def read(self, target):
            reads.append(len(super().read(target)))
            return super().read(target)

    context = DurableContext(config=config, store=CountingStore(), clock=lambda: FROZEN)
    tail = context.load_log_tail()

    assert tail[-1] == "- entry 199999"
    assert len(tail) == 5
    assert reads == []  # the whole file was never pulled into memory


def test_status_does_not_count_every_line_of_the_log(tmp_path):
    """``status`` is a debugging command, not a full-table scan."""
    from durable_context import Config, FileStore

    config = Config(home=tmp_path / "state", tail_lines=5)
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    config.log_path.write_text("".join(f"- entry {i}\n" for i in range(50_000)), encoding="utf-8")

    context = DurableContext(config=config, store=FileStore(), clock=lambda: FROZEN)
    assert context.status()["log_tail_lines"] == 5
