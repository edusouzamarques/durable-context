"""Merge rules. These are where an overwrite-in-place layer stops losing data."""

from __future__ import annotations

from durable_context import Checkpoint, Decision, merge_checkpoint, merge_decision


def test_same_target_update_does_not_erase_the_reason():
    """A later distillation that forgets *why* must not delete the why.

    Distillation quality varies run to run. If each write replaced the record
    wholesale, one weak pass would silently destroy the reasoning that made the
    decision defensible - and reasoning is the first thing compaction drops, so
    it is the last thing this package may drop.
    """
    previous = Decision(
        decision="use the column store", target="ClickHouse", why="event volume is 40x reads"
    )
    new = Decision(decision="use the column store", target="ClickHouse")
    merged = merge_decision(previous, new)
    assert merged.why == "event volume is 40x reads"


def test_a_pivot_records_the_old_target_as_rejected():
    """This is the entire point of the package, in one rule.

    When work pivots from A to B and only "B" is recorded, the next session
    reads "we are doing B", helpfully suggests A, and the pivot is relitigated.
    Recording "B, and A was rejected" ends that loop - and the pivot itself is
    the evidence, so no model call is needed to infer it.
    """
    previous = Decision(decision="use MongoDB", target="MongoDB")
    new = Decision(decision="use Postgres", target="Postgres")
    merged = merge_decision(previous, new)
    assert merged.target == "Postgres"
    assert merged.discarded == "MongoDB"


def test_an_explicit_rejection_beats_the_inferred_one():
    """When the user names what they rejected, believe them.

    The inferred "previous target" is a fallback. Overwriting a stated rejection
    with a guess would degrade a strong record into a weak one.
    """
    previous = Decision(decision="use MongoDB", target="MongoDB")
    new = Decision(decision="use Postgres", target="Postgres", discarded="the managed DynamoDB plan")
    assert merge_decision(previous, new).discarded == "the managed DynamoDB plan"


def test_target_comparison_ignores_case_and_punctuation():
    """"Postgres", "postgres." and "  PostgreSQL " are not all the same thing.

    The first two are; the third is a different string and must not be silently
    treated as the same target. Normalisation is casefold plus punctuation only
    - anything fuzzier would merge two unrelated decisions, which is a worse
    failure than splitting one.
    """
    previous = Decision(decision="d", target="Postgres", why="ops already run it")
    merged = merge_decision(previous, Decision(decision="d", target="  postgres. "))
    assert merged.why == "ops already run it"
    assert merge_decision(previous, Decision(decision="d", target="Redis")).discarded == "Postgres"


def test_a_failed_distillation_leaves_the_record_intact():
    """No answer must never be written as an empty answer.

    A distiller returning nothing is routine (model down, timeout, silent
    session). Treating that as "there is no decision" would erase the live
    target at the precise moment it is about to be needed.
    """
    previous = Decision(decision="ship v2", target="v2")
    assert merge_decision(previous, None) == previous
    assert merge_decision(previous, Decision()) == previous


def test_checkpoint_merge_preserves_sections_the_caller_never_mentioned():
    """The hot layer is overwritten, so a partial write is a deletion.

    A caller that only knows about NOW would otherwise wipe GOTCHAS - which is
    exactly the knowledge nobody thinks to write down a second time.
    """
    previous = Checkpoint(sections={"NOW": "wiring the parser", "GOTCHAS": "the API lies about 404s"})
    merged = merge_checkpoint(previous, Checkpoint(sections={"NOW": "writing tests"}))
    assert merged.sections["NOW"] == "writing tests"
    assert merged.sections["GOTCHAS"] == "the API lies about 404s"


def test_clearing_a_section_has_to_be_asked_for():
    """Blank is ambiguous: it means "I have nothing to add", not "delete this".

    Making deletion explicit is what allows every other writer to be careless
    with sections it does not own.
    """
    previous = Checkpoint(sections={"GOTCHAS": "the API lies about 404s"})
    kept = merge_checkpoint(previous, Checkpoint(sections={"GOTCHAS": ""}))
    assert kept.sections["GOTCHAS"] == "the API lies about 404s"

    cleared = merge_checkpoint(
        previous, Checkpoint(sections={"GOTCHAS": ""}), allow_clear=("GOTCHAS",)
    )
    assert cleared.sections["GOTCHAS"] == ""


def test_checkpoint_merge_carries_the_previous_timestamp_when_none_is_given():
    """A content-only update must not make the checkpoint look undated.

    Losing the stamp would disable staleness detection, and state that is
    silently old is worse than state that is absent because the reader trusts it.
    """
    from datetime import datetime

    stamped = Checkpoint(sections={"NOW": "a"}, last_compact=datetime(2026, 3, 14, 9, 0))
    merged = merge_checkpoint(stamped, Checkpoint(sections={"NOW": "b"}))
    assert merged.last_compact == datetime(2026, 3, 14, 9, 0)


def test_two_target_less_decisions_are_not_treated_as_the_same_subject():
    """An empty target is not a target, and two blanks are not a match.

    The heuristic leaves the target empty whenever a turn contains no decision
    cue, which is most turns. Comparing the normalised blanks made every such
    record "the same subject" as the last one, so an unrelated new decision
    inherited the old REJECTED and WHY - a constraint the next session is told
    to honour, attached to something nobody rejected, that never expires
    because each later target-less record re-inherits it.
    """
    previous = Decision(
        decision="We are dropping the Kafka bridge for the audit trail.",
        discarded="the Kafka bridge for the audit trail",
        why="it never kept up with the write rate",
    )
    new = Decision(decision="The staging box drops connections every midnight.")
    merged = merge_decision(previous, new)
    assert merged.decision == "The staging box drops connections every midnight."
    assert merged.discarded == ""
    assert merged.why == ""


def test_a_record_that_names_no_target_cannot_pivot_away_from_one():
    """A subject that was never stated is not a subject that was rejected.

    Otherwise an ordinary turn distilled during the next compaction ("the
    staging box drops connections") inherits the decision slot and pushes the
    real, still-live target into REJECTED.
    """
    previous = Decision(decision="use Postgres", target="Postgres for the event store")
    new = Decision(decision="The staging box drops connections every midnight.")
    merged = merge_decision(previous, new)
    assert merged.discarded == ""
    assert merged.target == ""


def test_a_patch_with_no_subject_fills_gaps_instead_of_replacing():
    """``decision --why "..."`` is a correction, not a new decision.

    This is the documented guarantee that setting one field never blanks the
    others. It used to be read as a pivot to an unnamed target: decision,
    target and why were wiped and the live target was written into REJECTED,
    so a one-field correction told the next session to abandon the thing it had
    just been corrected about.
    """
    previous = Decision(
        decision="use ClickHouse",
        target="ClickHouse",
        why="reads are 40x writes",
        discarded="sharded Postgres",
    )
    merged = merge_decision(previous, Decision(why="reads are 40x writes and growing"))
    assert merged.decision == "use ClickHouse"
    assert merged.target == "ClickHouse"
    assert merged.why == "reads are 40x writes and growing"
    assert merged.discarded == "sharded Postgres"
