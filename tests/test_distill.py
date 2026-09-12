"""Distillation: the deterministic default, and the optional model behind it."""

from __future__ import annotations

import pytest

from durable_context import (
    CommandDistiller,
    FallbackDistiller,
    HeuristicDistiller,
    Message,
    build_prompt,
    extract_json,
)


def users(*texts: str) -> list[Message]:
    return [Message(role="user", text=text) for text in texts]


def test_heuristic_extracts_decision_target_reason_and_rejection():
    """The default path must work with no model, no network and no clock.

    This distiller runs inside a pre-compaction hook. A hook that waits on a
    model can hang the host at the exact moment the user is losing context, so
    the guaranteed-available path has to be pure computation - and it has to
    capture the rejected option, which is the field that stops the next session
    re-proposing it.
    """
    decision = HeuristicDistiller().distill(
        users("Let's go with Postgres instead of MongoDB because the ops team already runs it.")
    )
    assert decision is not None
    assert decision.target == "Postgres"
    assert decision.discarded == "MongoDB"
    assert decision.why == "the ops team already runs it"
    assert decision.source == "heuristic"


def test_heuristic_prefers_the_most_recent_decision():
    """In a long session the newest statement is the one in force.

    Reading oldest-first would lock in a choice the user has already moved on
    from, which is the same relitigation failure in reverse.
    """
    decision = HeuristicDistiller().distill(
        users(
            "We'll use Redis for the job queue.",
            "Actually, switching to RabbitMQ for the job queue.",
        )
    )
    assert decision is not None
    assert "RabbitMQ" in decision.target
    assert "Redis" not in decision.decision


def test_heuristic_records_the_latest_turn_when_nobody_said_decide():
    """Most real sessions never contain the word "decided".

    A weak record of what the user last asked for still beats an empty file,
    because an empty file is indistinguishable from "nothing was happening".
    """
    decision = HeuristicDistiller().distill(
        users("The staging box drops connections every midnight.")
    )
    assert decision is not None
    assert "staging box" in decision.decision


def test_heuristic_never_promotes_an_assistant_suggestion_to_a_decision():
    """Only the human decides.

    If the agent's own proposal could be distilled into the active decision, the
    system would manufacture consent: the next session would read a locked
    target the user never agreed to.
    """
    messages = [
        Message(role="assistant", text="We should go with gRPC for the internal transport."),
        Message(role="user", text="Not yet, leave the transport alone."),
    ]
    decision = HeuristicDistiller().distill(messages)
    assert decision is not None
    assert "gRPC" not in decision.decision


def test_heuristic_returns_none_without_human_turns():
    """No human input means nothing to preserve.

    Writing a decision anyway would create a confident-looking record derived
    entirely from machine output.
    """
    assert HeuristicDistiller().distill([Message(role="assistant", text="Working on it.")]) is None


def test_extract_json_skips_a_preamble_object():
    """Models emit their own scratch objects before the answer.

    Taking the first ``{`` would capture a reasoning blob and silently record an
    empty decision. Scanning for the first object that carries decision fields
    is what makes the parser robust to model chattiness.
    """
    payload = extract_json(
        '{"thinking": "let me consider"} then prose '
        '{"decision": "ship v2", "target": "v2", "why": "deadline"}'
    )
    assert payload == {"decision": "ship v2", "target": "v2", "why": "deadline"}


def test_extract_json_handles_code_fences():
    """Fenced JSON is the single most common model output shape."""
    payload = extract_json('```json\n{"decision": "use the cache", "target": "cache"}\n```')
    assert payload is not None and payload["target"] == "cache"


def test_extract_json_returns_none_for_prose():
    """A model that ignores the format contract must degrade, not corrupt.

    Returning a half-parsed dict here would write nonsense into the durable
    layer, which is worse than writing nothing.
    """
    assert extract_json("I think you should probably use Postgres.") is None


def test_command_distiller_swallows_every_failure_mode():
    """A distiller must never raise inside a pre-compaction hook.

    Missing binary, non-zero exit, timeout, prose instead of JSON - all four are
    routine. Any of them propagating would cost the user the context this
    package exists to protect.
    """

    def explode(command, prompt, timeout):
        raise TimeoutError("model took too long")

    distiller = CommandDistiller(command=("model-runner",), runner=explode)
    assert distiller.distill(users("Go with the queue.")) is None

    prose = CommandDistiller(command=("model-runner",), runner=lambda c, p, t: "sure thing!")
    assert prose.distill(users("Go with the queue.")) is None


def test_command_distiller_parses_a_well_behaved_model():
    """The contract is stdin prompt, stdout JSON - no vendor SDK involved.

    Keeping the seam at argv is what holds runtime dependencies at zero while
    still allowing a better distiller to be plugged in.
    """
    captured = {}

    def runner(command, prompt, timeout):
        captured["prompt"] = prompt
        return '{"decision": "adopt the new schema", "target": "schema v3", "discarded": "schema v2"}'

    distiller = CommandDistiller(command=("model-runner", "--json"), runner=runner)
    decision = distiller.distill(users("We are adopting schema v3, dropping v2."))
    assert decision is not None
    assert decision.discarded == "schema v2"
    assert decision.source == "command"
    assert "We are adopting schema v3" in captured["prompt"]


def test_fallback_keeps_a_record_when_the_model_is_unavailable():
    """This is how an LLM stays an upgrade rather than a dependency.

    With the command first and the heuristic behind it, a broken or slow model
    degrades the *quality* of the record, never its existence.
    """
    dead = CommandDistiller(command=("missing",), runner=lambda c, p, t: "")
    distiller = FallbackDistiller(distillers=(dead, HeuristicDistiller()))
    decision = distiller.distill(users("Let's go with the batch importer."))
    assert decision is not None
    assert decision.source == "heuristic"


def test_fallback_prefers_the_first_usable_answer():
    """Order expresses preference, and the better distiller goes first."""
    good = CommandDistiller(
        command=("model",), runner=lambda c, p, t: '{"decision": "model answer", "target": "T"}'
    )
    distiller = FallbackDistiller(distillers=(good, HeuristicDistiller()))
    decision = distiller.distill(users("Let's go with something else entirely."))
    assert decision is not None
    assert decision.source == "command"


def test_prompt_truncation_keeps_the_recent_turns():
    """When the window overflows, the *opening* of the session is expendable.

    The live decision is stated late. Truncating the tail would throw away the
    only part of the conversation that answers the question being asked.
    """
    messages = users(*[f"old turn {i}" for i in range(400)], "Final call: we ship on Friday.")
    prompt = build_prompt(messages, max_chars=200)
    assert "Final call: we ship on Friday." in prompt


@pytest.mark.parametrize(
    "text, expected",
    [
        ("We are dropping the Kafka bridge and going with direct writes.", "direct writes"),
        ("Switching to the async client rather than the threaded one.", "the async client"),
    ],
)
def test_heuristic_handles_common_phrasings(text, expected):
    """Cue coverage is the heuristic's whole quality budget.

    These are the phrasings people actually type when they change direction;
    missing them means the pivot is recorded as ordinary chatter.
    """
    decision = HeuristicDistiller().distill(users(text))
    assert decision is not None
    # Assert on the target itself. Falling back to ``decision`` - the whole
    # turn, which trivially contains the expected substring - made this test
    # pass with target extraction completely broken, and an always-empty target
    # sends every later merge down the wrong branch.
    assert expected in decision.target


@pytest.mark.parametrize(
    "text",
    [
        "Do not use the Redis cache for sessions, it loses data on failover.",
        "We are not going with the Redis cache for sessions.",
        "Never switch to the Redis cache for sessions.",
    ],
)
def test_a_prohibition_is_not_recorded_as_the_locked_target(text):
    """"Do not use X" must never come back as "TARGET: X".

    The decision cues match inside the prohibition ("do not **use** ..."), so
    without a negation guard the package injected the thing the user forbade
    under a header telling the next session it is the target currently locked
    in - the exact inverse of what was said, stated with authority.
    """
    decision = HeuristicDistiller().distill(users(text))
    assert decision is not None
    assert "redis" not in decision.target.lower()


def test_hedging_is_not_recorded_as_a_rejection():
    """"I am not sure" is doubt, not a ruled-out path.

    The bare ``not`` discard cue turned the rest of the sentence into a
    REJECTED line, which the injection tells the next session to honour.
    """
    decision = HeuristicDistiller().distill(users("I am not sure we should use Redis here."))
    assert decision is not None
    assert decision.discarded == ""
    assert "redis" not in decision.target.lower()


def test_a_negation_in_an_earlier_clause_does_not_suppress_a_real_decision():
    """The guard looks back within the clause, not across the sentence."""
    decision = HeuristicDistiller().distill(
        users("We are not using Mongo, so let's go with Postgres.")
    )
    assert decision is not None
    assert decision.target == "Postgres"
