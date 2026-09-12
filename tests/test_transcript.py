"""Parsing rules that exist because of real incidents."""

from __future__ import annotations

import json

from durable_context import is_noise, parse_transcript, user_messages

from conftest import block_transcript, transcript


def test_compaction_summary_is_never_recorded_as_intent():
    """The output of compaction must never become an input to compaction.

    Hosts resume a compacted session by injecting the summary as a user-role
    message. Recording it as fresh intent creates a feedback loop: the summary
    is logged, re-injected at the next session start, makes the session larger,
    triggers compaction sooner, and is logged again. In production this stacked
    hundreds of identical copies and drove a compaction every few minutes.
    """
    lines = transcript(
        ("user", "This session is being continued from a previous conversation. Earlier we chose X."),
        ("user", "Now add retry handling to the uploader please."),
    )
    messages = parse_transcript(lines)
    assert [m.text for m in messages] == ["Now add retry handling to the uploader please."]


def test_a_torn_line_does_not_lose_the_rest_of_the_transcript():
    """Transcripts are append-only logs read while another process writes them.

    A pre-compaction hook can read the file mid-write, so the last line may be
    truncated JSON. Losing one line is acceptable; raising and losing the whole
    checkpoint is not.
    """
    lines = transcript(("user", "Ship the parser refactor first.")) + ['{"message": {"role": "u']
    messages = parse_transcript(lines)
    assert [m.text for m in messages] == ["Ship the parser refactor first."]


def test_only_text_blocks_carry_intent():
    """Structured content mixes intent with transport.

    Tool calls and tool results are how the agent works, not what the user
    decided. Flattening them into the distillation window buries the signal
    under machine chatter.
    """
    lines = block_transcript(
        "user",
        {"type": "text", "text": "Deploy to staging before production."},
        {"type": "tool_result", "content": "exit status 0, 412 files changed"},
    )
    messages = parse_transcript(lines)
    assert [m.text for m in messages] == ["Deploy to staging before production."]


def test_bare_acknowledgements_are_dropped():
    """"ok" and "yes" add length to the window without adding intent.

    The distillation window is a scarce budget; filling it with agreement
    tokens costs the decision itself.
    """
    lines = transcript(("user", "ok"), ("user", "yes"), ("user", "Use the column store for events."))
    assert [m.text for m in parse_transcript(lines)] == ["Use the column store for events."]


def test_scaffolding_is_dropped_but_a_mention_of_it_is_kept():
    """The noise filter keys on how a turn *starts*, not on what it contains.

    A turn that is entirely a host-injected tag is scaffolding. A turn that
    merely talks about one is a person giving an instruction, and dropping it
    would silently discard real work.
    """
    assert is_noise("<system-reminder>be concise</system-reminder>")
    assert is_noise("Stop hook feedback: you ended the turn early")

    # The users of a package about agent hooks are people who type the words
    # "system-reminder" and "hookSpecificOutput" on purpose. A substring filter
    # deletes their instructions.
    kept = transcript(
        ("user", "Stop emitting a <system-reminder> block in the hook output."),
        ("user", "The hookSpecificOutput envelope needs an eventName field."),
    )
    assert len(parse_transcript(kept)) == 2


def test_a_pasted_file_cannot_dominate_the_window():
    """One turn must not consume the whole distillation budget.

    Users paste logs and files. Without a per-message bound, a single paste
    evicts every other turn from the window, including the one that states the
    decision.
    """
    lines = transcript(("user", "x" * 5000), ("user", "Go with the streaming parser."))
    messages = parse_transcript(lines, max_chars=400)
    assert all(len(m.text) <= 400 for m in messages)
    assert messages[-1].text == "Go with the streaming parser."


def test_user_messages_are_the_authoritative_statement_of_intent():
    """Assistant turns propose; user turns decide.

    Treating an assistant's suggestion as a recorded decision would let the
    agent lock in a choice the user never made.
    """
    lines = transcript(
        ("assistant", "I suggest we go with GraphQL for the public API."),
        ("user", "No, we are staying on REST for now."),
    )
    messages = parse_transcript(lines)
    assert len(messages) == 2
    assert [m.text for m in user_messages(messages)] == ["No, we are staying on REST for now."]


def test_non_dict_envelopes_are_skipped():
    """Log files accumulate lines from more than one writer.

    A bare JSON array or string on its own line must be ignored, not crash the
    parse.
    """
    lines = [json.dumps([1, 2, 3]) + "\n", json.dumps("hello") + "\n"]
    lines += transcript(("user", "Keep the batch size at 64."))
    assert [m.text for m in parse_transcript(lines)] == ["Keep the batch size at 64."]
