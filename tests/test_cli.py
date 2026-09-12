"""CLI behaviour. Destructive operations are opt-in; writes merge by default."""

from __future__ import annotations

import io
import json

from durable_context.cli import main

from conftest import transcript


def run(*argv, stdin=None):
    out = io.StringIO()
    code = main(list(argv), out=out, stdin=stdin or io.StringIO(""))
    return code, out.getvalue()


def test_reset_refuses_without_explicit_confirmation(tmp_path):
    """The only destructive command must never fire by accident.

    A bare ``reset`` typed from muscle memory, or reached by a script's default
    arguments, would discard exactly the state this package exists to keep.
    """
    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "ship v2", "--target", "v2")

    code, output = run("--home", home, "reset")
    assert code == 2
    assert "--yes" in output

    _, shown = run("--home", home, "decision")
    assert "ship v2" in shown


def test_reset_never_touches_the_append_only_log(tmp_path):
    """History is the layer that cannot be regenerated.

    Clearing the hot layer is a recovery action; deleting the record of how the
    project got here is data loss, so ``reset`` is not allowed to do it.
    """
    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "ship v2", "--target", "v2")
    run("--home", home, "reset", "--what", "all", "--yes")

    _, log = run("--home", home, "log")
    assert "ship v2" in log
    _, shown = run("--home", home, "decision")
    assert "no decision recorded" in shown


def test_setting_one_field_does_not_blank_the_others(tmp_path):
    """Hand-correcting the durable layer is expected, and must be safe.

    If ``--next`` erased ``--why``, the first correction a user made would
    destroy the reasoning behind the decision.
    """
    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "use ClickHouse", "--target", "ClickHouse",
        "--why", "40x read volume")
    run("--home", home, "decision", "--target", "ClickHouse", "--next", "write the adapter")

    _, output = run("--home", home, "decision", "--json")
    payload = json.loads(output)
    assert payload["why"] == "40x read volume"
    assert payload["next_step"] == "write the adapter"


def test_replace_is_available_when_you_really_mean_it(tmp_path):
    """Merging is the safe default, not a cage."""
    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "old", "--target", "old", "--why", "stale reason")
    run("--home", home, "decision", "--set", "new", "--target", "new", "--replace")

    _, output = run("--home", home, "decision", "--json")
    assert json.loads(output)["why"] == ""


def test_pre_compact_from_a_transcript_file(tmp_path):
    """The CLI has to be usable without a host, for testing and for cron.

    If the only way to exercise the engine were a live agent hook, nobody could
    verify their configuration before trusting it.
    """
    home = str(tmp_path / "state")
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "".join(transcript(("user", "Let's go with Postgres instead of MongoDB."))),
        encoding="utf-8",
    )
    code, output = run("--home", home, "pre-compact", "--transcript", str(path))
    assert code == 0
    assert "preserved decision" in output

    _, injected = run("--home", home, "inject")
    assert "REJECTED: MongoDB" in injected


def test_pre_compact_reports_an_unreadable_transcript(tmp_path):
    """A typo in a path should say so, not look like success."""
    code, output = run(
        "--home", str(tmp_path / "state"), "pre-compact", "--transcript", str(tmp_path / "nope")
    )
    assert code == 1
    assert "cannot read transcript" in output


def test_inject_prints_nothing_on_a_fresh_install(tmp_path):
    """Silence is the correct first-run output."""
    code, output = run("--home", str(tmp_path / "state"), "inject")
    assert code == 0
    assert output == ""


def test_inject_hook_mode_emits_valid_json(tmp_path):
    """The host parses this. Anything non-JSON on stdout breaks the session start."""
    code, output = run("--home", str(tmp_path / "state"), "inject", "--hook")
    assert code == 0
    assert json.loads(output) == {}


def test_pre_compact_hook_mode_reads_stdin(tmp_path):
    """End to end through the documented host contract: JSON in, JSON out."""
    home = str(tmp_path / "state")
    path = tmp_path / "transcript.jsonl"
    path.write_text(
        "".join(transcript(("user", "Let's go with the batch importer."))), encoding="utf-8"
    )
    payload = json.dumps({"transcript_path": str(path), "trigger": "auto"})
    code, output = run("--home", home, "pre-compact", "--hook", stdin=io.StringIO(payload))
    assert code == 0
    assert json.loads(output) == {}

    _, status = run("--home", home, "status", "--json")
    assert "batch importer" in json.loads(status)["decision"]["decision"]


def test_checkpoint_sections_are_set_and_shown(tmp_path):
    """Sections are free text so a project can carry its own vocabulary."""
    home = str(tmp_path / "state")
    run("--home", home, "checkpoint", "--set", "NOW=porting the uploader",
        "--set", "GOTCHAS=the CDN caches 404s")
    _, output = run("--home", home, "checkpoint")
    assert "porting the uploader" in output
    assert "the CDN caches 404s" in output


def test_a_malformed_section_argument_is_rejected(tmp_path):
    """Silently creating a section named after the whole argument would be worse."""
    code, output = run("--home", str(tmp_path / "state"), "checkpoint", "--set", "NOW porting")
    assert code == 2
    assert "NAME=BODY" in output


def test_clearing_a_section_requires_naming_it(tmp_path):
    """Merging protects untouched sections; ``--clear`` is the explicit escape."""
    home = str(tmp_path / "state")
    run("--home", home, "checkpoint", "--set", "GOTCHAS=the CDN caches 404s")
    run("--home", home, "checkpoint", "--set", "NOW=writing tests")
    _, kept = run("--home", home, "checkpoint")
    assert "the CDN caches 404s" in kept

    run("--home", home, "checkpoint", "--clear", "GOTCHAS")
    _, cleared = run("--home", home, "checkpoint")
    assert "the CDN caches 404s" not in cleared


def test_hook_config_is_copy_pasteable_json(tmp_path):
    """Installation instructions that require hand-editing JSON get typo'd."""
    code, output = run("--home", str(tmp_path / "state"), "hook-config")
    assert code == 0
    config = json.loads(output)
    assert "PreCompact" in config["hooks"]
    assert "SessionStart" in config["hooks"]


def test_status_is_readable_by_a_human_and_by_a_script(tmp_path):
    """Both audiences exist; neither should have to parse the other's format."""
    home = str(tmp_path / "state")
    code, human = run("--home", home, "status")
    assert code == 0
    assert "last compaction never" in human

    code, machine = run("--home", home, "status", "--json")
    assert code == 0
    assert json.loads(machine)["last_compact"] is None


def test_a_one_field_edit_never_fabricates_a_rejection(tmp_path):
    """The documented guarantee, on the path the README itself suggests.

    ``decision --why "..."`` used to be read as a pivot to an unnamed target:
    it wiped decision, target and why, and wrote the live target into REJECTED.
    The next session was then told to honour a rejection of the very thing that
    had just been confirmed.
    """
    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "use ClickHouse", "--target", "ClickHouse",
        "--why", "reads are 40x writes", "--rejected", "sharded Postgres")
    run("--home", home, "decision", "--why", "reads are 40x writes and growing")

    payload = json.loads(run("--home", home, "decision", "--json")[1])
    assert payload["decision"] == "use ClickHouse"
    assert payload["target"] == "ClickHouse"
    assert payload["discarded"] == "sharded Postgres"

    _, injected = run("--home", home, "inject")
    assert "REJECTED: ClickHouse\n" not in injected


def test_a_multi_line_value_is_kept_whole(tmp_path):
    """Heredocs and multi-line strings from scripts hit this immediately.

    The decision document is one line per field, so an unfolded value parsed
    back truncated at the first newline - losing the reason - while the CLI
    still printed "decision saved".
    """
    home = str(tmp_path / "state")
    _, saved = run("--home", home, "decision", "--json",
                   "--set", "use ClickHouse\nbecause reads are 40x writes", "--target", "ClickHouse")
    # What the write reports and what the file holds have to agree.
    assert "\n" not in json.loads(saved)["decision"]

    payload = json.loads(run("--home", home, "decision", "--json")[1])
    assert "because reads are 40x writes" in payload["decision"]


def test_hook_mode_prints_json_even_with_a_broken_environment(tmp_path, monkeypatch):
    """The host json.loads() this stream; prose on it breaks the session start.

    A plausible typo in an environment variable used to print an error sentence
    and exit 2 from ``inject --hook`` and ``pre-compact --hook`` - every session
    start and every compaction, for as long as the typo survived.
    """
    home = str(tmp_path / "state")
    monkeypatch.setenv("DURABLE_CONTEXT_STALE_AFTER", "6 hours")
    code, output = run("--home", home, "inject", "--hook")
    assert code == 0
    assert json.loads(output) == {}

    monkeypatch.setenv("DURABLE_CONTEXT_STALE_AFTER", "6h")
    monkeypatch.setenv("DURABLE_CONTEXT_MAX_CHARS", "abc")
    code, output = run("--home", home, "pre-compact", "--hook", stdin=io.StringIO("{}"))
    assert code == 0
    assert json.loads(output) == {}


def test_a_broken_environment_is_still_reported_to_a_human(tmp_path, monkeypatch):
    """Silence is right for the host, wrong for someone typing the command."""
    monkeypatch.setenv("DURABLE_CONTEXT_STALE_AFTER", "6 hours")
    code, output = run("--home", str(tmp_path / "state"), "status")
    assert code == 2
    assert "configuration error" in output


def test_output_survives_a_console_that_cannot_encode_it(tmp_path):
    """The state files are UTF-8 by design; a stock Windows console is not.

    One character outside the console code page used to turn ``inject``,
    ``status``, ``log`` and ``decision`` into a traceback - so any non-English
    user, or any decision quoting a non-Latin-1 identifier, lost the commands
    entirely.
    """

    class NarrowStream:
        encoding = "cp1252"

        def __init__(self):
            self.written = []

        def write(self, text):
            text.encode(self.encoding)  # raises UnicodeEncodeError, like the console
            self.written.append(text)
            return len(text)

    home = str(tmp_path / "state")
    run("--home", home, "decision", "--set", "use the 日本語 parser", "--target", "parser")

    stream = NarrowStream()
    code = main(["--home", home, "inject"], out=stream, stdin=io.StringIO(""))
    assert code == 0
    assert "parser" in "".join(stream.written)


def test_nothing_is_defined_after_the_main_guard():
    """Code stranded below ``if __name__ == "__main__"`` is dead by construction.

    The one function that lived there was imported by nothing and its comment
    was wrong in both halves: no test used it, and it read the real process
    environment rather than avoiding it.
    """
    import pathlib

    from durable_context import cli

    assert not hasattr(cli, "default_config")
    source = pathlib.Path(cli.__file__).read_text(encoding="utf-8")
    tail = source.split('if __name__ == "__main__":', 1)[1]
    assert "\ndef " not in tail
    assert "\nclass " not in tail
