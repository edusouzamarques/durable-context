"""Configuration seams. Every vendor-specific detail has to be data, not code."""

from __future__ import annotations

from datetime import timedelta

import pytest

from durable_context import Config, HeuristicDistiller, build_distiller, parse_duration
from durable_context.distill import CommandDistiller, FallbackDistiller


def test_a_bare_number_means_minutes():
    """It is the unit people use when they talk about session length.

    Guessing seconds would make "30" mean half a minute of tolerance and mark
    every checkpoint stale.
    """
    assert parse_duration("30") == timedelta(minutes=30)
    assert parse_duration("6h") == timedelta(hours=6)
    assert parse_duration("2d") == timedelta(days=2)


def test_an_unparseable_duration_fails_loudly():
    """A silently-defaulted staleness window trains the reader to ignore it.

    If a typo in the environment quietly became "zero", every checkpoint would
    be flagged stale, the warning would become background noise, and the one
    time it mattered it would be skipped.
    """
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_the_distiller_command_is_argv_from_the_environment():
    """No vendor is named in this package, and none may be.

    The optional model-backed distiller is whatever command you configure, which
    is what keeps runtime dependencies at zero and keeps the package usable with
    a local runner, a hosted wrapper, or nothing at all.
    """
    config = Config.from_env(
        {"DURABLE_CONTEXT_DISTILL_CMD": "my-runner --model local --json"}, home="/tmp/x"
    )
    assert config.distill_command == ("my-runner", "--model", "local", "--json")


def test_no_command_configured_means_a_pure_default():
    """Out of the box this must work offline, with no model and no network."""
    distiller = build_distiller(Config.from_env({}, home="/tmp/x"))
    assert isinstance(distiller, HeuristicDistiller)


def test_a_configured_command_keeps_the_heuristic_behind_it():
    """An LLM is an upgrade, never a dependency.

    If the command is the only distiller, a timeout means no record at all. With
    the heuristic behind it, the worst case is a weaker record.
    """
    config = Config.from_env({"DURABLE_CONTEXT_DISTILL_CMD": "my-runner"}, home="/tmp/x")
    distiller = build_distiller(config)
    assert isinstance(distiller, FallbackDistiller)
    kinds = [type(d) for d in distiller.distillers]
    assert kinds == [CommandDistiller, HeuristicDistiller]


def test_an_explicit_home_beats_the_environment():
    """A ``--home`` flag has to win, or per-project state is impossible."""
    config = Config.from_env({"DURABLE_CONTEXT_HOME": "/from/env"}, home="/from/flag")
    assert str(config.home).replace("\\", "/").endswith("/from/flag")


def test_the_three_layers_live_in_separate_files():
    """A corrupted hot document must not take the append-only history with it.

    Separate files is the cheapest possible blast radius control, and the cold
    log is the layer that can never be regenerated.
    """
    config = Config(home="/state")
    paths = {config.checkpoint_path, config.decision_path, config.log_path}
    assert len(paths) == 3


def test_environment_overrides_are_typed():
    """Budgets arrive as strings and must not silently become string comparisons."""
    config = Config.from_env(
        {"DURABLE_CONTEXT_TAIL_LINES": "9", "DURABLE_CONTEXT_MAX_CHARS": "5000"}, home="/tmp/x"
    )
    assert config.tail_lines == 9
    assert config.max_injection_chars == 5000


def test_a_windows_command_keeps_its_path_separators():
    """POSIX splitting eats backslashes, and the damage is invisible.

    ``C:\tools\run.exe`` parsed as ``C:toolsrun.exe`` makes the configured
    distiller raise FileNotFoundError inside a hook that is forbidden from
    raising, so it degrades to the heuristic and the user is never told their
    model is not being called.
    """
    from durable_context.config import split_command

    assert split_command(r"C:\tools\model-runner\run.exe --json", windows=True) == (
        r"C:\tools\model-runner\run.exe",
        "--json",
    )
    assert split_command(r'"C:\Program Files\runner\run.exe" --json', windows=True) == (
        r"C:\Program Files\runner\run.exe",
        "--json",
    )
    # POSIX rules are unchanged: a backslash there really is an escape.
    assert split_command("runner --json", windows=False) == ("runner", "--json")


def test_from_env_splits_the_command_for_the_host_platform(monkeypatch):
    """The whole point is that this reaches the configured binary unmangled."""
    from durable_context import config as _cfg
    from durable_context.config import Config as _Config

    # Patch the seam, not os.name: faking os.name globally also makes
    # pathlib.Path() build a WindowsPath, which explodes on POSIX runners.
    monkeypatch.setattr(_cfg, "_running_on_windows", lambda: True)
    parsed = _Config.from_env({"DURABLE_CONTEXT_DISTILL_CMD": r"C:\tools\run.exe --json"})
    assert parsed.distill_command == (r"C:\tools\run.exe", "--json")
