"""Configuration and its environment seams.

Everything a deployment might want to change lives here as data: where state is
kept, how long the hot layer stays trustworthy, and which command (if any) does
the optional model-backed distillation.

No vendor is named anywhere in this package. The optional distiller is
*whatever argv you configure*, which keeps the dependency surface at zero and
makes the package equally usable with a local runner, a hosted API wrapper, or
nothing at all.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Mapping

__all__ = [
    "Config",
    "ENV_PREFIX",
    "parse_duration",
    "split_command",
    "default_home",
]

ENV_PREFIX = "DURABLE_CONTEXT_"

_DURATION = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smhd]?)\s*$", re.IGNORECASE)
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "": 60}


def parse_duration(text: str) -> timedelta:
    """Parse ``90s`` / ``30m`` / ``6h`` / ``2d`` into a timedelta.

    A bare number means minutes, which is the unit people reach for when they
    talk about how long a session has been running. Anything unparseable raises
    rather than silently defaulting: a staleness window that quietly became
    "zero" would mark every checkpoint stale and train the reader to ignore the
    warning, which is worse than a loud startup failure.
    """
    match = _DURATION.match(text or "")
    if not match:
        raise ValueError(f"cannot parse duration: {text!r} (try 30m, 6h, 2d)")
    value = float(match.group("value"))
    unit = match.group("unit").lower()
    return timedelta(seconds=value * _UNIT_SECONDS[unit])


def _running_on_windows() -> bool:
    """Platform check behind a seam, so tests never fake ``os.name`` globally.

    Patching ``os.name`` to ``"nt"`` also makes ``pathlib.Path()`` return a
    ``WindowsPath``, which raises ``NotImplementedError`` on POSIX. That is a
    test-only hazard with no bearing on the behaviour under test, and it took
    down the whole CI run on Linux. Patch this function instead.
    """
    return os.name == "nt"


def split_command(text: str, *, windows: bool | None = None) -> tuple[str, ...]:
    """Split a configured command line into argv, correctly on either platform.

    :func:`shlex.split` defaults to POSIX rules, where a backslash is an escape
    character. Applied to a Windows path that costs you the separators -
    ``C:\\tools\\run.exe`` becomes ``C:toolsrun.exe`` - and the failure is
    invisible: the distiller raises :class:`FileNotFoundError` inside a hook
    that is forbidden from raising, so it degrades to the heuristic and nobody
    is told their model is never being called.

    On Windows the split is therefore non-POSIX (backslash is a path separator)
    and surrounding double quotes are stripped, which is what ``CreateProcess``
    would have done anyway.
    """
    if windows is None:
        windows = _running_on_windows()
    if not windows:
        return tuple(shlex.split(text))
    parts = []
    for part in shlex.split(text, posix=False):
        if len(part) >= 2 and part[0] == '"' and part[-1] == '"':
            part = part[1:-1]
        if part:
            parts.append(part)
    return tuple(parts)


def default_home() -> Path:
    """State directory used when nothing is configured.

    Deliberately *not* inside any agent vendor's directory: this package is the
    owner of its own state, and a user who changes agent tooling should keep
    their decision history.
    """
    return Path.home() / ".durable-context"


@dataclass(frozen=True)
class Config:
    """Immutable settings bundle; pass it, don't reach for globals."""

    home: Path
    stale_after: timedelta = timedelta(hours=6)
    tail_lines: int = 25
    max_injection_chars: int = 12000
    max_intent_lines: int = 12
    max_message_chars: int = 400
    distill_command: tuple[str, ...] = ()
    distill_timeout: float = 45.0

    def __post_init__(self) -> None:
        # Accept a string home so callers (and tests) do not have to import
        # pathlib just to point at a directory.
        if not isinstance(self.home, Path):
            object.__setattr__(self, "home", Path(str(self.home)).expanduser())

    # -- derived paths ----------------------------------------------------
    # Three files, one job each. Splitting them means a corrupted hot document
    # never takes the append-only history with it.

    @property
    def checkpoint_path(self) -> Path:
        """Hot layer: overwritten, re-injected at every session start."""
        return self.home / "checkpoint.md"

    @property
    def decision_path(self) -> Path:
        """Hot layer: the single active decision, including what was rejected."""
        return self.home / "active_decision.md"

    @property
    def log_path(self) -> Path:
        """Cold layer: append-only. Never rewritten, only ever tailed."""
        return self.home / "decision_log.md"

    def with_home(self, home: Path | str) -> "Config":
        return replace(self, home=Path(home).expanduser())

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        home: Path | str | None = None,
    ) -> "Config":
        """Build config from environment variables, with explicit override.

        Reading a mapping rather than :data:`os.environ` directly is what lets
        the resolution rules be tested without mutating global process state.
        """
        source = os.environ if env is None else env
        base = cls(home=default_home())

        resolved_home = home or source.get(f"{ENV_PREFIX}HOME")
        if resolved_home:
            base = base.with_home(resolved_home)

        stale = source.get(f"{ENV_PREFIX}STALE_AFTER")
        if stale:
            base = replace(base, stale_after=parse_duration(stale))

        command = source.get(f"{ENV_PREFIX}DISTILL_CMD")
        if command and command.strip():
            base = replace(base, distill_command=split_command(command))

        timeout = source.get(f"{ENV_PREFIX}DISTILL_TIMEOUT")
        if timeout:
            base = replace(base, distill_timeout=float(timeout))

        for key, field_name in (
            (f"{ENV_PREFIX}TAIL_LINES", "tail_lines"),
            (f"{ENV_PREFIX}MAX_CHARS", "max_injection_chars"),
            (f"{ENV_PREFIX}MAX_INTENT_LINES", "max_intent_lines"),
        ):
            raw = source.get(key)
            if raw:
                base = replace(base, **{field_name: int(raw)})

        return base
