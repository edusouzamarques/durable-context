"""Assembling a working context from configuration.

Kept separate from :mod:`session` so that constructing a :class:`DurableContext`
by hand - with a fake store, a frozen clock and a stub distiller - stays the
easy path in tests, while the convenience path stays one call for real use.
"""

from __future__ import annotations

from typing import Mapping

from .config import Config
from .distill import CommandDistiller, Distiller, FallbackDistiller, HeuristicDistiller
from .session import DurableContext
from .store import FileStore

__all__ = ["build_distiller", "open_context"]


def build_distiller(config: Config) -> Distiller:
    """Pick a distiller for this configuration.

    With no command configured you get the deterministic heuristic - the whole
    package works, offline, with no model and no network. Configure a command
    and you get that command *with the heuristic behind it*, so a timeout, a
    missing binary or a model that answers in prose degrades to a weaker record
    instead of to no record at all.
    """
    if not config.distill_command:
        return HeuristicDistiller()
    return FallbackDistiller(
        distillers=(
            CommandDistiller(command=config.distill_command, timeout=config.distill_timeout),
            HeuristicDistiller(),
        )
    )


def open_context(
    config: Config | None = None,
    *,
    env: Mapping[str, str] | None = None,
    home: str | None = None,
) -> DurableContext:
    """Build a :class:`DurableContext` backed by real files."""
    resolved = config or Config.from_env(env, home=home)
    return DurableContext(
        config=resolved,
        store=FileStore(),
        distiller=build_distiller(resolved),
    )
