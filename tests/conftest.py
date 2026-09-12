"""Shared fixtures.

The clock is frozen and the store is in memory on purpose: every behaviour worth
pinning in this package is a transformation of values, so the tests should not
need a temporary directory or a sleep to observe it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from durable_context import Config, DurableContext, HeuristicDistiller, MemoryStore

FROZEN = datetime(2026, 3, 14, 9, 30)


@pytest.fixture
def now():
    return FROZEN


@pytest.fixture
def clock():
    return lambda: FROZEN


@pytest.fixture
def config(tmp_path):
    return Config(home=tmp_path / "state", stale_after=timedelta(hours=6))


@pytest.fixture
def store():
    return MemoryStore()


@pytest.fixture
def context(config, store, clock):
    return DurableContext(
        config=config, store=store, distiller=HeuristicDistiller(), clock=clock
    )


def transcript(*turns: tuple[str, str]) -> list[str]:
    """Render (role, text) pairs as JSONL transcript lines."""
    return [
        json.dumps({"message": {"role": role, "content": text}}) + "\n" for role, text in turns
    ]


def block_transcript(role: str, *blocks: dict) -> list[str]:
    """Render one turn whose content is a list of typed blocks."""
    return [json.dumps({"message": {"role": role, "content": list(blocks)}}) + "\n"]
