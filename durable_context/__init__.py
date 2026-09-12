"""durable-context: state that survives conversation compaction.

Long agent sessions get compacted. Compaction is lossy in a specific way: it
keeps what was *done* and drops what was *decided* - the reason behind a choice,
the option that was explicitly rejected, the threads still open. The next
session then relitigates settled questions.

This package keeps two layers next to the conversation:

* a **hot checkpoint**, overwritten on every compaction and re-injected at every
  session start;
* a **cold decision log**, append-only, never rewritten, only ever tailed.

The mechanism that makes it work is reactive. Distillation is triggered by the
host's pre-compaction event, so nothing depends on the agent remembering to
save - which it does not.

Typical use::

    from durable_context import open_context

    context = open_context()
    context.pre_compact(open(transcript).readlines(), trigger="auto")
    print(context.session_start().text)
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Eduardo de Souza Marques"
__license__ = "MIT"

from .coldlog import render_decision_entry, render_entry, select_tail
from .config import Config, parse_duration
from .distill import (
    CommandDistiller,
    Distiller,
    FallbackDistiller,
    HeuristicDistiller,
    build_prompt,
    extract_json,
)
from .documents import (
    parse_checkpoint,
    parse_decision,
    render_checkpoint,
    render_decision,
    staleness,
)
from .factory import build_distiller, open_context
from .injection import build_injection
from .merge import merge_checkpoint, merge_decision
from .models import Checkpoint, Decision, InjectionPayload, Message, Staleness
from .session import DurableContext, PreCompactResult
from .store import FileStore, MemoryStore, Store
from .transcript import is_compaction_summary, is_noise, parse_transcript, user_messages

__all__ = [
    "__version__",
    "Checkpoint",
    "CommandDistiller",
    "Config",
    "Decision",
    "Distiller",
    "DurableContext",
    "FallbackDistiller",
    "FileStore",
    "HeuristicDistiller",
    "InjectionPayload",
    "MemoryStore",
    "Message",
    "PreCompactResult",
    "Staleness",
    "Store",
    "build_distiller",
    "build_injection",
    "build_prompt",
    "extract_json",
    "is_compaction_summary",
    "is_noise",
    "merge_checkpoint",
    "merge_decision",
    "open_context",
    "parse_checkpoint",
    "parse_decision",
    "parse_duration",
    "parse_transcript",
    "render_checkpoint",
    "render_decision",
    "render_decision_entry",
    "render_entry",
    "select_tail",
    "staleness",
    "user_messages",
]
