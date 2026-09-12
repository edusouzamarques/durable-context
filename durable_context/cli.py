"""Command line interface.

Design rules, both learned the hard way:

* **Nothing destructive by default.** ``reset`` is the only command that removes
  anything and it refuses to run without ``--yes``. Everything else either reads
  or merges.
* **Writes merge by default.** Setting one field of the decision, or one section
  of the checkpoint, must not silently blank the others. ``--replace`` exists for
  when you really do mean "throw the rest away".
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import __version__
from .factory import open_context
from .hooks import pre_compact_hook, read_hook_input, session_start_hook
from .models import Checkpoint, Decision
from .session import DurableContext

__all__ = ["main", "build_parser"]

_HOOK_SNIPPET = {
    "hooks": {
        "PreCompact": [
            {"hooks": [{"type": "command", "command": "durable-context pre-compact --hook"}]}
        ],
        "SessionStart": [
            {"hooks": [{"type": "command", "command": "durable-context inject --hook"}]}
        ],
    }
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="durable-context",
        description="State that survives conversation compaction.",
    )
    parser.add_argument("--version", action="version", version=f"durable-context {__version__}")
    parser.add_argument(
        "--home",
        default=None,
        help="state directory (default: $DURABLE_CONTEXT_HOME or ~/.durable-context)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    inject = sub.add_parser("inject", help="print the context for a starting session")
    inject.add_argument(
        "--hook",
        action="store_true",
        help="emit the host hook envelope as JSON instead of plain text",
    )
    inject.add_argument("--max-chars", type=int, default=None, help="override the budget")

    pre = sub.add_parser("pre-compact", help="preserve state before a compaction")
    source = pre.add_mutually_exclusive_group(required=True)
    source.add_argument("--hook", action="store_true", help="read the hook payload from stdin")
    source.add_argument("--transcript", help="path to a JSONL transcript")
    pre.add_argument("--trigger", default="manual", help="what caused this compaction")

    status = sub.add_parser("status", help="show what is currently preserved")
    status.add_argument("--json", action="store_true", help="machine-readable output")

    decision = sub.add_parser("decision", help="read or set the active decision")
    decision.add_argument("--json", action="store_true", help="machine-readable output")
    decision.add_argument("--set", dest="set_decision", help="what is locked in now")
    decision.add_argument("--target", help="the exact object of the decision")
    decision.add_argument("--why", help="the real reason")
    decision.add_argument("--rejected", help="the path explicitly ruled out")
    decision.add_argument("--open", dest="open_thread", help="what is still unresolved")
    decision.add_argument("--next", dest="next_step", help="the concrete next action")
    decision.add_argument(
        "--replace",
        action="store_true",
        help="overwrite instead of merging with what is already recorded",
    )

    checkpoint = sub.add_parser("checkpoint", help="read or set hot checkpoint sections")
    checkpoint.add_argument(
        "--set",
        dest="sections",
        action="append",
        default=[],
        metavar="NAME=BODY",
        help="set one section; repeatable",
    )
    checkpoint.add_argument(
        "--clear",
        action="append",
        default=[],
        metavar="NAME",
        help="explicitly empty a section (merging never clears by accident)",
    )

    log = sub.add_parser("log", help="tail the append-only decision log")
    log.add_argument("--lines", type=int, default=None, help="how many lines to show")

    reset = sub.add_parser("reset", help="discard preserved state (destructive)")
    reset.add_argument(
        "--what",
        choices=("decision", "checkpoint", "all"),
        default="decision",
        help="what to discard; the append-only log is never touched",
    )
    reset.add_argument("--yes", action="store_true", help="required: confirm the deletion")

    sub.add_parser("hook-config", help="print a host hook configuration snippet")
    return parser


class _SafeWriter:
    """Write text that the console cannot encode, instead of crashing on it.

    The state files are UTF-8 by design, so a decision can legitimately quote a
    filename, an identifier or a sentence outside the console's code page. On a
    stock Windows console ``sys.stdout`` is that code page with strict error
    handling, and a single such character turned ``inject``, ``status``, ``log``
    and ``decision`` into a traceback. Escaping the unencodable characters loses
    a little fidelity; raising loses the whole command.
    """

    def __init__(self, stream) -> None:
        self._stream = stream

    def write(self, text: str):
        try:
            return self._stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self._stream, "encoding", None) or "ascii"
            safe = text.encode(encoding, errors="backslashreplace").decode(
                encoding, errors="replace"
            )
            return self._stream.write(safe)

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _context(args: argparse.Namespace) -> DurableContext:
    return open_context(home=args.home)


def _cmd_inject(args: argparse.Namespace, context: DurableContext, out) -> int:
    if args.hook:
        out.write(json.dumps(session_start_hook(context)) + "\n")
        return 0
    payload = context.session_start(max_chars=args.max_chars)
    if payload:
        out.write(payload.text + "\n")
    return 0


def _cmd_pre_compact(args: argparse.Namespace, context: DurableContext, out, stdin) -> int:
    if args.hook:
        payload = read_hook_input(stdin)
        pre_compact_hook(payload, context)
        out.write(json.dumps({}) + "\n")
        return 0
    try:
        with open(args.transcript, encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
    except OSError as error:
        out.write(f"cannot read transcript: {error}\n")
        return 1
    result = context.pre_compact(lines, trigger=args.trigger)
    out.write(result.summary() + "\n")
    return 0


def _cmd_status(args: argparse.Namespace, context: DurableContext, out) -> int:
    snapshot = context.status()
    if args.json:
        out.write(json.dumps(snapshot, indent=2) + "\n")
        return 0
    out.write(f"home            {snapshot['home']}\n")
    out.write(f"distiller       {snapshot['distiller']}\n")
    out.write(f"last compaction {snapshot['last_compact'] or 'never'} ({snapshot['staleness_reason']})\n")
    out.write(f"stale           {'yes' if snapshot['stale'] else 'no'}\n")
    out.write(f"sections        {', '.join(snapshot['checkpoint_sections']) or '(none)'}\n")
    out.write(f"log tail lines  {snapshot['log_tail_lines']}\n")
    decision = snapshot["decision"]
    out.write("decision        " + (decision.get("decision") or "(none recorded)") + "\n")
    if decision.get("discarded"):
        out.write("rejected        " + decision["discarded"] + "\n")
    return 0


def _cmd_decision(args: argparse.Namespace, context: DurableContext, out) -> int:
    fields = {
        "decision": args.set_decision,
        "target": args.target,
        "why": args.why,
        "discarded": args.rejected,
        "open_thread": args.open_thread,
        "next_step": args.next_step,
    }
    provided = {key: value for key, value in fields.items() if value is not None}
    if provided:
        # ``from_dict`` collapses whitespace. The decision document is one line
        # per field, so a value carrying a newline used to be written whole and
        # read back truncated at the newline - losing, silently, the reason the
        # record exists for - while the CLI still reported success.
        updated = context.save_decision(
            Decision.from_dict({**provided, "source": "manual"}), merge=not args.replace
        )
        if args.json:
            out.write(json.dumps(updated.to_dict(), indent=2) + "\n")
        else:
            out.write("decision saved\n")
        return 0

    current, distilled_at = context.load_decision()
    if args.json:
        payload = current.to_dict()
        payload["distilled_at"] = distilled_at.isoformat() if distilled_at else None
        out.write(json.dumps(payload, indent=2) + "\n")
        return 0
    if current.is_empty():
        out.write("no decision recorded\n")
        return 0
    out.write(context.store.read(context.config.decision_path))
    return 0


def _cmd_checkpoint(args: argparse.Namespace, context: DurableContext, out) -> int:
    if args.sections or args.clear:
        sections: dict[str, str] = {}
        for item in args.sections:
            name, separator, body = item.partition("=")
            if not separator:
                out.write(f"bad --set value {item!r}: expected NAME=BODY\n")
                return 2
            sections[name.strip()] = body.strip()
        for name in args.clear:
            sections[name.strip()] = ""
        context.save_checkpoint(
            Checkpoint(sections=sections), allow_clear=tuple(n.strip() for n in args.clear)
        )
        out.write("checkpoint saved\n")
        return 0
    text = context.store.read(context.config.checkpoint_path)
    out.write(text if text else "no checkpoint recorded\n")
    return 0


def _cmd_log(args: argparse.Namespace, context: DurableContext, out) -> int:
    lines = context.load_log_tail(max_lines=args.lines)
    if not lines:
        out.write("decision log is empty\n")
        return 0
    out.write("\n".join(lines) + "\n")
    return 0


def _cmd_reset(args: argparse.Namespace, context: DurableContext, out) -> int:
    if not args.yes:
        out.write("refusing to discard state without --yes\n")
        return 2
    targets = []
    if args.what in ("decision", "all"):
        targets.append(context.config.decision_path)
    if args.what in ("checkpoint", "all"):
        targets.append(context.config.checkpoint_path)
    for path in targets:
        context.store.write(path, "")
    out.write(f"cleared: {', '.join(p.name for p in targets)} (the decision log is untouched)\n")
    return 0


def main(argv: Sequence[str] | None = None, *, out=None, stdin=None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if out is None:
        out = sys.stdout
        # We own the process stream here, so widen it rather than only escaping
        # at our own write calls.
        reconfigure = getattr(out, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="backslashreplace")
            except (ValueError, OSError):  # pragma: no cover - exotic streams
                pass
    out = _SafeWriter(out)
    stdin = stdin if stdin is not None else sys.stdin

    try:
        context = _context(args)
    except ValueError as error:  # a malformed duration or count in the environment
        if getattr(args, "hook", False):
            # Hook mode is a JSON-in/JSON-out contract with the host. A bad
            # environment variable must not put prose on stdout: the host
            # json.loads() this, and breaking that breaks the session start -
            # the one thing this package promises never to cost the user.
            out.write(json.dumps({}) + "\n")
            return 0
        out.write(f"configuration error: {error}\n")
        return 2

    if args.command == "inject":
        return _cmd_inject(args, context, out)
    if args.command == "pre-compact":
        return _cmd_pre_compact(args, context, out, stdin)
    if args.command == "status":
        return _cmd_status(args, context, out)
    if args.command == "decision":
        return _cmd_decision(args, context, out)
    if args.command == "checkpoint":
        return _cmd_checkpoint(args, context, out)
    if args.command == "log":
        return _cmd_log(args, context, out)
    if args.command == "reset":
        return _cmd_reset(args, context, out)
    if args.command == "hook-config":
        out.write(json.dumps(_HOOK_SNIPPET, indent=2) + "\n")
        return 0
    parser.error(f"unknown command {args.command}")
    return 2  # pragma: no cover - argparse exits


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
