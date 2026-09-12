# durable-context

[![CI](https://github.com/edusouzaxGV/durable-context/actions/workflows/ci.yml/badge.svg)](https://github.com/edusouzaxGV/durable-context/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://github.com/edusouzaxGV/durable-context)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/edusouzaxGV/durable-context/blob/main/LICENSE)

State that survives conversation compaction. Zero runtime dependencies.

## The failure this prevents

You spend forty minutes with an agent working out that the event store should be
ClickHouse, not sharded Postgres, because reads are 40x writes. The session gets
long. The host compacts it.

The compaction summary is accurate and useless:

> Explored database options. Set up the ClickHouse schema. Wrote the ingestion adapter.

It kept what was **done**. It dropped what was **decided**: the reason, and the
option that was explicitly ruled out. Ten minutes later the agent says:

> Have you considered sharded Postgres here? It would simplify the deployment.

You now have the same argument a second time. If the pivot happened mid-session
it is worse — the agent quietly resumes work on the abandoned path, because the
last thing it can see is the summary, and the summary never said "rejected".

Two properties cause this, and neither is a bug in the summariser:

1. **Summarisation optimises for narrative, not for constraints.** "We chose X"
   compresses well. "We chose X *and ruled out Y*" is the half that gets cut.
2. **Anything that depends on the agent remembering to save state does not
   happen.** Mid-session is exactly when the agent is busy, and mid-session is
   exactly where compaction hurts most.

## What it does instead

Two layers of state next to the conversation, plus one reactive trigger.

| Layer | File | Lifecycle |
|---|---|---|
| Hot checkpoint | `checkpoint.md` | Overwritten every compaction, re-injected every session start |
| Active decision | `active_decision.md` | Overwritten; always carries the **rejected** path |
| Cold decision log | `decision_log.md` | Append-only. Never rewritten. Only ever tailed |

The trigger is the part that matters. Distillation runs on the host's
**pre-compaction** event, not on the agent's good intentions:

```
host: "about to compact"
  -> raw human intent appended to the cold log      (written first; never lost)
  -> decision distilled, merged with what is on record
  -> hot layer overwritten, timestamped
host: compacts
...
host: "session starting"
  -> active decision, checkpoint and log tail injected, in that order
```

The pivot itself is treated as evidence. When the recorded target changes from A
to B and nothing says why, the old target is written into the `REJECTED` field
automatically — no model call required to infer that moving away from something
means you are not doing it. That inference needs **both** targets to be named: a
record that does not name a target cannot pivot away from one, and cannot
inherit one either.

## Install

Not on PyPI yet, so install from the repository:

```bash
pip install git+https://github.com/edusouzaxGV/durable-context
```

## Use it as a hook

```bash
durable-context hook-config
```

prints a snippet for a host that runs hooks as subprocesses (JSON on stdin, JSON
on stdout):

```json
{
  "hooks": {
    "PreCompact":   [{"hooks": [{"type": "command", "command": "durable-context pre-compact --hook"}]}],
    "SessionStart": [{"hooks": [{"type": "command", "command": "durable-context inject --hook"}]}]
  }
}
```

`pre-compact` writes state and prints `{}` — anything it printed would land in
the context it is trying to protect. `inject` prints the host envelope
containing the context block, or `{}` when there is nothing worth saying.

## Use it as a library

```python
from durable_context import open_context

context = open_context()                       # state in ~/.durable-context

with open(transcript_path) as handle:
    context.pre_compact(handle.readlines(), trigger="auto")

print(context.session_start().text)
```

## Use it from the CLI

```bash
durable-context status                       # what is preserved, and how old it is
durable-context decision                     # the active decision, rejection included
durable-context decision --set "use ClickHouse" --target ClickHouse \
                         --why "reads are 40x writes" --rejected "sharded Postgres"
durable-context checkpoint --set "GOTCHAS=the CDN caches 404s for an hour"
durable-context log --lines 40               # tail the append-only history
durable-context inject                       # what the next session would be told
durable-context reset --what decision --yes  # the only destructive command
```

Writes merge by default: setting one field never blanks the others, and clearing
a checkpoint section requires `--clear`. `reset` refuses to run without `--yes`,
and never touches the append-only log. Decision fields are one line each, so a
value spanning several lines is folded into one rather than silently truncated.

## The distiller is pluggable, and the default needs no model

`HeuristicDistiller` is pure Python: cue matching over the human turns, no
network, no clock, no model. It is the default because this code runs inside a
pre-compaction hook, where a hang is worse than a mediocre answer.

To upgrade, point at any command that reads a prompt on stdin and prints JSON:

```bash
export DURABLE_CONTEXT_DISTILL_CMD="my-model-runner --json"
```

That gives you `FallbackDistiller(command, heuristic)` — the model when it
answers, the heuristic when it times out, is missing, or replies in prose. No
vendor is named anywhere in this package, and there are no runtime dependencies
to install.

| Variable | Default | Meaning |
|---|---|---|
| `DURABLE_CONTEXT_HOME` | `~/.durable-context` | State directory |
| `DURABLE_CONTEXT_DISTILL_CMD` | *(unset)* | argv for the optional distiller |
| `DURABLE_CONTEXT_DISTILL_TIMEOUT` | `45` | Seconds before giving up on it |
| `DURABLE_CONTEXT_STALE_AFTER` | `6h` | When to warn that the checkpoint is old |
| `DURABLE_CONTEXT_TAIL_LINES` | `25` | Cold-log lines to inject |
| `DURABLE_CONTEXT_MAX_CHARS` | `12000` | Total injection budget |

## Design

The rules live in pure functions and the I/O lives at the edge, so the behaviour
that matters is testable without a network, a clock or a filesystem:

```
models.py       immutable values
transcript.py   JSONL -> messages            (pure)
distill.py      messages -> Decision         (pure default; optional subprocess)
merge.py        old + new -> Decision        (pure)  <- the rejection rules
documents.py    render/parse + staleness     (pure)
coldlog.py      append-only entries, tail    (pure)
injection.py    values -> injected text      (pure)  <- ordering and truncation
session.py      sequences the above          (takes store + clock + distiller)
store.py        files, atomic writes         (the only filesystem code)
hooks.py        host adapter                 (thin, and forbidden from raising)
cli.py          argparse
```

`DurableContext` takes its store, clock and distiller as arguments, so the whole
engine runs in memory at a fixed instant in the test suite.

## Things that are deliberately defensive

- **The compaction summary is never recorded as intent.** A host resumes a
  compacted session by injecting the summary as a user-role message. Recording
  it creates a loop: logged, re-injected, session born large, compacts sooner,
  logged again. Observed in production as several hundred stacked copies and a
  compaction every few minutes.
- **Hooks never raise.** The worst case of installing this is that it does
  nothing. It must never cost you a compaction or a session start.
- **Nothing fits? The decision survives, the history does not.** Truncation drops
  from the bottom, and an oversized decision block is cut rather than dropped.
- **Old state is labelled, not silently trusted.** An aged checkpoint is injected
  with its age; an undated one is marked unverified; a future timestamp is
  injected with a clock-skew warning rather than treated as eternally fresh.
- **Hook mode only ever prints JSON.** Even a malformed environment variable
  yields `{}` on stdout, because the host parses that stream and prose in it
  breaks the session start this package exists to improve.
- **Writes are atomic.** Compaction is triggered by the host at a moment of its
  choosing; a half-written checkpoint reads as valid and is wrong.

## Limitations, honestly

- **The heuristic distiller is shallow, and English-only.** It matches cue
  phrases ("going with", "switching to", "instead of") over the human turns. It
  will miss a decision reached implicitly over several exchanges, and it will
  happily record a hypothetical phrased like a decision. It guarantees *a*
  structured record, not a good one. Configure a model distiller if you want
  nuance.
- **It records what was said, not what is true.** If the user changed direction
  in a tool call, a file, or another window, the decision file will be confidently
  wrong. This is why staleness is surfaced: live state always wins over recorded
  state.
- **One active decision per state directory.** Projects with several independent
  threads in flight need several directories (`--home`), because merging
  unrelated decisions into one record is worse than keeping none.
- **The pivot inference is a heuristic too.** A changed target is usually a
  rejection; sometimes it is just a different subject. It produces an occasional
  spurious `REJECTED` line, which is the direction the error was chosen to fall
  in — over-recording a constraint is cheaper than re-arguing a settled one. It
  fires only when the old and new records both *name* a target; a target-less
  record neither inherits a rejection nor creates one, because a false
  constraint that never expires is a worse failure than a thinner record.
- **The heuristic distiller is English-only in its negation handling too.** It
  refuses to read "do not use X" as a decision for X, but the guard is a list of
  English negators; another language will fall through it.
- **No host is bundled.** The hook adapter matches the common
  JSON-in/JSON-out subprocess convention. Other hosts need a small shim; that is
  what `session.py` is for.
- **The cold log grows forever, on purpose.** There is no rotation, and a record
  you are willing to rewrite is not a record. Only the tail is ever read — the
  file is read back-to-front within a byte budget, so a year of history costs
  the session-start hook no more than a day of it.

## Licence

MIT. Copyright (c) 2026 Eduardo de Souza Marques.
