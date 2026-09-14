# Provenance

## Authorship

**Eduardo de Souza Marques** — author, architect and maintainer.
GitHub: [edusouzamarques](https://github.com/edusouzamarques)

## What this was extracted from

`durable-context` is a generalised, dependency-free extraction of a mechanism
the author built, ran and iterated on inside a private multi-agent automation
system over the course of 2026.

The private originals were three scripts wired into an agent host's hook system:

| Original role | Extracted into |
|---|---|
| Pre-compaction hook that appended raw human intent to an append-only log and stamped the hot checkpoint | `session.pre_compact`, `coldlog`, `transcript` |
| Decision distiller invoked synchronously before compaction, writing an "active decision" document | `distill`, `merge`, `documents` |
| Session-start hook that injected the active decision, the checkpoint and the log tail | `injection`, `hooks`, `session.session_start` |

### What was kept: the model, not the code

The scripts were not transliterated. They were read for the operating model they
encoded, and that model was rebuilt with the layering the originals lacked —
pure rules in the middle, I/O at the edge, everything injectable.

The lessons that survived the rewrite, each because it was paid for in
production and each now pinned by a test:

- **Durability must be reactive.** The first version depended on the agent
  updating its own checkpoint at milestones. It did not, reliably, and the
  misses clustered mid-session where the loss hurts most. Moving the trigger to
  the host's pre-compaction event is the single change that made the mechanism
  work.
- **The rejected option is the field that matters.** Recording "we chose X"
  leaves the next session free to propose Y. Recording "we chose X, Y was
  rejected" does not. This was discovered by watching settled decisions get
  relitigated after a compaction.
- **Never record the output of compaction as input to compaction.** The host
  reinjects its summary as a user-role message. Logging it produced a feedback
  loop: several hundred stacked copies of the same block and sessions that
  compacted every few minutes. The filter and its incident note are in
  `transcript.py`.
- **Two layers, not one.** A hot layer that is overwritten and a cold layer that
  is only appended to. Merging them means either losing history or never being
  able to say what is true *now*.
- **The distiller must degrade, never block.** The original called models
  synchronously inside the hook, with a timeout and a deterministic seed written
  first. The extraction keeps that shape and inverts the default: the pure
  heuristic is the baseline, and any model is an optional upgrade behind a
  fallback.
- **Fail soft at the edge.** A hook that raises costs the user the very context
  it exists to protect.

### What was deliberately left behind

Everything operational, private or vendor-specific. None of it is in this
repository and none of it informs its design:

- hardcoded personal paths, home directories and machine names;
- credentials, keyring lookups, API keys and tokens of any kind;
- a specific model-provider script and its named models — replaced by a
  `command` seam that takes argv you supply, so no vendor is named anywhere;
- chat/notification integrations used for operational alerting;
- the multi-model "distil twice and have a third model reconcile" arrangement,
  which was useful in a system with free inference capacity and is the wrong
  default for a library: it triples latency inside a blocking hook for a
  marginal quality gain. The `FallbackDistiller` seam makes it reconstructible
  in a few lines by anyone who wants it;
- all business-domain content, project names and operational context from the
  system the originals ran in.

## Development

Developed with substantial AI assistance. The architecture, the layering
decisions, the operational lessons encoded in the defensive rules, and the
judgements about what to extract and what to leave behind are the author's,
drawn from running the private original in production and from the failures it
produced along the way.

## Licence

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Eduardo de Souza Marques.
