"""Merge rules for the hot layer.

The hot layer is overwritten, which means every write is a chance to lose
something. These two functions are where that is prevented, and they are pure
so the rules can be pinned by tests instead of by hope.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from .models import Checkpoint, Decision, normalise_target

__all__ = ["merge_decision", "merge_checkpoint"]


def _fill_gaps(previous: Decision, new: Decision) -> Decision:
    """Non-empty new fields win; empty new fields inherit the old ones."""
    merged: dict[str, str] = {}
    for name in Decision.CONTENT_FIELDS:
        new_value = (getattr(new, name) or "").strip()
        merged[name] = new_value or (getattr(previous, name) or "").strip()
    merged["source"] = new.source or previous.source
    return Decision(**merged)


def merge_decision(previous: Decision | None, new: Decision | None) -> Decision:
    """Combine the stored decision with a freshly distilled one.

    Four cases, in order, and the ordering is the whole design:

    **The update names no subject at all** (no target, no decision sentence -
    e.g. ``decision --why "..."`` on its own). That is a patch to the record in
    force, so every field it does not mention is inherited, ``discarded``
    included. Setting one field never blanks the others.

    **Same named target -> fill the gaps.** A later distillation that forgets
    *why* must not erase the why. This is what makes ``discarded`` sticky: once
    a path is rejected it stays rejected for as long as the decision is about
    the same thing.

    **Two different named targets -> the old target becomes the rejected path.**
    This is the expensive lesson. When work pivots from A to B and only "B" is
    recorded, the next session cheerfully re-proposes A and the pivot is
    relitigated. Recording "B, and A was rejected" ends that loop, and the pivot
    itself is the evidence - no model call required.

    **Otherwise the subject cannot be compared** - one side names a target and
    the other does not, or neither does - so the new record stands on its own.
    Nothing is inherited and, above all, nothing is invented. An empty target is
    not a target: it is the common case (most turns contain no decision cue at
    all), and treating two blanks as "the same subject" used to paste one
    decision's rejection onto an unrelated one, permanently. A false constraint
    that never expires is worse than a thinner record.
    """
    if new is None or new.is_empty():
        return previous or Decision()
    if previous is None or previous.is_empty():
        return new

    previous_key = normalise_target(previous.target)
    new_key = normalise_target(new.target)
    new_names_a_subject = bool((new.target or "").strip() or (new.decision or "").strip())

    if not new_names_a_subject:
        return _fill_gaps(previous, new)
    if previous_key and new_key and previous_key == new_key:
        return _fill_gaps(previous, new)
    if previous_key and new_key:
        discarded = (new.discarded or "").strip() or previous.target.strip()
        return replace(new, discarded=discarded)
    return new


def merge_checkpoint(
    previous: Checkpoint | None,
    updates: Checkpoint,
    *,
    allow_clear: Sequence[str] = (),
) -> Checkpoint:
    """Apply an update to the checkpoint without dropping untouched sections.

    An overwrite-in-place file invites partial writes: a caller updates NOW and
    silently destroys GOTCHAS, which is exactly the knowledge nobody will think
    to write down twice. Sections absent (or blank) in the update are carried
    forward; clearing one has to be asked for explicitly via ``allow_clear``.
    """
    if previous is None:
        base: dict[str, str] = {}
    else:
        base = dict(previous.sections)

    for name, body in updates.sections.items():
        if body.strip() or name in allow_clear:
            base[name] = body
        elif name not in base:
            base[name] = body

    for name in allow_clear:
        if name in updates.sections and not updates.sections[name].strip():
            base[name] = ""

    last_compact = updates.last_compact or (previous.last_compact if previous else None)
    trigger = updates.trigger or (previous.trigger if previous else "")
    return Checkpoint(sections=base, last_compact=last_compact, trigger=trigger)
