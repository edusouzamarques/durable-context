"""The filesystem edge, kept deliberately small.

Every interesting rule in this package is a pure function elsewhere. This module
exists so those functions never touch a path, and so the whole engine can be
driven in tests against :class:`MemoryStore` with no temporary directories and
no cleanup.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

__all__ = ["Store", "FileStore", "MemoryStore"]


@runtime_checkable
class Store(Protocol):
    """Minimal text persistence contract: read, overwrite, append, exists."""

    def read(self, path: Path) -> str: ...

    def read_tail(self, path: Path, max_bytes: int) -> str: ...

    def write(self, path: Path, text: str) -> None: ...

    def append(self, path: Path, text: str) -> None: ...

    def exists(self, path: Path) -> bool: ...


@dataclass(frozen=True)
class FileStore:
    """UTF-8 files, with atomic overwrite.

    The hot documents are rewritten on every compaction, and a compaction can be
    triggered by the host at a moment of its choosing. A non-atomic write that is
    interrupted leaves a half-written checkpoint, which is strictly worse than
    the previous one: it looks valid and reads wrong. So writes go to a temp file
    in the same directory and are moved into place with :func:`os.replace`, which
    is atomic on POSIX and on Windows.

    Reads of a missing file return ``""`` rather than raising. A first run has no
    state, and that is a normal condition, not an error.
    """

    encoding: str = "utf-8"

    def read(self, path: Path) -> str:
        try:
            return Path(path).read_text(encoding=self.encoding)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError):
            return ""

    def read_tail(self, path: Path, max_bytes: int) -> str:
        """Read at most the last ``max_bytes`` of a file.

        The cold log is append-only and never rotated, by design, so it grows
        without bound. Only its tail is ever injected, and the hook that injects
        it runs at session start, where the host is waiting on a subprocess.
        Reading the whole file to return twenty-five lines puts an unbounded
        cost in that path; this keeps it flat.

        A partial first line is dropped: the window starts mid-file and half a
        log entry is not a log entry.
        """
        target = Path(path)
        try:
            with open(target, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                size = stream.tell()
                start = max(0, size - max(0, max_bytes))
                stream.seek(start)
                chunk = stream.read()
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError):
            return ""
        text = chunk.decode(self.encoding, errors="replace")
        if start > 0:
            _, newline, rest = text.partition("\n")
            text = rest if newline else ""
        return text

    def write(self, path: Path, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(
            dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding=self.encoding, newline="\n") as stream:
                stream.write(text)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def append(self, path: Path, text: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding=self.encoding, newline="\n") as stream:
            stream.write(text)

    def exists(self, path: Path) -> bool:
        return Path(path).exists()


@dataclass
class MemoryStore:
    """In-memory store for tests and dry runs.

    Keys are stringified paths so behaviour matches :class:`FileStore` closely
    enough to be a useful stand-in without a real filesystem.
    """

    files: dict[str, str] = field(default_factory=dict)

    def read(self, path: Path) -> str:
        return self.files.get(str(path), "")

    def read_tail(self, path: Path, max_bytes: int) -> str:
        text = self.files.get(str(path), "")
        encoded = text.encode("utf-8")
        if len(encoded) <= max(0, max_bytes):
            return text
        clipped = encoded[len(encoded) - max(0, max_bytes) :].decode("utf-8", errors="replace")
        _, newline, rest = clipped.partition("\n")
        return rest if newline else ""

    def write(self, path: Path, text: str) -> None:
        self.files[str(path)] = text

    def append(self, path: Path, text: str) -> None:
        self.files[str(path)] = self.files.get(str(path), "") + text

    def exists(self, path: Path) -> bool:
        return str(path) in self.files
