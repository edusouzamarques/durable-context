"""Claims made outside the code still have to be true.

The README is the install instruction, PROVENANCE is the authorship record and
the classifiers are what a package index shows. Each of these has been wrong in
a way no amount of passing unit tests would have caught, so they are pinned
here like any other behaviour.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import durable_context

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    path = ROOT / name
    if not path.exists():  # pragma: no cover - running from an installed wheel
        pytest.skip(f"{name} is not shipped in this layout")
    return path.read_text(encoding="utf-8")


def test_the_package_ships_the_pep_561_marker():
    """The "Typing :: Typed" classifier is a promise to type checkers.

    Without ``py.typed`` next to the modules, mypy and Pyright report "missing
    library stubs or py.typed marker" and ignore every annotation in the
    package, while the index page advertises it as typed.
    """
    marker = Path(durable_context.__file__).resolve().parent / "py.typed"
    assert marker.exists()

    pyproject = _read("pyproject.toml")
    assert "Typing :: Typed" in pyproject
    assert "py.typed" in pyproject  # explicitly included in the built wheel


def test_the_install_instruction_is_one_that_works_today():
    """The first command in the README has to succeed for a first-time reader.

    ``pip install durable-context`` fails while the name is unregistered, and a
    reader who hits that on line one does not reach the rest of the document.
    """
    readme = _read("README.md")
    install = readme.split("## Install", 1)[1].split("##", 1)[0]
    assert "pip install git+https://github.com/edusouzaxGV/durable-context" in install
    assert "pip install durable-context" not in install


def test_the_provenance_window_is_not_in_the_future():
    """Provenance is read for credibility; a date that has not happened destroys it."""
    provenance = _read("PROVENANCE.md")
    copyright_years = {int(year) for year in re.findall(r"Copyright \(c\) (\d{4})", _read("LICENSE"))}
    assert copyright_years, "the licence should carry a copyright year"
    latest = max(copyright_years)
    claimed = {int(year) for year in re.findall(r"\b(20\d{2})\b", provenance)}
    assert claimed, "the provenance should date the work it describes"
    assert max(claimed) <= latest
