"""Uploaded documents must never be somewhere git can see them.

The upload directory is configurable, and today it is safe only because its
DEFAULT happens to sit under `data/`, which .gitignore covers. That is implicit
protection: point UPLOAD_DIR at any other directory inside the working tree and
every document a user uploads becomes a candidate for the next `git add`.

The invariant is therefore not "the default is right" but the weaker, checkable
one: the configured directory is either OUTSIDE the repository, or inside it
and ignored. Same for the other directories that receive document-derived
content, since a deleted .gitignore line is silent until the damage is done.
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Not read from pipeline.api.app: importing it creates directories and connects
# the configuration to a running application. The name of the variable is the
# contract here, not the module that reads it.
DOCUMENT_DIRS = (
    os.getenv("UPLOAD_DIR", "./data/uploads"),
    "data",
    "output",
    "logs",
    "uploads",  # the obvious value someone overrides UPLOAD_DIR with
)


def _is_ignored(path):
    probe = f"{path}/probe-belge.pdf"
    return subprocess.run(
        ["git", "check-ignore", "-q", probe],
        cwd=ROOT, capture_output=True,
    ).returncode == 0


def test_directories_holding_document_content_are_not_visible_to_git():
    exposed = []
    for setting in DOCUMENT_DIRS:
        path = Path(setting)
        absolute = path if path.is_absolute() else ROOT / path
        try:
            relative = absolute.resolve().relative_to(ROOT.resolve())
        except ValueError:
            continue  # outside the working tree: git cannot pick it up at all
        if not _is_ignored(relative.as_posix()):
            exposed.append(str(relative))

    assert exposed == [], (
        "belge iceren dizin git tarafindan goruluyor: " + ", ".join(exposed))
