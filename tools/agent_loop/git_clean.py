"""Git's own clean conversion, used as an equality test. PACKAGE B6-R1.

THE QUESTION. A tracked file has TWO legitimate byte representations in
a working tree -- git's clean conversion is what maps either of them onto
the one blob that is stored. So "has this file drifted from the baseline"
cannot be answered by comparing working-tree bytes to blob bytes: under
`core.autocrlf` a perfectly untouched checkout differs from its own blob
in every line ending, and refusing it calls the operator's clean tree
drift. MEASURED on this machine (git 2.51.0.windows.1): an LF blob
checked out under `core.autocrlf=true` is 24 bytes of CRLF on disk
against 21 bytes of LF in the object, and `git status` is empty.

WHAT IS ACTUALLY ASKED HERE, then, is one thing:

    does the checkout's file, after the CLEAN conversion git itself
    would apply at this path, still name the baseline's blob?

THIS IS NOT "IGNORE LINE ENDINGS". Nothing is normalised, nothing is
stripped and no comparison is loosened. The conversion is performed BY
GIT, at the exact path whose attributes govern it, and the result is
compared to an object id EXACTLY. Content that differs by so much as one
byte outside a line terminator produces a different id and is refused --
including content that kept the original size and had its timestamp put
back, because an object id is not a stat.

WHY `git status` IS NOT THE AUTHORITY, and not merely insufficient.
MEASURED, both directions: in the shipped run it answered "clean" for a
tree whose bytes the application gate then refused; and in a repository
whose index stat entry still describes the LF bytes it answers ` M` on
three consecutive calls while `git diff --quiet` returns 0 for the very
same tree. Two git views, one state, opposite answers. `--assume-unchanged`
and `--skip-worktree` were measured too: both make `status` and `diff`
report a clean tree over content that really did change. And the index,
the stage and HEAD describe what git was TOLD, not what is on disk. None
of them is evidence about bytes, so none of them is consulted.

THE TWO CONVERSIONS THAT MUST NEVER RUN, and why the refusal is not
caution but arithmetic:

  * `filter` -- an arbitrary program of the repository's choosing. Its
    output is whatever it likes, so oid equality would prove nothing at
    all. MEASURED: `hash-object --path=` really does execute it, so
    detecting it afterwards is too late; it is refused BEFORE the call
    that would start it.
  * `ident` -- collapses `$Id: <anything>$` to `$Id$` on the way in.
    MEASURED: a file carrying 24 smuggled bytes inside that span cleaned
    to the baseline blob exactly. An `ident` path could hide content
    from the comparison, so it is refused.
  * `working-tree-encoding` -- a whole-file recoding. Refused unread
    rather than guessed at.

WHAT REMAINS after those three are refused is the CRLF family alone --
`text`, `eol`, `core.autocrlf`, `core.eol` -- whose entire effect is
turning CRLF into LF. MEASURED against a hostile `.gitattributes` in six
variants (absent, `text`, `-text`, `text eol=crlf`, `text eol=lf`,
`text=auto`): not one of them mapped genuinely different content onto the
baseline blob. That is why this module asks git for the attributes
instead of parsing a `.gitattributes` an attacker could write -- the
answer it could tamper with cannot widen what gets accepted, only narrow
it, and the object id is still the gate.

WHERE THIS RUNS, and how that differs from `git_objects`. That module is
reachable only while a workspace is being CREATED, and says so. This one
is reached from the application's precondition, which means B6-R1
introduced a git subprocess into the verification surface for the first
time. It is bounded by the same transport, it runs no filter, it writes
nothing -- MEASURED: no `-w`, and the whole repository is byte-identical
across a call -- and it is consulted ONLY after raw byte equality has
already failed. But it is a subprocess where there used to be none, and
a future reader should know that was a deliberate trade and not an
oversight: the alternative was reimplementing git's clean conversion
here, which is the one thing guaranteed to disagree with git.

WHAT MAY LEAVE. Fixed sentences and closed reasons. Never git's stderr,
never a path, never a byte of anyone's file.
"""
from __future__ import annotations

import re

from tools.agent_loop import contract
from tools.agent_loop import git_objects
from tools.agent_loop import git_transport

# `check-attr -z` answers one NUL-terminated triple per attribute for one
# path; three short triples do not approach this.
_ATTR_CEILING = 4 << 10
# The whole answer is one 40-character object id and a newline. MEASURED:
# a 200,000-byte payload still produced exactly 41 bytes of stdout, so
# this ceiling bounds the reply rather than the question.
_OID_CEILING = 41

_OID = re.compile(r"^[0-9a-f]{40}$")

# The attributes whose conversions are not line endings. Every one of
# them is refused; see the module docstring for what each measured.
REFUSED_ATTRIBUTES = ("filter", "ident", "working-tree-encoding")
# `unspecified` -- never mentioned. `unset` -- explicitly turned off.
# Both mean the conversion does not run. ANYTHING ELSE is refused,
# including a value naming a filter that happens not to be configured:
# fail-closed does not depend on a second lookup agreeing.
_ABSENT = ("unspecified", "unset")


class CleanConversionRefused(RuntimeError):
    """The clean conversion could not be performed, or must not be.

    Fixed text and a closed reason -- never git's stderr, never a path.
    A caller that sees this has been told NOTHING about whether the file
    matches; it has been told the question cannot be answered safely,
    which is a refusal and not a mismatch."""

    def __init__(self, message, *,
                 reason=contract.StopReason.PATH_NOT_ALLOWED):
        super().__init__(message)
        self.reason = reason


def _relative(value):
    """A repository-relative POSIX path, or a refusal.

    The callers upstream have already canonicalised this, so the check is
    defence in depth -- but it is the cheap kind: a NUL would truncate
    the `check-attr -z` stream the parser below reads back, and an
    absolute path would ask git about a different file than the one whose
    bytes are in hand."""
    if type(value) is not str or not value:
        raise CleanConversionRefused("yol dizge degil")
    if "\0" in value or "\\" in value or value.startswith("/") or \
            re.match(r"^[A-Za-z]:", value):
        raise CleanConversionRefused("yol kanonik degil")
    return value


def _run(repo, *args, stdout_limit, stdin_bytes=None) -> bytes:
    """One contained git call, through the transport that already owns
    the timeout, the output ceiling, the checked return code and the
    proof that the container is empty.

    The refusal is BUILT inside the handler and RAISED outside it, so the
    lower layer's exception is not carried along as `__context__`."""
    refusal = None
    try:
        return git_transport.git_bytes(repo, *args, stdout_limit=stdout_limit,
                                      stdin_bytes=stdin_bytes)
    except git_transport.FlatWorkspaceError:
        # Timeout, output overflow, a surviving process tree, a nonzero
        # return code and a truncated stdin all arrive here, and all mean
        # the same thing to this layer: no answer was obtained. The
        # transport's own battery is what distinguishes them.
        refusal = CleanConversionRefused("git temizleme donusumu olculemedi")
    raise refusal


def _attribute_values(repo, relative):
    """What git says the three refused attributes are for this path.

    `-z` because an attribute value is arbitrary text and the
    newline-delimited form cannot be parsed safely. The reply is
    `<path>\\0<attribute>\\0<value>\\0` per attribute."""
    data = _run(repo, "check-attr", "-z", *REFUSED_ATTRIBUTES, "--",
                relative, stdout_limit=_ATTR_CEILING)
    fields = data.split(b"\0")
    values = {}
    for start in range(0, len(fields) - 2, 3):
        try:
            name = fields[start + 1].decode("utf-8")
            value = fields[start + 2].decode("utf-8")
        except UnicodeDecodeError:
            raise CleanConversionRefused("git nitelik cevabi gecerli UTF-8 "
                                         "degil") from None
        values[name] = value
    return values


def assert_builtin_conversion(repo, relative) -> None:
    """Refuse every conversion that is not line endings -- BEFORE one runs.

    THE ORDER IS THE WHOLE POINT. MEASURED: `hash-object --path=` starts
    a configured clean filter, so a check made after the conversion would
    be a check made after the program had already run. Nothing here runs
    a filter, and nothing downstream is reached unless this returns."""
    relative = _relative(relative)
    values = _attribute_values(repo, relative)
    for name in REFUSED_ATTRIBUTES:
        if name not in values:
            # git did not answer for this attribute. That is NOT the same
            # as "no attribute is set", and treating silence as absence is
            # how a fail-open hole gets built.
            raise CleanConversionRefused("git nitelik cevabi kanitlanamadi")
        if values[name] not in _ABSENT:
            raise CleanConversionRefused(
                "yol yerlesik olmayan bir donusum tasiyor")


def clean_object_id(repo, relative, data: bytes) -> str:
    """The object id `data` would have, cleaned AT `relative`.

    `--stdin` so the bytes hashed are the CALLER'S -- read earlier
    through the no-follow, handle-bound transport -- and never something
    git went and found on disk. `--path` supplies the attribute context
    and nothing else: MEASURED across four variants, including a path
    that does not exist and one that traverses upward, every call
    returned the id of the stdin bytes and the file sitting at that path
    was irrelevant. No `-w`, and MEASURED that the object really is not
    written: `cat-file -e` on the result fails.

    Nothing is written to a temporary file. The bytes go over a pipe to a
    process that cannot write them back."""
    relative = _relative(relative)
    if type(data) is not bytes:
        raise CleanConversionRefused("temizlenecek icerik bayt dizisi degil")
    out = _run(repo, "hash-object", "-t", "blob", f"--path={relative}",
               "--stdin", stdout_limit=_OID_CEILING, stdin_bytes=data)
    try:
        text = out.decode("ascii").strip()
    except UnicodeDecodeError:
        raise CleanConversionRefused("git nesne kimligi ASCII degil") from None
    if not _OID.match(text):
        raise CleanConversionRefused("git nesne kimligi tam SHA degil")
    return text


def clean_equivalent(repo, *, relative, current_bytes,
                     baseline_bytes) -> bool:
    """Does the checkout's file still name the baseline's blob?

    THE BASELINE OID IS DERIVED, NOT RECORDED. `baseline_bytes` are the
    materialised reference tree's, and `flat_workspace._materialise`
    refuses to write a blob whose recomputed id disagrees with the id the
    RAW TREE OBJECT carried -- so hashing those bytes here reproduces
    that same id with no second authority that could disagree with it,
    and no extra git call to be tampered with.

    Returns `True` only for the one equivalence git's own clean defines.
    `False` is a real mismatch. A refusal is neither, and is raised."""
    if type(baseline_bytes) is not bytes:
        raise CleanConversionRefused("taban icerigi bayt dizisi degil")
    assert_builtin_conversion(repo, relative)
    return clean_object_id(repo, relative, current_bytes) == \
        git_objects.blob_object_id(baseline_bytes)
