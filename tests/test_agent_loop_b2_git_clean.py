"""PACKAGE B6-R1 -- git's clean conversion as an equality test.

WHAT THIS FILE IS ABOUT. Not "are these bytes equal" but "does this
working file still name that blob". A tracked file has two legitimate
working-tree representations and git's clean conversion maps either onto
the single object stored, so a checkout that is genuinely untouched can
differ from its own blob in every line ending. The shipped B5 run was
refused for exactly that, and the gate was right about the bytes and
wrong about the question.

REAL GIT, REAL REPOSITORIES. Every repository below is built with the
real `git` binary in `tmp_path`, because the whole subject is what git's
own conversion does -- a stand-in would only measure this file's opinion
of it. Nothing here touches the project's repository.

EVERY NEGATIVE TEST PROVES ITS INJECTION POINT WAS REACHED. The
filter-refusal tests are the sharpest case: a test that merely sees a
refusal proves nothing, because a refusal for an unrelated reason looks
identical. So the filter's own side effect is measured in BOTH
directions -- absent when the gate runs, present when the gate is
bypassed -- which is what makes "the process never started" a
measurement instead of a hope.
"""
from __future__ import annotations

import hashlib
import subprocess

import pytest

from tools.agent_loop import git_clean, git_transport

SENTINEL = "KURGU-GIZLI-ICERIK-" + "q" * 9
LF = b"satir1\nsatir2\nsatir3\n"
CRLF = b"satir1\r\nsatir2\r\nsatir3\r\n"


def _git(repo, *args, check=True):
    """The REAL subprocess module, for fixture setup only."""
    done = subprocess.run(["git", "-C", str(repo), *args],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check:
        assert done.returncode == 0, done.stderr[:400]
    return done


def _write(path, data: bytes):
    """Bytes, exactly. `write_text` would apply Python's own newline
    translation and the fixture would then be measuring that."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


def _blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0"
                        + data).hexdigest()


@pytest.fixture
def depo(tmp_path):
    """A repository whose blob is LF and whose checkout GIT ITSELF wrote
    under `core.autocrlf=true`.

    The checkout is materialised by `git checkout` rather than by this
    fixture, which matters: a hand-written CRLF file leaves a stale index
    stat entry and produces a DIFFERENT repository state -- one where
    `git status` reports a modification that `git diff` denies. The
    operator's real shape is the one git produced, so that is the one
    built here, and `git status` being empty is asserted rather than
    assumed."""
    repo = tmp_path / "kurgu-depo"
    repo.mkdir()
    for argv in (("init", "-q"), ("config", "core.autocrlf", "true"),
                 ("config", "user.email", "k@example.invalid"),
                 ("config", "user.name", "Kurgu")):
        _git(repo, *argv)
    _write(repo / "a.py", LF)
    _write(repo / "alt" / "b.py", LF)
    # committed with the conversion off, so the blobs really are LF
    _git(repo, "add", "-A")
    _git(repo, "-c", "core.autocrlf=false", "commit", "-qm", "kurgu-taban")
    for relative in ("a.py", "alt/b.py"):
        (repo / relative).unlink()
        _git(repo, "checkout", "--", relative)
    assert (repo / "a.py").read_bytes() == CRLF, \
        "senaryo kurulmadi: checkout CRLF degil"
    assert _git(repo, "status", "--porcelain").stdout == b"", \
        "senaryo kurulmadi: agac temiz degil"
    return repo


def _attributes(repo, text: bytes):
    _write(repo / ".gitattributes", text)


# =====================================================================
# THE EQUIVALENCE GIT'S OWN CLEAN DEFINES
# =====================================================================

def test_a_crlf_checkout_over_an_lf_blob_is_clean_equivalent(depo):
    """THE shipped failure, as an assertion. The raw bytes differ -- that
    is the whole point -- and the file has still not drifted."""
    assert (depo / "a.py").read_bytes() != LF, \
        "senaryo kurulmadi: ham baytlar ayni"
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True


def test_an_lf_checkout_is_clean_equivalent_too(depo):
    """The representation that needs no conversion must not be refused by
    the machinery that exists for the one that does."""
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=LF,
                                      baseline_bytes=LF) is True


def test_a_nested_path_is_asked_about_at_its_own_path(depo):
    assert git_clean.clean_equivalent(depo, relative="alt/b.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True


@pytest.mark.parametrize("degisen", [
    b"satir1\r\nSATIR2-DEGISTI\r\nsatir3\r\n",       # CRLF form
    b"satir1\nSATIR2-DEGISTI\nsatir3\n",             # LF form
    b"satir1\r\nsatir2\r\nsatir3\r\nEKLENEN\r\n",    # an added line
    b"satir1\r\nsatir3\r\n",                         # a removed line
])
def test_real_content_drift_is_never_clean_equivalent(depo, degisen):
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=degisen,
                                      baseline_bytes=LF) is False


def test_a_same_size_edit_is_not_clean_equivalent(depo):
    """An object id is not a stat. This is the class a size comparison
    misses and a restored timestamp hides."""
    degisen = b"satir1\r\nsatir9\r\nsatir3\r\n"
    assert len(degisen) == len(CRLF), "senaryo kurulmadi: boyutlar farkli"
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=degisen,
                                      baseline_bytes=LF) is False


def test_a_lone_carriage_return_is_not_a_line_ending_git_converts(depo):
    """The accepted equivalence class is CRLF, not "any carriage return".
    A lone CR survives the clean, so content differing by one is drift."""
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=b"satir1\rsatir2\r\n",
                                      baseline_bytes=LF) is False


def test_the_eol_attribute_is_honoured_rather_than_guessed(depo):
    """`text eol=crlf` is a different conversion from `core.autocrlf`, and
    git is the one that knows the difference."""
    _attributes(depo, b"*.py text eol=crlf\n")
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True


def test_a_binary_path_gets_no_conversion_at_all(depo):
    """`-text` means the raw bytes ARE the blob's bytes, so a CRLF file
    over an LF blob is genuine drift there. Proof that nothing in this
    module normalises line endings on its own."""
    _attributes(depo, b"*.py -text\n")
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is False
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=LF,
                                      baseline_bytes=LF) is True


def test_an_explicitly_unset_attribute_is_not_a_refusal(depo):
    """`unset` and `unspecified` both mean the conversion does not run.
    Refusing `-filter` would refuse a repository that had gone out of its
    way to be safe."""
    _attributes(depo, b"*.py -filter -ident\n")
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True


def test_a_hostile_gitattributes_cannot_widen_what_is_accepted(depo):
    """The attributes come from a file inside the checkout, so an attacker
    can choose them. MEASURED: with `filter`, `ident` and the encoding
    refused, every remaining variant leaves the CRLF family -- and none
    of them maps different content onto the baseline blob."""
    farkli = b"satir1\r\nsatir2\r\nsatir3\r\nEKLENEN-SATIR\r\n"
    for attributes in (b"", b"*.py text\n", b"*.py -text\n",
                       b"*.py text eol=crlf\n", b"*.py text eol=lf\n",
                       b"*.py text=auto\n", b"* text\n", b"a.py text\n"):
        _attributes(depo, attributes)
        assert git_clean.clean_equivalent(
            depo, relative="a.py", current_bytes=farkli,
            baseline_bytes=LF) is False, f"nitelikler kabul genisletti: " \
                                         f"{attributes!r}"


# =====================================================================
# THE CONVERSIONS THAT MUST NEVER RUN
# =====================================================================

def _filter_repo(depo, sentinel):
    """A repository whose `.gitattributes` names a clean filter that
    PROVES it ran by creating a file."""
    _attributes(depo, b"*.py filter=kurgu\n")
    _git(depo, "config", "filter.kurgu.clean",
         f"sh -c 'echo calisti > \"{sentinel.as_posix()}\"; cat'")


def test_a_custom_filter_is_refused_and_its_process_never_starts(depo,
                                                                 tmp_path):
    """THE order that makes the refusal worth anything.

    MEASURED in Task A and asserted in both directions here: the filter
    really does run when `hash-object --path=` is called, so a check made
    afterwards would be a check made after an arbitrary program of the
    repository's choosing had already executed. The second half of this
    test bypasses the gate on purpose -- if the sentinel did not appear
    there, the first half would be proving nothing at all."""
    nobetci = tmp_path / "FILTRE-CALISTI"
    _filter_repo(depo, nobetci)

    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    assert not nobetci.exists(), "filtre sureci basladi"
    assert str(ret.value) == "yol yerlesik olmayan bir donusum tasiyor"

    # THE LOAD-BEARING HALF: the sentinel mechanism really does fire
    git_clean.clean_object_id(depo, "a.py", CRLF)
    assert nobetci.exists(), \
        "senaryo kurulmadi: filtre kapi atlaninca da calismadi"


def test_the_attribute_gate_runs_before_the_conversion_is_asked_for(depo,
                                                                   tmp_path,
                                                                   monkeypatch):
    """The same ordering, pinned at the seam rather than at the side
    effect -- so a refactor that reorders the two calls fails here even on
    a machine where `sh` is missing."""
    _filter_repo(depo, tmp_path / "olmayacak")

    def asla(*args, **kwargs):
        raise AssertionError("donusum nitelik kapisindan once cagrildi")

    monkeypatch.setattr(git_clean, "clean_object_id", asla)
    with pytest.raises(git_clean.CleanConversionRefused):
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)


def test_an_ident_path_is_refused_because_it_hides_content(depo):
    """MEASURED: `ident` collapses `$Id: <anything>$` to `$Id$` on the way
    in, so bytes smuggled inside that span would clean to the baseline
    blob exactly. This is not caution -- it is a hole with a demo."""
    _attributes(depo, b"*.py ident\n")
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    assert str(ret.value) == "yol yerlesik olmayan bir donusum tasiyor"


def test_ident_really_can_collapse_different_bytes_onto_one_blob(tmp_path):
    """The demo the refusal above exists for, run against real git. If
    this ever stops holding, the refusal has become mere superstition and
    somebody should be told."""
    repo = tmp_path / "ident-depo"
    repo.mkdir()
    for argv in (("init", "-q"), ("config", "core.autocrlf", "false"),
                 ("config", "user.email", "k@example.invalid"),
                 ("config", "user.name", "Kurgu")):
        _git(repo, *argv)
    _write(repo / "f.py", b"$Id$\ngovde\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "kurgu")
    _attributes(repo, b"*.py ident\n")
    kacirilan = f"$Id: {SENTINEL}$\ngovde\n".encode("ascii")

    # asked WITHOUT the gate, which is the only way to see it happen
    assert git_clean.clean_object_id(repo, "f.py", kacirilan) == \
        _blob_oid(b"$Id$\ngovde\n")


@pytest.mark.parametrize("nitelik", [
    b"*.py working-tree-encoding=UTF-16\n",
    b"*.py working-tree-encoding=UTF-16LE\n",
    b"*.py working-tree-encoding=SHIFT-JIS\n",
])
def test_a_working_tree_encoding_is_refused_unread(depo, nitelik):
    """Refused rather than guessed at. Whether this build of git even
    supports the named encoding is not a question worth asking, because
    the answer cannot make oid equality mean what it needs to mean."""
    _attributes(depo, nitelik)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    assert str(ret.value) == "yol yerlesik olmayan bir donusum tasiyor"


def test_the_refused_attribute_set_is_exactly_the_three_measured_ones(depo):
    """A pin, so a fourth conversion class cannot be added to git and
    silently inherited as safe -- and so none of the three can be dropped
    without this file objecting."""
    assert git_clean.REFUSED_ATTRIBUTES == ("filter", "ident",
                                            "working-tree-encoding")


# =====================================================================
# FAIL-CLOSED WHEN THE ANSWER CANNOT BE PROVEN
# =====================================================================

def test_an_attribute_git_did_not_answer_for_is_not_treated_as_absent(
        depo, monkeypatch):
    """Silence is not absence. Treating a missing triple as "no attribute
    is set" is a fail-open hole shaped exactly like the silent-zero ones
    this project keeps finding."""
    eksik = b"a.py\0filter\0unspecified\0a.py\0ident\0unspecified\0"

    def kirpik(repo, *args, stdout_limit, stdin_bytes=None):
        if args and args[0] == "check-attr":
            return eksik
        raise AssertionError("beklenmeyen git cagrisi")

    monkeypatch.setattr(git_clean, "_run", kirpik)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.assert_builtin_conversion(depo, "a.py")
    assert str(ret.value) == "git nitelik cevabi kanitlanamadi"


def test_an_attribute_value_that_is_not_utf8_is_refused(depo, monkeypatch):
    bozuk = (b"a.py\0filter\0" + b"\xff\xfe" + b"\0"
             b"a.py\0ident\0unspecified\0"
             b"a.py\0working-tree-encoding\0unspecified\0")
    monkeypatch.setattr(git_clean, "_run",
                        lambda *a, **k: bozuk)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.assert_builtin_conversion(depo, "a.py")
    assert str(ret.value) == "git nitelik cevabi gecerli UTF-8 degil"


def test_a_git_call_that_fails_becomes_a_refusal_and_not_an_answer(tmp_path):
    """A directory that is not a repository at all: git exits nonzero and
    this layer must say "no answer", never `False` -- which the caller
    would read as drift and blame on the operator."""
    disari = tmp_path / "depo-degil"
    disari.mkdir()
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(disari, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    assert str(ret.value) == "git temizleme donusumu olculemedi"


@pytest.mark.parametrize("sentence", [
    "git komutu zaman asimina ugradi",
    "git ciktisi sozlesme tavanini asiyor",
    "git sureci temizlenemedi",
    "git girdisi tam yazilamadi",
    "git komutu basarisiz",
])
def test_every_transport_refusal_class_becomes_one_closed_refusal(depo,
                                                                 monkeypatch,
                                                                 sentence):
    """Timeout, output overflow, a surviving process tree, a truncated
    stdin and a nonzero exit all mean the same thing here: no answer was
    obtained. The transport's own battery is what tells them apart; this
    layer must not turn any of them into a verdict about the file."""
    def patlar(*args, **kwargs):
        raise git_transport.FlatWorkspaceError(sentence)

    monkeypatch.setattr(git_transport, "git_bytes", patlar)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    assert str(ret.value) == "git temizleme donusumu olculemedi"
    assert ret.value.__cause__ is None
    assert ret.value.__context__ is None, "ham hata baglam olarak tasindi"


def test_an_attributes_line_git_ignores_is_ignored_by_the_conversion_too(
        depo, tmp_path):
    """THE fail-open hole this had to be checked for, and it is not there.

    MEASURED: git refuses to parse an attributes line past roughly 2 KiB
    and answers `unspecified` for every attribute on it. Had the
    conversion still applied that line's filter, `check-attr` would have
    reported "no filter" while an arbitrary program ran anyway -- the gate
    would be fail-OPEN and this whole approach unusable. It does not: the
    filter's sentinel never appears and the resulting object id is exactly
    the RAW blob id, so both of git's answers agree.

    The control half proves the filter is live in the first place; without
    it a broken filter command would make this test pass for free."""
    nobetci = tmp_path / "UZUN-SATIR-FILTRESI"
    _filter_repo(depo, nobetci)
    # control: on a line git accepts, the filter runs and is refused here
    with pytest.raises(git_clean.CleanConversionRefused):
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=LF, baseline_bytes=LF)
    git_clean.clean_object_id(depo, "a.py", LF)
    assert nobetci.exists(), "senaryo kurulmadi: filtre hic calismiyor"
    nobetci.unlink()

    # the same filter, on a line long enough for git to drop
    _attributes(depo, b"*.py filter=kurgu dolgu=" + b"z" * 8192 + b"\n")
    reported = git_clean._attribute_values(depo, "a.py").get("filter")
    # THE INVARIANT, whichever way this build of git answers: the gate and
    # the conversion must AGREE. Written this way rather than pinning
    # git's own line limit, which is a number in git and not a promise to
    # us -- but either branch would be a defect if it did not hold.
    if reported in ("unspecified", "unset"):
        assert git_clean.clean_equivalent(depo, relative="a.py",
                                          current_bytes=LF,
                                          baseline_bytes=LF) is True
        assert git_clean.clean_object_id(depo, "a.py", LF) == _blob_oid(LF)
        assert not nobetci.exists(), \
            "git niteligi yok sayarken filtre yine calisti -- fail-OPEN"
    else:
        with pytest.raises(git_clean.CleanConversionRefused):
            git_clean.clean_equivalent(depo, relative="a.py",
                                       current_bytes=LF, baseline_bytes=LF)
        assert not nobetci.exists(), "reddedilirken filtre sureci basladi"


def test_the_attribute_reply_ceiling_bounds_a_reply_it_cannot_reach(depo):
    """MEASURED, and the reason there is no real-overflow test above it:
    a 1000-byte attribute value produces a 1075-byte reply, and anything
    longer makes git drop the line entirely -- so the VALUE channel tops
    out far below the ceiling. What can still approach it is the PATH,
    which the reply repeats once per attribute; a repo-relative path past
    roughly 1300 characters would overflow and be REFUSED, which is
    fail-closed. The ceiling's wiring is proven by the transport-refusal
    battery above, where the overflow sentence is one of the five.

    The value below is 200 bytes rather than the measured 1000: the point
    is only that a long value is STILL A VALUE, and 200 is under any
    git's attributes-line limit, so this does not become a test of which
    git is installed."""
    _attributes(depo, b"*.py filter=" + b"z" * 200 + b"\n")
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    # a value that long is still a value: refused as a non-builtin
    # conversion, not swallowed and not overflowed
    assert str(ret.value) == "yol yerlesik olmayan bir donusum tasiyor"
    assert git_clean._ATTR_CEILING == 4 << 10


@pytest.mark.parametrize("cevap", [b"", b"\n", b"kisa\n", b"z" * 40,
                                   b"FB70D2EE1611F46EEB71A149EFE7AF6358" +
                                   b"53A993",
                                   b"fb70d2ee1611f46eeb71a149efe7af635853a99"])
def test_an_object_id_that_is_not_a_full_lowercase_sha_is_refused(depo,
                                                                 monkeypatch,
                                                                 cevap):
    """Uppercase, short, empty or noise -- none of them is an object id,
    and comparing one to a real id would answer `False` and call it
    drift."""
    monkeypatch.setattr(git_clean, "_run", lambda *a, **k: cevap)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_object_id(depo, "a.py", CRLF)
    assert str(ret.value) == "git nesne kimligi tam SHA degil"


# =====================================================================
# THE INPUTS ARE THE CALLER'S, AND THEY ARE CHECKED
# =====================================================================

@pytest.mark.parametrize("yol", ["", None, 5, b"a.py", "/etc/passwd",
                                  "C:/mutlak/yol", "c:\\mutlak\\yol",
                                  "alt\\b.py", "a\0b.py"])
def test_a_path_that_is_not_repository_relative_is_refused(depo, yol):
    """Defence in depth -- the callers upstream canonicalise already --
    but the cheap kind: a NUL would truncate the `-z` stream the parser
    reads back, and an absolute path would ask git about a different file
    than the one whose bytes are in hand."""
    with pytest.raises(git_clean.CleanConversionRefused):
        git_clean.clean_equivalent(depo, relative=yol, current_bytes=CRLF,
                                   baseline_bytes=LF)


@pytest.mark.parametrize("icerik", ["metin", bytearray(b"ab"),
                                     memoryview(b"ab"), 5, None])
def test_content_that_is_not_bytes_is_refused(depo, icerik):
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=icerik, baseline_bytes=LF)
    assert "bayt dizisi degil" in str(ret.value)


@pytest.mark.parametrize("taban", ["metin", bytearray(b"ab"), 5, None])
def test_a_baseline_that_is_not_bytes_is_refused_before_any_git_call(
        depo, taban, monkeypatch):
    def asla(*args, **kwargs):
        raise AssertionError("gecersiz tabana ragmen git cagrildi")

    monkeypatch.setattr(git_clean, "_run", asla)
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=taban)
    assert str(ret.value) == "taban icerigi bayt dizisi degil"


def test_an_empty_file_is_a_legal_blob_and_not_a_special_case(depo):
    """MEASURED: the empty blob has a real object id. A guard that
    treated `b""` as "no content" would refuse an empty tracked file."""
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=b"",
                                      baseline_bytes=b"") is True
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=b"",
                                      baseline_bytes=LF) is False


# =====================================================================
# THE BYTES ARE THE CALLER'S, AND THEY DO NOT TRAVEL
# =====================================================================

def test_the_conversion_reads_the_callers_bytes_and_never_the_file(depo):
    """`--path` supplies attribute context and nothing else. MEASURED
    across four variants in Task A; asserted here against a file whose
    on-disk content is deliberately different from the payload."""
    _write(depo / "a.py", b"DISKTEKI-TAMAMEN-BASKA-ICERIK\n")
    assert git_clean.clean_object_id(depo, "a.py", LF) == _blob_oid(LF)
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True


def test_the_conversion_writes_no_object_into_the_repository(depo):
    """No `-w`. A verification step that grew the object store would be
    writing to the repository it is verifying."""
    oid = git_clean.clean_object_id(depo, "a.py", b"asla-yazilmamali\n")
    assert _git(depo, "cat-file", "-e", oid, check=False).returncode != 0


def test_the_conversion_writes_nothing_at_all_into_the_repository(depo):
    """The whole tree, before and after. Not just the object store: no
    index write, no config write, no working-tree byte."""
    def anlik():
        return {yol.relative_to(depo).as_posix(): yol.read_bytes()
                for yol in sorted(depo.rglob("*")) if yol.is_file()}

    once = anlik()
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=CRLF,
                                      baseline_bytes=LF) is True
    assert anlik() == once, "depo icinde bir sey degisti"


def test_no_content_byte_or_path_survives_a_refusal(depo, tmp_path):
    """The payload is a file of the operator's, and the refusal is the
    last place it may appear -- message, repr, notes, cause or context."""
    nobetci = tmp_path / "FILTRE"
    _filter_repo(depo, nobetci)
    gizli = f"# {SENTINEL}\n".encode("ascii")
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=gizli, baseline_bytes=LF)
    parcalar, dugum, derinlik = [], ret.value, 0
    while dugum is not None and derinlik < 10:
        parcalar.append(str(dugum))
        parcalar.append(repr(dugum))
        parcalar.extend(getattr(dugum, "__notes__", None) or [])
        dugum = dugum.__cause__ or dugum.__context__
        derinlik += 1
    metin = " ".join(parcalar)
    assert SENTINEL not in metin, "icerik disari cikti"
    assert str(tmp_path) not in metin and "a.py" not in metin, \
        "yol disari cikti"
    assert ret.value.__cause__ is None
    assert ret.value.__context__ is None


def test_the_refusal_carries_a_closed_contract_reason(depo):
    _attributes(depo, b"*.py ident\n")
    with pytest.raises(git_clean.CleanConversionRefused) as ret:
        git_clean.clean_equivalent(depo, relative="a.py",
                                   current_bytes=CRLF, baseline_bytes=LF)
    from tools.agent_loop import contract
    assert ret.value.reason in contract.ALL_STOP_REASONS


# =====================================================================
# WHAT IS NOT CONSULTED
# =====================================================================

def test_the_index_flags_that_blind_git_status_do_not_blind_this(depo):
    """MEASURED: `--skip-worktree` and `--assume-unchanged` make both
    `status` and `diff --quiet` report a clean tree over content that
    really changed. This layer never asks either of them, so a real edit
    is still a real edit."""
    degisen = b"satir1\r\nTAMAMEN-BASKA\r\n"
    _write(depo / "a.py", degisen)
    for flag in ("--skip-worktree", "--assume-unchanged"):
        _git(depo, "update-index", flag, "--", "a.py")
        assert _git(depo, "status", "--porcelain").stdout == b"", \
            f"senaryo kurulmadi: {flag} korluk yaratmadi"
        assert _git(depo, "diff", "--quiet", check=False).returncode == 0, \
            f"senaryo kurulmadi: {flag} diff'i de kormedi"
        assert git_clean.clean_equivalent(
            depo, relative="a.py", current_bytes=degisen,
            baseline_bytes=LF) is False
        _git(depo, "update-index", f"--no{flag[1:]}", "--", "a.py")


def test_staging_the_change_does_not_make_it_disappear(depo):
    """The index records what git was TOLD, not what is on disk. A drift
    that has been `git add`ed is still drift."""
    degisen = b"satir1\r\nSTAGED-BASKA\r\n"
    _write(depo / "a.py", degisen)
    _git(depo, "add", "--", "a.py")
    assert _git(depo, "diff", "--quiet", check=False).returncode == 0, \
        "senaryo kurulmadi: worktree diff temiz degil"
    assert git_clean.clean_equivalent(depo, relative="a.py",
                                      current_bytes=degisen,
                                      baseline_bytes=LF) is False


def test_this_module_never_asks_git_for_a_verdict(depo):
    """A pin on the SOURCE, because the temptation is permanent. `status`,
    `diff`, `ls-files` and `update-index` are answers about what git
    believes; the only two commands here ask for an attribute and an
    object id."""
    import inspect

    kaynak = inspect.getsource(git_clean)
    for yasak in ('"status"', '"diff"', '"ls-files"', '"update-index"',
                  '"rev-parse"', '"cat-file"', '"-w"'):
        assert yasak not in kaynak, f"otorite olmayan git cagrisi: {yasak}"
    assert '"check-attr"' in kaynak and '"hash-object"' in kaynak


def _identifiers(module):
    """Every name this module's CODE mentions, docstrings excluded.

    Read from the AST rather than the text on purpose: the first version
    of this pin searched raw source, and the module's own prose -- which
    explains why a git subprocess now exists -- tripped it. Prose that
    NAMES a hazard is not the hazard."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(module))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names, tree


def test_the_operators_bytes_are_never_written_anywhere():
    """They travel over a PIPE, to a process that cannot write them back.

    A temporary file holding a copy of the operator's source would be a
    second place their content exists, outside the holder discipline this
    package is built on -- and it would survive a crash. Pinned at the
    source, because the claim is that no such code path exists at all."""
    import ast

    names, tree = _identifiers(git_clean)
    imported = {alias.name.split(".")[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported == {"__future__", "re", "tools"}, \
        f"beklenmeyen ice aktarim: {sorted(imported)}"
    for yasak in ("open", "tempfile", "mkstemp", "NamedTemporaryFile",
                  "write_bytes", "write_text", "shutil", "mkdtemp"):
        assert yasak not in names, f"dosya yazma yolu: {yasak}"


def test_the_transport_is_the_only_way_this_module_starts_a_process():
    """No `subprocess`, no shell, no second launcher. The bounded,
    contained transport owns the timeout, the output ceiling, the checked
    return code and the proof that the container is empty -- and a module
    that reached around it would have none of them."""
    names, _tree = _identifiers(git_clean)
    for yasak in ("subprocess", "Popen", "system", "popen", "shell",
                  "spawnv", "execv", "run"):
        assert yasak not in names, f"tasima disi surec baslatma: {yasak}"
    # and the one call that IS allowed is really the one being used
    assert "git_bytes" in names and "git_transport" in names
