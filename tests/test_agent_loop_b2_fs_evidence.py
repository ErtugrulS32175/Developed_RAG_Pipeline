"""PACKAGE B2B-A-D2 -- the handle-bound filesystem evidence walker.

WHY THIS EXISTS AT ALL. Git was the evidence authority twice and was
defeated twice: first through the index (`skip-worktree` made a
modified control-plane file invisible), then through shared metadata (a
clean filter defined in `.git/config` hid a change AND ran a
model-supplied command during verification). Then the PATH was the
authority and was defeated three times, each time by moving one swap to
a different instant -- the last one landing between the post-listing
check and the child's own `lstat`, where both sides then agreed on the
outside file. So the authority is an OPEN DIRECTORY HANDLE: a handle
names an object, a path names a lookup.

NOTHING HERE RUNS GIT. No repository is created, no subprocess is
launched. Every scenario is a real directory tree in `tmp_path`.

TWO EVIDENCE CLASSES, and the difference is a measured cost decision:

  CONTENT   full streaming SHA-256 plus every metadata field.
  METADATA  every metadata field, and no right to read data.

THE METADATA INVARIANT, in the words this file tests. It is NOT "the
file is never opened" -- Windows has no `fstatat`, so a stat relative
to a directory handle IS a handle. What is guaranteed, and asserted
here, is that the access mask is exactly FILE_READ_ATTRIBUTES |
SYNCHRONIZE, that no data descriptor is ever made from it, and that no
data byte is ever read. Measured over the protected roots: 230,479
attribute opens, 399 data opens, and every byte read belongs to a
content-class file.

The metadata class is a NARROW claim and this file pins its edges: it
detects addition, deletion, rename, size change, mode change, file-id
change and timestamp change. A rewrite that preserves the entire tuple
is not detected, which is why the production seam waits out the
measured filesystem timestamp quantum before the model may run.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import stat
import time
import unicodedata
from pathlib import Path

import pytest

from tools.agent_loop import fs_evidence, fs_transport

SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
# one key for the whole file: before/after manifests must share
# it or their link fingerprints can never compare equal
KEY = b"K" * 32

# The protected roots as MEASURED on the reference machine. Every one of
# these is a real number this package has to keep admitting, so they are
# pinned here rather than re-derived by building a 184,685-file tree.
OLCULEN_GIRIS = 184_685
OLCULEN_TOPLAM = 19_951_871_042
OLCULEN_EN_BUYUK = 1_659_834_880


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _tree(root: Path):
    """A small tree with the shapes the walker has to distinguish.

    `write_bytes` throughout, never `write_text`: on Windows the text
    form translates the newline, so a scenario built on "exactly the
    same size" is silently built on two different sizes -- which is how
    three setup assertions in this file first fired."""
    (root / "pipeline").mkdir(parents=True)
    (root / "pipeline" / "a.py").write_bytes(b"AAAA\n")
    (root / "pipeline" / "b.py").write_bytes(b"BBBB\n")
    (root / "bos-dizin").mkdir()
    (root / "sahte_env" / "Lib").mkdir(parents=True)
    (root / "sahte_env" / "Lib" / "paket.py").write_bytes(b"PP\n")
    (root / "sahte_env" / "pyvenv.cfg").write_bytes(b"home = X\n")
    return root


def _scan(root, **kwargs):
    kwargs.setdefault("key", KEY)
    return fs_evidence.scan(root, **kwargs)


def _changed(before, after):
    return set(fs_evidence.diff(before, after))


def _quiesce():
    """The measured wait. 1 ms already gave 200/200 distinct timestamps
    on this filesystem; the production constant is ten times that."""
    time.sleep(fs_evidence.QUIESCENCE_NS / 1e9)


@pytest.fixture
def tree(tmp_path):
    return _tree(tmp_path / "kok")


# ---------------------------------------------------------------------
# THE CLAIM THE WHOLE PACKAGE RESTS ON
# ---------------------------------------------------------------------

def test_an_untouched_tree_produces_an_identical_manifest(tree):
    """No false positives, or every refusal below is noise."""
    before = _scan(tree)
    _quiesce()
    after = _scan(tree)
    assert before.digest == after.digest
    assert _changed(before, after) == set()


def test_the_manifest_digest_is_order_independent_and_stable(tree):
    """A walker whose answer depends on directory iteration order gives
    a different verdict on the same tree."""
    first = _scan(tree)
    second = _scan(tree)
    assert first.digest == second.digest
    assert len(first.digest) == 64
    assert [entry.path for entry in first.entries] == sorted(
        entry.path for entry in first.entries)


def test_a_same_size_rewrite_after_quiescence_is_detected(tree):
    """THE metadata-class claim, in the shape the production seam
    guarantees it: the manifest is taken, the measured quantum is
    waited out, and only then may anything write."""
    before = _scan(tree, metadata_only=("sahte_env",))
    _quiesce()
    target = tree / "sahte_env" / "Lib" / "paket.py"
    onceki = os.lstat(target)
    target.write_bytes(b"QQ\n")                 # exactly the same size
    assert os.lstat(target).st_size == onceki.st_size, "senaryo kurulmadi"
    after = _scan(tree, metadata_only=("sahte_env",))
    assert "sahte_env/Lib/paket.py" in _changed(before, after)


def test_an_atomic_replace_is_detected_by_file_identity(tree):
    """A careful writer replaces rather than rewrites. The timestamp
    moves, and so does the file id -- two independent reasons."""
    before = _scan(tree, metadata_only=("sahte_env",))
    _quiesce()
    target = tree / "sahte_env" / "Lib" / "paket.py"
    onceki = os.lstat(target)
    gecici = target.with_suffix(".tmp")
    gecici.write_bytes(b"RR\n")
    os.replace(gecici, target)
    sonraki = os.lstat(target)
    assert onceki.st_size == sonraki.st_size, "senaryo kurulmadi"
    after = _scan(tree, metadata_only=("sahte_env",))
    assert "sahte_env/Lib/paket.py" in _changed(before, after)


def test_content_class_catches_what_metadata_alone_would_miss(tree):
    """The two classes are not interchangeable, and the test says so by
    forging the whole metadata tuple: same size, and the timestamp put
    back where it was. Only the content digest still differs."""
    target = tree / "pipeline" / "a.py"
    onceki = os.lstat(target)
    before = _scan(tree)
    target.write_bytes(b"ZZZZ\n")               # same size
    os.utime(target, ns=(onceki.st_atime_ns, onceki.st_mtime_ns))
    sonraki = os.lstat(target)
    assert (onceki.st_size, onceki.st_mtime_ns) == (sonraki.st_size,
                                                    sonraki.st_mtime_ns), \
        "senaryo kurulmadi: metadata geri alinamadi"
    after = _scan(tree)
    assert "pipeline/a.py" in _changed(before, after), \
        "icerik sinifi ayni metadata altinda degisikligi kaciriyor"


def test_the_metadata_class_records_no_content_digest(tree):
    """The cost decision, made observable: a metadata-class entry must
    not be secretly hashing 19 GB."""
    manifest = _scan(tree, metadata_only=("sahte_env",))
    by_path = {entry.path: entry for entry in manifest.entries}
    assert by_path["sahte_env/Lib/paket.py"].sha256 == ""
    assert by_path["pipeline/a.py"].sha256 != ""
    assert manifest.metadata_only >= 2 and manifest.content_hashed >= 2


# ---------------------------------------------------------------------
# ORDINARY CHANGES
# ---------------------------------------------------------------------

@pytest.mark.parametrize("islem", ["ekle", "sil", "degistir", "yeniden-adlandir"])
def test_ordinary_changes_are_detected(tree, islem):
    before = _scan(tree)
    _quiesce()
    if islem == "ekle":
        (tree / "pipeline" / "yeni.py").write_text("Y\n", encoding="utf-8")
        beklenen = "pipeline/yeni.py"
    elif islem == "sil":
        (tree / "pipeline" / "b.py").unlink()
        beklenen = "pipeline/b.py"
    elif islem == "degistir":
        (tree / "pipeline" / "a.py").write_text("COK DAHA UZUN\n",
                                                encoding="utf-8")
        beklenen = "pipeline/a.py"
    else:
        (tree / "pipeline" / "b.py").rename(tree / "pipeline" / "c.py")
        beklenen = "pipeline/c.py"
    after = _scan(tree)
    assert beklenen in _changed(before, after)


def test_an_empty_directory_is_part_of_the_inventory(tree):
    """Explicitly defined rather than left to chance: an added or
    removed empty directory is a change."""
    before = _scan(tree)
    _quiesce()
    (tree / "bos-dizin").rmdir()
    (tree / "yeni-bos").mkdir()
    after = _scan(tree)
    degisen = _changed(before, after)
    assert "bos-dizin" in degisen and "yeni-bos" in degisen


def test_an_executable_bit_change_is_detected(tree):
    if os.name == "nt":
        pytest.skip("calistirilabilir biti POSIX'e ozgu")
    target = tree / "pipeline" / "a.py"
    before = _scan(tree)
    _quiesce()
    os.chmod(target, os.lstat(target).st_mode | stat.S_IXUSR)
    after = _scan(tree)
    assert "pipeline/a.py" in _changed(before, after)


# ---------------------------------------------------------------------
# FILE TYPES -- no-follow, fail-closed
# ---------------------------------------------------------------------


def test_a_symlink_is_fingerprinted_without_being_followed(tmp_path, tree):
    """Both platforms now describe a link by a keyed code taken from
    the link's OWN data -- `readlink` relative to the parent descriptor
    on POSIX, FSCTL_GET_REPARSE_POINT on a handle-bound, identity-bound
    reparse point on Windows. Neither follows it."""
    disarisi = tmp_path / "disarida.txt"
    disarisi.write_text(SENTINEL, encoding="utf-8")
    link = tree / "pipeline" / "baglanti.py"
    try:
        os.symlink(str(disarisi), str(link))
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    manifest = _scan(tree)
    entry = {e.path: e for e in manifest.entries}["pipeline/baglanti.py"]
    assert entry.kind == "link"
    assert entry.sha256 == "", "baglanti takip edilip icerigi okundu"
    assert SENTINEL not in str(entry), "hedef icerigi manifeste sizdi"
    assert len(entry.link_target_mac) == 64, "baglanti hedefi parmak izlenmedi"
    assert manifest.reparse_count >= 1


def test_a_link_is_never_descended_into(tmp_path, tree):
    """Fingerprinted is not the same as followed. Whatever the tag, the
    thing it points at must not enter the inventory."""
    disari = tmp_path / "disarida"
    (disari / "derin").mkdir(parents=True)
    (disari / "derin" / "gizli.py").write_text(SENTINEL, encoding="utf-8")
    try:
        os.symlink(str(disari), str(tree / "pipeline" / "baglanti"),
                   target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    manifest = _scan(tree)
    yollar = {e.path for e in manifest.entries}
    assert "pipeline/baglanti" in yollar
    assert not [y for y in yollar if y.startswith("pipeline/baglanti/")], \
        "baglantinin arkasi envantere girdi"
    assert SENTINEL not in " ".join(f"{e.path}{e.link_target_mac}{e.sha256}"
                                    for e in manifest.entries)


def test_a_retargeted_symlink_is_detected(tmp_path, tree):
    ilk = tmp_path / "ilk.txt"
    ikinci = tmp_path / "ikinci-farkli-ad.txt"
    ilk.write_text("A", encoding="utf-8")
    ikinci.write_text("B", encoding="utf-8")
    link = tree / "pipeline" / "baglanti.py"
    try:
        os.symlink(str(ilk), str(link))
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    before = _scan(tree)
    _quiesce()
    link.unlink()
    os.symlink(str(ikinci), str(link))
    after = _scan(tree)
    assert "pipeline/baglanti.py" in _changed(before, after)


def test_a_junction_is_fingerprinted_and_not_followed(tmp_path, tree):
    """A junction is a reparse point with a different tag, and it gets
    exactly the same treatment: a keyed code, and no descent."""
    if os.name != "nt":
        pytest.skip("kavsak noktasi Windows'a ozgu")
    hedef = tmp_path / "kavsak-hedefi"
    hedef.mkdir()
    (hedef / "icerde.txt").write_text(SENTINEL, encoding="utf-8")
    import _winapi

    _winapi.CreateJunction(str(hedef), str(tree / "pipeline" / "kavsak"))
    manifest = _scan(tree)
    kayit = {e.path: e for e in manifest.entries}["pipeline/kavsak"]
    yollar = {e.path for e in manifest.entries}
    assert kayit.kind == "link" and kayit.reparse_tag
    assert len(kayit.link_target_mac) == 64
    assert "pipeline/kavsak/icerde.txt" not in yollar, \
        "kavsak takip edildi ve disaridaki icerik envantere girdi"
    assert SENTINEL not in str(kayit)


def test_a_special_file_is_refused(tree):
    if not hasattr(os, "mkfifo"):
        pytest.skip("bu platformda FIFO yaratilamiyor")
    os.mkfifo(str(tree / "pipeline" / "boru"))
    with pytest.raises(fs_evidence.UnsupportedEntry):
        _scan(tree)


def test_a_hardlinked_file_is_refused_when_it_cannot_be_represented(tree):
    hedef = tree / "pipeline" / "b.py"
    link = tree / "pipeline" / "sert-baglanti.py"
    try:
        os.link(str(hedef), str(link))
    except OSError:
        pytest.skip("bu ortamda sert baglanti kurulamiyor")
    assert os.lstat(link).st_nlink > 1, "senaryo kurulmadi"
    with pytest.raises(fs_evidence.UnsupportedEntry):
        _scan(tree)


# ---------------------------------------------------------------------
# TIMESTAMP SANITY
# ---------------------------------------------------------------------

def test_a_future_timestamp_fails_closed(tree):
    """A newly written file's timestamp was measured to sit slightly
    BEHIND the clock, never ahead. One in the future is a tree this
    walker cannot reason about."""
    target = tree / "pipeline" / "a.py"
    ileri = time.time_ns() + 3_600_000_000_000        # bir saat ileri
    os.utime(target, ns=(ileri, ileri))
    assert os.lstat(target).st_mtime_ns > time.time_ns(), "senaryo kurulmadi"
    with pytest.raises(fs_evidence.EvidenceError):
        _scan(tree)


# ---------------------------------------------------------------------
# PATHS AND LIMITS
# ---------------------------------------------------------------------

def test_paths_are_canonical_relative_posix(tree):
    manifest = _scan(tree)
    for entry in manifest.entries:
        assert not entry.path.startswith("/")
        assert "\\" not in entry.path
        assert ".." not in entry.path.split("/")
        assert "" not in entry.path.split("/")
        assert all(ord(ch) >= 32 for ch in entry.path)


def test_an_invisible_or_control_character_in_a_name_is_refused(tree):
    """Cc/Cf and the invisible separators: a name nobody can read in a
    report is a name nobody can review."""
    try:
        # chr(), not the character itself: a test whose whole subject is
        # a name nobody can see must not be invisible in its own source
        (tree / "pipeline" / f"gorunmez{chr(0x200B)}.py").write_bytes(b"X")
    except OSError:
        pytest.skip("bu dosya sistemi bu adi kabul etmiyor")
    with pytest.raises(fs_evidence.UnsupportedEntry):
        _scan(tree)


@pytest.mark.parametrize("sinir", ["dosya-sayisi", "tek-dosya", "toplam"])
def test_the_measured_limits_fail_closed(tree, sinir):
    """Bounds derived from a measurement (162 tracked files, 1.8 MB,
    largest 0.1 MB), not from taste -- and refused rather than
    truncated."""
    if sinir == "dosya-sayisi":
        limits = fs_evidence.Limits(max_entries=2)
    elif sinir == "tek-dosya":
        limits = fs_evidence.Limits(max_content_file_bytes=1)
    else:
        limits = fs_evidence.Limits(max_logical_total_bytes=1)
    with pytest.raises(fs_evidence.EvidenceError):
        _scan(tree, limits=limits)



def test_an_unreadable_file_is_typed_and_carries_no_path(tree, monkeypatch):
    """A raw `OSError` is not an acceptable outcome, and this test used
    to say it was.

    It accepted `(EvidenceError, OSError)` and then ran its leak checks
    only in the typed branch -- so when the walker let the operating
    system's own message through, with an absolute path in it, the test
    stayed green and checked nothing. Measured: all seven transport
    seams leaked that way."""
    gercek = fs_transport.open_child_file

    def patlayan(directory, record):
        if record.name == "a.py":
            raise OSError(f"{SENTINEL} {tree}")
        return gercek(directory, record)

    monkeypatch.setattr(fs_transport, "open_child_file", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as refusal:
        _scan(tree)
    metin = str(refusal.value) + repr(refusal.value)
    assert SENTINEL not in metin
    assert str(tree) not in metin
    assert "/" not in metin and "\\" not in metin


def test_no_manifest_or_error_carries_absolute_paths_or_content(tree):
    (tree / "pipeline" / "a.py").write_text(SENTINEL, encoding="utf-8")
    manifest = _scan(tree)
    tasinan = " ".join(
        f"{e.path}{e.kind}{e.mode}{e.sha256}{e.link_target_mac}"
        for e in manifest.entries)
    assert SENTINEL not in tasinan, "manifest dosya icerigi tasiyor"
    assert str(tree) not in tasinan, "manifest mutlak yol tasiyor"


def test_the_manifest_reports_its_own_cost(tree):
    """The combined scan time is a number this package has to publish,
    because the protected roots were measured at roughly 50 seconds a
    pass and that is a budget somebody has to see."""
    manifest = _scan(tree)
    assert isinstance(manifest.duration_ms, int)
    assert manifest.file_count >= 4 and manifest.total_bytes > 0


# =====================================================================
# THE SEAM BETWEEN LISTING AND OPENING
#
# `O_NOFOLLOW` and `FILE_FLAG_OPEN_REPARSE_POINT` constrain the FINAL
# component of a path and nothing else. Every test below injects its
# swap at the exact instant the walker crosses that seam, and each one
# proves its own injection landed -- a scenario that silently failed to
# arm is a green test measuring nothing, which this package has already
# been caught by twice.
# =====================================================================

@pytest.fixture
def tek_kok(tmp_path):
    """One content-class file, so an injection keyed on its name is
    deterministic rather than dependent on directory order."""
    kok = tmp_path / "kok"
    (kok / "pipeline").mkdir(parents=True)
    (kok / "pipeline" / "a.py").write_bytes(b"ICERIDE\n")
    return kok


def _baglanti_kurulabiliyor(tmp_path) -> bool:
    deneme = tmp_path / "baglanti-denemesi"
    hedef = tmp_path / "baglanti-hedefi"
    hedef.write_bytes(b"X")
    try:
        os.symlink(str(hedef), str(deneme))
    except OSError:
        return False
    deneme.unlink()
    return True



def test_a_file_swapped_for_a_symlink_at_the_open_is_refused(
        tmp_path, tek_kok, monkeypatch):
    """The original P0: a plain file replaced by a link to somewhere
    else, at the instant the walker opens it."""
    if not _baglanti_kurulabiliyor(tmp_path):
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    disarida = tmp_path / "disarida.txt"
    disarida.write_bytes(SENTINEL.encode("ascii"))
    hedef = tek_kok / "pipeline" / "a.py"

    saglam = {e.path: e for e in _scan(tek_kok).entries}
    assert saglam["pipeline/a.py"].sha256, "olumlu kontrol kurulmadi"

    gercek = fs_transport.open_child_file
    durum = {"cagrildi": False}

    def yaris(directory, record):
        if record.name == "a.py" and not durum["cagrildi"]:
            durum["cagrildi"] = True
            hedef.unlink()
            os.symlink(str(disarida), str(hedef))
        return gercek(directory, record)

    monkeypatch.setattr(fs_transport, "open_child_file", yaris)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tek_kok)

    assert durum["cagrildi"], "saldiri geri cagrimina hic ulasilmadi"
    assert os.path.islink(hedef), "senaryo kurulmadi: dosya baglantiya donmedi"
    assert SENTINEL not in str(ret.value) + repr(ret.value)



def test_a_parent_directory_swapped_for_a_junction_is_refused(
        tmp_path, tek_kok, monkeypatch):
    """The escape one level up, which no flag on the child's own open
    can see. The handle-bound walk never asks the path again, so the
    swapped name is not consulted at all."""
    disari = _disari_kur(tmp_path)
    ebeveyn = tek_kok / "pipeline"

    saglam = {e.path: e for e in _scan(tek_kok).entries}
    assert saglam["pipeline/a.py"].sha256, "olumlu kontrol kurulmadi"

    gercek = fs_transport.open_child_file
    durum = {"cagrildi": False, "kuruldu": False}

    def yaris(directory, record):
        if record.name == "a.py" and not durum["cagrildi"]:
            durum["cagrildi"] = True
            durum["kuruldu"] = _ebeveyni_takas_et(ebeveyn, disari)
        return gercek(directory, record)

    monkeypatch.setattr(fs_transport, "open_child_file", yaris)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tek_kok)

    assert durum["cagrildi"], "saldiri geri cagrimina hic ulasilmadi"
    assert durum["kuruldu"], "senaryo kurulmadi: takas yapilamadi"
    assert os.stat(ebeveyn / "a.py").st_size == len(SENTINEL), \
        "senaryo kurulmadi: yol disaridaki dosyaya cozulmuyor"
    assert SENTINEL not in str(ret.value) + repr(ret.value)



def test_a_directory_swapped_for_a_junction_before_listing_is_refused(
        tmp_path, tek_kok, monkeypatch):
    """The swap at the RECURSION seam, between a directory being
    recorded and being descended into."""
    disari = tmp_path / "disarida"
    disari.mkdir()
    (disari / "gizli.py").write_bytes(SENTINEL.encode("ascii"))
    ebeveyn = tek_kok / "pipeline"

    gercek = fs_transport.open_child_directory
    durum = {"cagrildi": False, "kuruldu": False}

    def yaris(directory, record):
        if record.name == "pipeline" and not durum["cagrildi"]:
            durum["cagrildi"] = True
            durum["kuruldu"] = _ebeveyni_takas_et(ebeveyn, disari)
        return gercek(directory, record)

    monkeypatch.setattr(fs_transport, "open_child_directory", yaris)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tek_kok)

    assert durum["cagrildi"], "saldiri geri cagrimina hic ulasilmadi"
    assert durum["kuruldu"], "senaryo kurulmadi: kavsak yaratilamadi"
    assert (disari / "gizli.py").exists(), "senaryo kurulmadi"
    assert SENTINEL not in str(ret.value) + repr(ret.value)



def test_a_file_that_changes_during_its_own_read_is_refused(
        tek_kok, monkeypatch):
    """A digest taken across a change describes neither version. The
    refusal is driven by the two descriptor identities disagreeing, so
    the test captures both and asserts they moved."""
    hedef = tek_kok / "pipeline" / "a.py"
    hedef.write_bytes(b"A" * (4 << 20))          # birden fazla okuma turu
    once = os.lstat(hedef)

    gercek = fs_transport.descriptor_identity
    kayitlar = []
    durum = {"enjekte": False}

    def izleyen(descriptor):
        kimlik = gercek(descriptor)
        kayitlar.append(kimlik)
        if not durum["enjekte"]:
            durum["enjekte"] = True
            with open(hedef, "r+b") as akis:
                akis.seek(0, os.SEEK_END)
                akis.write(b"B" * 4096)          # boyut kesin degisir
        return kimlik

    monkeypatch.setattr(fs_transport, "descriptor_identity", izleyen)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tek_kok)

    sonra = os.lstat(hedef)
    assert durum["enjekte"], "enjeksiyon hic calismadi"
    assert sonra.st_size == once.st_size + 4096, \
        "senaryo kurulmadi: dosya gercekten degismedi"
    assert len(kayitlar) >= 2, "iki tanimlayici kimligi alinmadi"
    assert kayitlar[0] != kayitlar[-1], \
        "senaryo kurulmadi: handle kimlikleri ayni kaldi"
    assert type(ret.value) is fs_evidence.EvidenceError, \
        "okuma sirasi degisim yol reddi olarak siniflandirildi"
    assert "okunurken" in str(ret.value)


# =====================================================================
# D2: THE PATH IS NOT THE AUTHORITY
#
# Three rounds of this package tried to keep a PATH honest -- no-follow
# on the final component, then an identity pin against the child's own
# `lstat`, then identity checks on both sides of the listing. Each one
# was defeated by moving the swap to a different instant, because every
# check was itself a fresh question asked of a path.
#
# The counter-example that ended it: swap the parent AFTER the
# post-listing check and BEFORE the child's `lstat`. The child's
# `lstat` and the descriptor then agree -- on the outside file.
#
#     race_fired=True  accepted=True  outside_hashed=True
#
# So the authority is an OPEN DIRECTORY HANDLE. A handle names an
# object; a path names a lookup. The tests below are written against
# that: the tree a scan describes is the tree of the handle it took,
# and nothing done to the path afterwards changes the answer.
# =====================================================================

def _disari_kur(tmp_path, ad="a.py", icerik=None):
    """An outside directory holding a same-named file, which is what
    makes the swap invisible to any name-based check."""
    disari = tmp_path / "disarida"
    disari.mkdir(exist_ok=True)
    (disari / ad).write_bytes(icerik or SENTINEL.encode("ascii"))
    return disari


def _ebeveyni_takas_et(ebeveyn: Path, disari: Path) -> bool:
    """Replace a directory PATH with a link to somewhere else, leaving
    the name identical. Returns whether the swap actually landed --
    every caller asserts on it, because a scenario that failed to arm
    is a green test measuring nothing."""
    try:
        for cocuk in ebeveyn.iterdir():
            cocuk.unlink()
        ebeveyn.rmdir()
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(disari), str(ebeveyn))
        else:
            os.symlink(str(disari), str(ebeveyn))
        return True
    except OSError:
        return False


def test_a_parent_swapped_after_listing_before_child_stat_is_refused(
        tmp_path, tek_kok, monkeypatch):
    """THE counter-example that ended the path design, in the window it
    used: after the listing, before the child is committed to.

    Measured against the path walker: race_fired=True, accepted=True,
    outside_hashed=True. Here the child is opened relative to the
    directory OBJECT that listed it, and that object no longer has the
    name -- so the walk fails closed instead of following it."""
    disari = _disari_kur(tmp_path)
    ebeveyn = tek_kok / "pipeline"
    durum = {"atesledi": False, "kuruldu": False}

    saglam = {e.path: e for e in _scan(tek_kok).entries}
    assert saglam["pipeline/a.py"].sha256, "olumlu kontrol kurulmadi"

    gercek = fs_transport.list_directory

    def yaris(directory):
        kayitlar = gercek(directory)
        if not durum["atesledi"] and any(k.name == "a.py" for k in kayitlar):
            durum["atesledi"] = True
            durum["kuruldu"] = _ebeveyni_takas_et(ebeveyn, disari)
        return kayitlar

    monkeypatch.setattr(fs_transport, "list_directory", yaris)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tek_kok)

    assert durum["atesledi"], "saldiri geri cagrimina hic ulasilmadi"
    assert durum["kuruldu"], "senaryo kurulmadi: takas yapilamadi"
    assert os.stat(ebeveyn / "a.py").st_size == len(SENTINEL), \
        "senaryo kurulmadi: yol disaridaki dosyaya cozulmuyor"
    assert SENTINEL not in str(ret.value) + repr(ret.value)


def test_the_same_race_is_refused_for_a_metadata_only_child(
        tmp_path, tek_kok, monkeypatch):
    """The metadata class never opens a file, so it cannot be defended
    by anything about opening -- and it does not need to be. Its record
    came from the directory object before the swap existed, so the
    outside file's size is simply never seen.

    Measured against the path walker: the outside SIZE was accepted
    under the inside path, which is a wrong verdict with no content in
    it at all."""
    disari = _disari_kur(tmp_path)
    ebeveyn = tek_kok / "pipeline"
    durum = {"atesledi": False, "kuruldu": False}
    gercek = fs_transport.list_directory

    def yaris(directory):
        kayitlar = gercek(directory)
        if not durum["atesledi"] and any(k.name == "a.py" for k in kayitlar):
            durum["atesledi"] = True
            durum["kuruldu"] = _ebeveyni_takas_et(ebeveyn, disari)
        return kayitlar

    monkeypatch.setattr(fs_transport, "list_directory", yaris)
    manifest = _scan(tek_kok, metadata_only=("pipeline",))

    assert durum["atesledi"] and durum["kuruldu"], "senaryo kurulmadi"
    assert os.stat(ebeveyn / "a.py").st_size == len(SENTINEL), \
        "senaryo kurulmadi: yol disaridaki dosyaya cozulmuyor"
    kayit = {e.path: e for e in manifest.entries}["pipeline/a.py"]
    assert kayit.size == len(b"ICERIDE\n"), \
        "disaridaki dosyanin metadata'si iceridekinin yerine gecti"
    assert kayit.size != len(SENTINEL)


def test_not_one_byte_is_read_when_the_object_is_not_the_enumerated_one(
        tmp_path, tek_kok, monkeypatch):
    """`refused` is not the claim. The claim is that nothing outside is
    read, so the count of bytes read after the swap is the measurement
    and it has to be zero."""
    disari = _disari_kur(tmp_path)
    ebeveyn = tek_kok / "pipeline"
    durum = {"atesledi": False, "kuruldu": False}
    okunan = []

    gercek_liste = fs_transport.list_directory
    gercek_read = os.read

    def yaris(directory):
        kayitlar = gercek_liste(directory)
        if not durum["atesledi"] and any(k.name == "a.py" for k in kayitlar):
            durum["atesledi"] = True
            durum["kuruldu"] = _ebeveyni_takas_et(ebeveyn, disari)
        return kayitlar

    def sayan(descriptor, boyut):
        blok = gercek_read(descriptor, boyut)
        okunan.append(len(blok))
        return blok

    monkeypatch.setattr(fs_transport, "list_directory", yaris)
    monkeypatch.setattr(os, "read", sayan)
    with pytest.raises(fs_evidence.EvidenceError):
        _scan(tek_kok)

    assert durum["atesledi"] and durum["kuruldu"], "senaryo kurulmadi"
    assert sum(okunan) == 0, \
        f"takastan sonra {sum(okunan)} bayt okundu -- ret okumadan once degil"


def test_a_swap_during_listing_is_invisible_to_a_handle_bound_listing(
        tmp_path, tek_kok, monkeypatch):
    """The swap that beat checks on BOTH sides of a path-based listing:
    put the junction in place after the first check and take it away
    before the second. A listing that comes from a handle never looked
    at the name, so there is nothing to beat."""
    disari = tmp_path / "disarida"
    disari.mkdir()
    (disari / "gizli.py").write_bytes(SENTINEL.encode("ascii"))
    ebeveyn = tek_kok / "pipeline"
    saklanan = tmp_path / "saklanan"
    durum = {"atesledi": False, "kuruldu": False, "cagri": 0}
    gercek = fs_transport.list_directory

    def yaris(directory):
        # call 1 is the root's own listing; the swap has to land while
        # the CHILD directory is being listed, which is call 2
        durum["cagri"] += 1
        if durum["cagri"] != 2 or durum["atesledi"]:
            return gercek(directory)
        durum["atesledi"] = True
        try:
            ebeveyn.rename(saklanan)
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(str(disari), str(ebeveyn))
            else:
                os.symlink(str(disari), str(ebeveyn))
            durum["kuruldu"] = True
        except OSError:
            return gercek(directory)
        try:
            return gercek(directory)
        finally:
            if os.name == "nt":
                ebeveyn.rmdir()
            else:
                ebeveyn.unlink()
            saklanan.rename(ebeveyn)

    monkeypatch.setattr(fs_transport, "list_directory", yaris)
    manifest = _scan(tek_kok)

    assert durum["atesledi"] and durum["kuruldu"], "senaryo kurulmadi"
    assert not os.path.islink(ebeveyn), "senaryo kurulmadi: geri alinmadi"
    yollar = {e.path for e in manifest.entries}
    assert "pipeline/gizli.py" not in yollar, "disaridaki listeleme kabul edildi"
    assert "pipeline/a.py" in yollar


def test_the_scan_describes_the_tree_of_the_root_it_opened(
        tmp_path, tek_kok, monkeypatch):
    """The root is where a path-based walk is weakest, because every
    later lookup starts there. Once the root is OPEN, replacing its
    name redirects nothing: the scan keeps describing the object it
    took, which is the whole design in one sentence."""
    disari = tmp_path / "disarida"
    disari.mkdir()
    (disari / "disaridaki.py").write_bytes(SENTINEL.encode("ascii"))
    saklanan = tmp_path / "saklanan-kok"
    durum = {"atesledi": False, "kuruldu": False}
    gercek = fs_transport.list_directory

    def yaris(directory):
        if not durum["atesledi"]:
            durum["atesledi"] = True
            try:
                tek_kok.rename(saklanan)
                if os.name == "nt":
                    import _winapi
                    _winapi.CreateJunction(str(disari), str(tek_kok))
                else:
                    os.symlink(str(disari), str(tek_kok))
                durum["kuruldu"] = True
            except OSError:
                pass
        return gercek(directory)

    monkeypatch.setattr(fs_transport, "list_directory", yaris)
    try:
        manifest = _scan(tek_kok)
        assert durum["atesledi"] and durum["kuruldu"], "senaryo kurulmadi"
        assert os.path.exists(tek_kok / "disaridaki.py"), \
            "senaryo kurulmadi: kok adi disariya bakmiyor"
        yollar = {e.path for e in manifest.entries}
        assert "disaridaki.py" not in yollar, \
            "tarama kok adinin yeni hedefini takip etti"
        assert "pipeline/a.py" in yollar
    finally:
        if durum["kuruldu"]:
            if os.name == "nt":
                tek_kok.rmdir()
            else:
                tek_kok.unlink()
            saklanan.rename(tek_kok)



def test_a_metadata_only_file_is_never_read(tree, monkeypatch):
    """The invariant, corrected.

    The old test asserted "never opened" and watched `open_child_file`
    only -- so it stayed green while the metadata path opened a handle
    per file through a seam it could not see. Windows has no `fstatat`;
    a stat IS a handle there. What must be zero is the RIGHT to read
    data, the data descriptor, and the bytes."""
    veri_acilis, okumalar = [], []
    gercek_open = fs_transport.open_child_file
    gercek_read = os.read

    def izleyen_open(directory, record):
        veri_acilis.append(record.name)
        return gercek_open(directory, record)

    def izleyen_read(descriptor, boyut):
        blok = gercek_read(descriptor, boyut)
        okumalar.append(len(blok))
        return blok

    monkeypatch.setattr(fs_transport, "open_child_file", izleyen_open)
    monkeypatch.setattr(os, "read", izleyen_read)
    manifest = _scan(tree, metadata_only=("sahte_env",))

    assert manifest.metadata_only >= 2, "senaryo kurulmadi: metadata girdi yok"
    assert manifest.content_hashed >= 2, "senaryo kurulmadi: icerik girdi yok"
    assert "paket.py" not in veri_acilis, \
        "metadata sinifindaki dosya icin veri descriptor'i acildi"
    assert "a.py" in veri_acilis, "olumlu kontrol: icerik dosyasi acilmadi"
    # every byte read belongs to the content class; the metadata files
    # in this tree are 3 and 9 bytes and none of them may appear
    assert sum(okumalar) == 10, \
        f"okunan bayt {sum(okumalar)} -- metadata dosyasi da okunmus olabilir"


def test_the_metadata_and_content_paths_do_not_cross(tree):
    """One file may not be in both classes, and the counters must add
    up to the files actually seen."""
    manifest = _scan(tree, metadata_only=("sahte_env",),
                     content_always=("sahte_env/pyvenv.cfg",))
    by_path = {e.path: e for e in manifest.entries}
    icerikli = {p for p, e in by_path.items()
                if e.kind == "file" and e.sha256}
    bos = {p for p, e in by_path.items() if e.kind == "file" and not e.sha256}
    assert icerikli & bos == set()
    assert len(icerikli) == manifest.content_hashed
    assert len(bos) == manifest.metadata_only
    assert manifest.content_hashed + manifest.metadata_only == \
        manifest.file_count
    assert "sahte_env/pyvenv.cfg" in icerikli
    assert "sahte_env/Lib/paket.py" in bos



# =====================================================================
# LIMITS -- against the MEASURED protected roots, not a built tree
# =====================================================================

def test_the_default_limits_admit_the_measured_protected_roots():
    """The whole reason the limits are split by class. If any of these
    inverts, the walker refuses the real tree it exists to describe."""
    limits = fs_evidence.Limits()
    assert limits.max_entries > OLCULEN_GIRIS
    assert limits.max_logical_total_bytes > OLCULEN_TOPLAM
    assert limits.max_metadata_file_bytes > OLCULEN_EN_BUYUK
    # ... and the content ceiling must NOT admit it, or the split is
    # decorative and 19 GB would be hashed on every pass
    assert limits.max_content_file_bytes < OLCULEN_EN_BUYUK


def _seyrek_dosya(path: Path, size: int):
    """A file that CLAIMS the measured size without occupying it.

    184,685 real files are not built here; the one number that has to
    be exercised for real is the largest single file, and a sparse file
    is that number without 1.6 GB of disk.

    NOT `truncate`. It was measured writing 1,663 MB of real zeros on
    this volume with the sparse flag already set -- the CRT pads when
    it extends. Seeking past the end and writing one byte cost 0.1 MB
    for the same reported size, and the cost is asserted below rather
    than assumed, because a test that quietly writes 1.6 GB on every
    run is not the test being reported."""
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

    bos_once = shutil.disk_usage(path.parent).free
    with open(path, "wb") as akis:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            donen = wintypes.DWORD()
            if not kernel32.DeviceIoControl(
                    ctypes.c_void_p(msvcrt.get_osfhandle(akis.fileno())),
                    0x000900C4,                  # FSCTL_SET_SPARSE
                    None, 0, None, 0, ctypes.byref(donen), None):
                pytest.skip("bu birimde seyrek dosya kurulamiyor")
        akis.seek(size - 1)
        akis.write(b"\0")
    bos_sonra = shutil.disk_usage(path.parent).free

    info = os.lstat(path)
    if info.st_size != size:
        pytest.skip("dosya sistemi bu boyutu kabul etmedi")
    if bos_once - bos_sonra > (64 << 20):
        pytest.skip("dosya sistemi bu dosyayi seyrek tutmadi")
    if os.name == "nt":
        if not (getattr(info, "st_file_attributes", 0) & 0x200):
            pytest.skip("seyrek bayrak tutmadi")
    elif getattr(info, "st_blocks", 0) * 512 > size // 100:
        pytest.skip("dosya sistemi bu dosyayi seyrek tutmadi")
    return info


def test_the_largest_measured_file_is_admitted_only_as_metadata(tmp_path):
    kok = tmp_path / "kok"
    (kok / "buyuk_env").mkdir(parents=True)
    hedef = kok / "buyuk_env" / "kutuphane.bin"
    _seyrek_dosya(hedef, OLCULEN_EN_BUYUK)

    manifest = _scan(kok, metadata_only=("buyuk_env",))
    kayit = {e.path: e for e in manifest.entries}["buyuk_env/kutuphane.bin"]
    assert kayit.size == OLCULEN_EN_BUYUK
    assert kayit.sha256 == "", "metadata sinifindaki dosya acilip hash'lendi"

    # the same file in the content class is a refusal, never a
    # truncation and never a silent 1.6 GB read
    with pytest.raises(fs_evidence.EvidenceError):
        _scan(kok)


@pytest.mark.parametrize("sinir", ["metadata-tek-dosya", "icerik-toplami"])
def test_the_split_limits_also_fail_closed(tree, sinir):
    """The two ceilings the first version did not have. Low values, so
    the refusal is about the ceiling and not about the tree."""
    if sinir == "metadata-tek-dosya":
        limits = fs_evidence.Limits(max_metadata_file_bytes=1)
    else:
        limits = fs_evidence.Limits(max_content_total_bytes=1)
    with pytest.raises(fs_evidence.EvidenceError):
        _scan(tree, metadata_only=("sahte_env",), limits=limits)


# =====================================================================
# WINDOWS ATTRIBUTES, ROOT IDENTITY, EVIDENCE CLASS
# =====================================================================

def test_a_windows_attribute_change_alone_is_detected(tree):
    """HIDDEN, SYSTEM and READONLY change a file without moving size,
    timestamp or file id. The real API is used, because the point is
    that the operating system agrees the file changed."""
    if os.name != "nt":
        pytest.skip("dosya oznitelikleri Windows'a ozgu")
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    kernel32.SetFileAttributesW.restype = wintypes.BOOL

    hedef = tree / "pipeline" / "a.py"
    onceki = os.lstat(hedef)
    before = _scan(tree)
    try:
        assert kernel32.SetFileAttributesW(
            str(hedef), onceki.st_file_attributes | stat.FILE_ATTRIBUTE_HIDDEN), \
            "senaryo kurulmadi: oznitelik yazilamadi"
        sonraki = os.lstat(hedef)
        assert sonraki.st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN, \
            "senaryo kurulmadi: HIDDEN biti oturmadi"
        assert (onceki.st_size, onceki.st_mtime_ns, onceki.st_ino) == \
               (sonraki.st_size, sonraki.st_mtime_ns, sonraki.st_ino), \
            "senaryo kurulmadi: oznitelik disinda bir alan da degisti"
        after = _scan(tree)
        assert "pipeline/a.py" in _changed(before, after), \
            "yalnizca oznitelik degisimi gorulmuyor"
    finally:
        kernel32.SetFileAttributesW(str(hedef), onceki.st_file_attributes)


def test_the_root_replaced_by_another_directory_is_detected(tmp_path):
    """A root swapped for a different directory holding the SAME child
    left the entry list byte-identical. Only the root's own identity
    separates them, and this test proves the entries do not."""
    kok = tmp_path / "kok"
    kok.mkdir()
    (kok / "x.py").write_bytes(b"AAAA\n")
    before = _scan(kok)
    kok_onceki = os.lstat(kok)
    cocuk_onceki = os.lstat(kok / "x.py")

    _quiesce()
    yeni = tmp_path / "yeni"
    yeni.mkdir()
    os.replace(kok / "x.py", yeni / "x.py")      # ayni dosya, tasindi
    kok.rmdir()
    yeni.rename(kok)

    kok_sonraki = os.lstat(kok)
    cocuk_sonraki = os.lstat(kok / "x.py")
    assert kok_onceki.st_ino != kok_sonraki.st_ino, \
        "senaryo kurulmadi: kok kimligi degismedi"
    assert cocuk_onceki.st_ino == cocuk_sonraki.st_ino, \
        "senaryo kurulmadi: cocuk kimligi de degisti"

    after = _scan(kok)
    assert before.entries == after.entries, \
        "cocuk kayitlari farki tasiyor -- test kok kimligini olcmuyor"
    assert before.root_identity != after.root_identity
    assert fs_evidence.diff(before, after) == (".",)
    assert before.digest != after.digest, "kok kimligi ozete baglanmamis"


def test_content_always_is_an_exact_path_that_outranks_the_prefix(tmp_path):
    """`pyvenv.cfg` decides which interpreter runs, and it lives inside
    a tree that is metadata-only for measured cost reasons. Exact path
    beats prefix -- and only the exact path."""
    kok = tmp_path / "kok"
    (kok / "env" / "Lib").mkdir(parents=True)
    (kok / "env" / "pyvenv.cfg").write_bytes(b"home = X\n")
    (kok / "env" / "pyvenv.cfg.bak").write_bytes(b"home = X\n")
    (kok / "env" / "Lib" / "paket.py").write_bytes(b"PP\n")

    # NEGATIVE CONTROL: without the override the prefix does swallow it
    yalin = {e.path: e
             for e in _scan(kok, metadata_only=("env",)).entries}
    assert yalin["env/pyvenv.cfg"].sha256 == "", \
        "olumsuz kontrol kurulmadi: onek zaten yutmuyor"

    kayit = {e.path: e for e in _scan(
        kok, metadata_only=("env",),
        content_always=("env/pyvenv.cfg",)).entries}
    assert kayit["env/pyvenv.cfg"].sha256 != ""            # pozitif
    assert kayit["env/pyvenv.cfg.bak"].sha256 == "", \
        "tam yol degil onek gibi davraniyor"                # olumsuz
    assert kayit["env/Lib/paket.py"].sha256 == "", \
        "override tum agaci icerik sinifina cekti"          # olumsuz


# =====================================================================
# NAMES
# =====================================================================

@pytest.mark.parametrize("bosluk", ["\u00a0", "\u2007", "\u3000"])
def test_a_space_that_is_not_the_ordinary_one_is_refused(tree, bosluk):
    """U+00A0 renders exactly like U+0020. A reviewer comparing two
    report lines cannot tell them apart, so the name never enters an
    inventory in the first place."""
    ad = f"gorunen{bosluk}bosluk.py"
    try:
        (tree / "pipeline" / ad).write_bytes(b"X")
    except OSError:
        pytest.skip("bu dosya sistemi bu adi kabul etmiyor")
    if ad not in os.listdir(tree / "pipeline"):
        pytest.skip("dosya sistemi adi kendisi degistirdi")
    with pytest.raises(fs_evidence.UnsupportedEntry):
        _scan(tree)


def test_a_name_that_is_not_nfc_is_refused_rather_than_normalised(tree):
    """Refused, NOT rewritten: silently composing a name makes the
    manifest describe a file the filesystem does not have."""
    ad = "e\u0301klenti.py"                # ayrisik e + aksan
    assert unicodedata.normalize("NFC", ad) != ad, "senaryo kurulmadi"
    try:
        (tree / "pipeline" / ad).write_bytes(b"X")
    except OSError:
        pytest.skip("bu dosya sistemi bu adi kabul etmiyor")
    if ad not in os.listdir(tree / "pipeline"):
        pytest.skip("dosya sistemi adi kendisi normalize ediyor")
    with pytest.raises(fs_evidence.UnsupportedEntry):
        _scan(tree)


# =====================================================================
# THE LINK FINGERPRINT AND ITS KEY
# =====================================================================

def test_the_link_fingerprint_is_keyed_and_is_not_a_plain_digest(
        tmp_path, tree):
    """An unkeyed hash of a short path is a dictionary away from being
    readable, and a link may point somewhere private."""
    hedef = tmp_path / f"hedef-{SENTINEL}.txt"
    hedef.write_bytes(b"X")
    link = tree / "pipeline" / "baglanti.py"
    try:
        os.symlink(str(hedef), str(link))
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")

    ilk = _scan(tree, key=KEY)
    ikinci = _scan(tree, key=KEY)
    baska = _scan(tree, key=b"D" * 32)

    def al(m):
        return {e.path: e for e in m.entries}["pipeline/baglanti.py"]

    assert len(al(ilk).link_target_mac) == 64
    assert al(ilk).link_target_mac == al(ikinci).link_target_mac, \
        "ayni anahtar ayni kimligi vermiyor -- once/sonra kiyaslanamaz"
    assert al(ilk).link_target_mac != al(baska).link_target_mac, \
        "anahtar kimligi etkilemiyor"

    # the evidence bytes differ per platform -- a target string on
    # POSIX, a reparse buffer on Windows -- so the claim is pinned
    # against whatever the transport actually returned
    d = fs_transport.open_root(tree)
    try:
        pipeline = [k for k in fs_transport.list_directory(d)
                    if k.name == "pipeline"][0]
        alt = fs_transport.open_child_directory(d, pipeline)
        try:
            kayit = [k for k in fs_transport.list_directory(alt)
                     if k.name == "baglanti.py"][0]
            ham = fs_transport.link_evidence(alt, kayit)
        finally:
            fs_transport.close_directory(alt)
    finally:
        fs_transport.close_directory(d)

    assert isinstance(ham, bytes) and ham
    assert al(ilk).link_target_mac != hashlib.sha256(ham).hexdigest(), \
        "parmak izi anahtarsiz duz SHA-256"
    assert al(ilk).link_target_mac == hmac.new(
        KEY, ham, hashlib.sha256).hexdigest()

    tasinan = repr(ilk)
    assert SENTINEL not in tasinan, "hedef metni manifeste sizdi"
    assert KEY.hex() not in tasinan and str(KEY) not in tasinan, \
        "anahtar manifeste sizdi"


def test_an_unreadable_link_target_leaks_neither_target_nor_key(
        tmp_path, tree, monkeypatch):
    """Same correction on the link path: typed only, and the leak
    checks run unconditionally."""
    hedef = tmp_path / f"hedef-{SENTINEL}.txt"
    hedef.write_bytes(b"X")
    try:
        os.symlink(str(hedef), str(tree / "pipeline" / "baglanti.py"))
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")

    def patlayan(directory, record):
        raise OSError(f"{SENTINEL} {hedef}")

    monkeypatch.setattr(fs_transport, "link_evidence", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tree)
    metin = str(ret.value) + repr(ret.value)
    assert SENTINEL not in metin
    assert str(hedef) not in metin
    assert KEY.hex() not in metin
    assert "/" not in metin and "\\" not in metin


def test_the_link_key_is_required_and_exactly_typed(tree):
    """Exactly `bytes`, exactly one length. A `bytes` subclass can lie
    about its length while carrying nothing, and a `bytearray` can be
    edited between the before and after manifests."""
    with pytest.raises(TypeError):
        fs_evidence.scan(tree)                   # anahtar zorunlu

    class SahteBayt(bytes):
        def __len__(self):
            return fs_evidence.KEY_BYTES

    sahte = SahteBayt(b"")
    assert len(sahte) == fs_evidence.KEY_BYTES and bytes(sahte) == b"", \
        "senaryo kurulmadi: alt sinif uzunlugu taklit etmiyor"

    for kotu in (None, 32, "K" * 32, bytearray(b"K" * 32), b"",
                 b"K" * 16, b"K" * 64, sahte):
        with pytest.raises(fs_evidence.EvidenceError):
            fs_evidence.scan(tree, key=kotu)


# =====================================================================
# HANDLE LIFETIME
#
# The first walker queued every directory it opened and held them all
# until the scan ended: 24,209 simultaneous handles on the protected
# roots, for a walk that only ever needs the current path from the
# root. Residual zero was true and beside the point -- PEAK is the
# resource claim, and it is now bounded by depth.
# =====================================================================

def _handle_sayisi() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        sayi = wintypes.DWORD()
        ctypes.WinDLL("kernel32").GetProcessHandleCount(
            ctypes.c_void_p(-1), ctypes.byref(sayi))
        return sayi.value
    fd = os.open(os.devnull, os.O_RDONLY)
    os.close(fd)
    return fd


def _genis_agac(root: Path, kardes=400):
    root.mkdir(parents=True)
    for i in range(kardes):
        alt = root / f"d{i:04d}"
        alt.mkdir()
        (alt / "x.py").write_bytes(b"X\n")
    return root


def test_peak_open_directories_is_bounded_by_depth_not_by_width(tmp_path):
    """400 sibling directories, and the walk must never hold 400 open."""
    kok = _genis_agac(tmp_path / "genis")
    manifest = _scan(kok)
    assert manifest.file_count == 400, "senaryo kurulmadi"
    assert manifest.peak_directories <= 4, \
        f"ayni anda {manifest.peak_directories} dizin acik kaldi"


def test_peak_open_directories_tracks_depth(tmp_path):
    kok = tmp_path / "derin"
    yol = kok
    for i in range(12):
        yol = yol / f"k{i}"
    yol.mkdir(parents=True)
    (yol / "x.py").write_bytes(b"X\n")
    manifest = _scan(kok)
    assert manifest.peak_directories == 13, \
        f"derinlik 13 icin tepe {manifest.peak_directories}"


@pytest.mark.parametrize("nasil", ["basari", "ortada-hata", "limit-reddi"])
def test_no_directory_handle_survives_a_scan(tmp_path, monkeypatch, nasil):
    """Success, a failure halfway through, and a contract refusal all
    have to end with the same number of open objects as they started."""
    kok = _genis_agac(tmp_path / "genis", kardes=60)
    _scan(kok)                                   # warm any lazy state
    once = _handle_sayisi()

    if nasil == "basari":
        _scan(kok)
    elif nasil == "ortada-hata":
        gercek = fs_transport.open_child_file
        durum = {"sayi": 0}

        def patlayan(directory, record):
            durum["sayi"] += 1
            if durum["sayi"] == 30:
                raise fs_transport.TransportError("kurgu ariza")
            return gercek(directory, record)

        monkeypatch.setattr(fs_transport, "open_child_file", patlayan)
        with pytest.raises(fs_evidence.EvidenceError):
            _scan(kok)
        assert durum["sayi"] == 30, "senaryo kurulmadi: ariza tetiklenmedi"
    else:
        with pytest.raises(fs_evidence.EvidenceError):
            _scan(kok, limits=fs_evidence.Limits(max_entries=25))

    assert _handle_sayisi() <= once, "tarama acik nesne birakti"


def test_a_close_failure_is_reported_and_does_not_mask_the_real_error(
        tmp_path, monkeypatch):
    """Two rules at once: a cleanup problem must be visible, and it must
    never replace the error that caused the cleanup."""
    kok = _genis_agac(tmp_path / "genis", kardes=8)
    gercek_close = fs_transport.close_directory

    def kotu_close(directory):
        gercek_close(directory)
        raise fs_transport.TransportError("kurgu kapatma arizasi")

    # 1. close failure alone -- must surface as a typed refusal
    monkeypatch.setattr(fs_transport, "close_directory", kotu_close)
    with pytest.raises(fs_evidence.EvidenceError) as yalniz:
        _scan(kok)
    assert "kapat" in str(yalniz.value)

    # 2. close failure UNDER a real error -- the real error must win
    gercek_open = fs_transport.open_child_file

    def patlayan(directory, record):
        raise fs_transport.TransportError("kurgu birincil ariza")

    monkeypatch.setattr(fs_transport, "open_child_file", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as birlikte:
        _scan(kok)
    assert "birincil" in str(birlikte.value), \
        "temizlik hatasi birincil hatanin yerine gecti"
    notlar = " ".join(getattr(birlikte.value, "__notes__", []))
    assert "temizlik" in notlar, "temizlik hatasi gorunmez oldu"
    assert gercek_open is not None


def test_a_tree_dense_with_reparse_entries_is_accepted_deterministically(
        tmp_path):
    """The derived-export root, at its measured size.

    2,190 directory symlinks were counted there, and the earlier design
    refused all of them -- which would have taken a real part of the
    main checkout out of protected scope. They are fingerprinted now, so
    the size that mattered is the size the test builds, and the claim is
    that the answer is stable rather than merely produced."""
    kok = tmp_path / "turetilmis"
    (kok / "hedefler").mkdir(parents=True)
    hedef = kok / "hedefler" / "ortak"
    hedef.mkdir()
    (hedef / "x.bin").write_bytes(b"X\n")
    try:
        os.symlink(str(hedef), str(kok / "b0000"), target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    for i in range(1, 2190):
        os.symlink(str(hedef), str(kok / f"b{i:04d}"),
                   target_is_directory=True)

    ilk = _scan(kok, metadata_only=("hedefler",))
    ikinci = _scan(kok, metadata_only=("hedefler",))

    assert ilk.reparse_count == 2190, \
        f"senaryo kurulmadi: {ilk.reparse_count} ayrisma girdisi"
    assert ilk.digest == ikinci.digest, "ozet deterministik degil"
    assert _changed(ilk, ikinci) == set()
    # fingerprinted, never followed: the target's own child appears once,
    # under the real directory, and not once per link
    yollar = [e.path for e in ilk.entries]
    assert yollar.count("hedefler/ortak/x.bin") == 1
    assert not [y for y in yollar if y.startswith("b0000/")]
    # and every link got a distinct-from-nothing keyed code
    maclar = {e.link_target_mac for e in ilk.entries if e.kind == "link"}
    assert len(maclar) == 1 and "" not in maclar, \
        "ayni hedefe bakan baglantilar ayni kimligi almali ve bos olmamali"


# =====================================================================
# CLEANUP INFORMATION HAS TO SURVIVE THE BOUNDARY
#
# The transport marks a cleanup failure on the error it is raising. The
# walker then builds a NEW exception of its own type, and the mark was
# measured stopping right there: GUARD_TYPE=EvidenceError, GUARD_NOTES=[].
# A problem made visible in one layer and invisible in the next is the
# same silent-zero shape this package has been bitten by repeatedly.
# =====================================================================

def _tasima_hatasi(mesaj="kurgu tasima retti", *, temizlik=False,
                   keyfi=None):
    hata = fs_transport.TransportError(mesaj)
    if temizlik:
        fs_transport.mark_cleanup_failed(hata)
    if keyfi:
        hata.add_note(keyfi)
    return hata


def test_a_cleanup_failure_marked_below_is_still_visible_above(
        tree, monkeypatch):
    """The positive control: the flag crosses the guard."""
    def patlayan(directory):
        raise _tasima_hatasi(temizlik=True)

    monkeypatch.setattr(fs_transport, "list_directory", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tree)

    assert fs_transport.cleanup_failed(ret.value) is True, \
        "temizlik isareti _guard sinirinda kayboldu"
    assert fs_transport.CLEANUP_NOTE in getattr(ret.value, "__notes__", []), \
        "sabit temizlik cumlesi ust hatada yok"


def test_a_clean_transport_error_is_not_marked_as_a_cleanup_failure(
        tree, monkeypatch):
    """The other half: without a real cleanup problem the flag must be
    absent, or it means nothing when it is present."""
    def patlayan(directory):
        raise _tasima_hatasi()

    monkeypatch.setattr(fs_transport, "list_directory", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tree)

    assert fs_transport.cleanup_failed(ret.value) is False
    assert not getattr(ret.value, "__notes__", [])


def test_an_arbitrary_note_from_below_is_not_carried_across(tree,
                                                            monkeypatch):
    """NEGATIVE CONTROL, and the reason the flag exists instead of a
    note copy. Notes are free text written closer to the filesystem;
    copying them across is a path leak with extra steps."""
    keyfi = f"KEYFI-NOT {SENTINEL} {tree}"

    def patlayan(directory):
        raise _tasima_hatasi(temizlik=True, keyfi=keyfi)

    monkeypatch.setattr(fs_transport, "list_directory", patlayan)
    with pytest.raises(fs_evidence.EvidenceError) as ret:
        _scan(tree)

    notlar = list(getattr(ret.value, "__notes__", []))
    assert fs_transport.cleanup_failed(ret.value) is True, \
        "senaryo kurulmadi: temizlik isareti hic gecmemis"
    assert notlar == [fs_transport.CLEANUP_NOTE], \
        f"yalniz sabit cumle beklenirken {notlar}"
    tam = " ".join(notlar) + str(ret.value) + repr(ret.value)
    assert SENTINEL not in tam and str(tree) not in tam
