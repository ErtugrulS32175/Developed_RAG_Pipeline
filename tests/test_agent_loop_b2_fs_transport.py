"""PACKAGE B2B-A-D2 -- the handle-bound transport, on its own.

WHY SEPARATELY. The walker's guarantee is only as good as the primitive
underneath it, and that primitive is different code on each platform:
`O_DIRECTORY` + `dir_fd` on POSIX, `GetFileInformationByHandleEx` plus
`NtCreateFile` with a RootDirectory handle on Windows. A test that only
drives the walker cannot tell a working primitive from a walker that
has not hit the case yet.

WHAT IS PINNED HERE, and every one of them is a measurement this design
was changed by:

  * A relative name is DATA. `NtCreateFile` was measured accepting
    `pipeline\\a.py` and opening the grandchild, so separators are
    refused before any name is used.
  * The enumeration is trusted for NAMES ONLY. Its per-child metadata
    is a lazily flushed copy of the parent index: a same-size rewrite
    was invisible in 117 of 200 attempts and an atomic replace in 137
    of 200.
  * An enumeration buffer is attacker-shaped data with offsets in it,
    and is validated like a network packet.
"""
from __future__ import annotations

import os
import stat
import struct
import time

import pytest

from tools.agent_loop import fs_transport

SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
_WIN_MAX = 16384                    # MAXIMUM_REPARSE_DATA_BUFFER_SIZE
_WINDOWS = os.name == "nt"

if _WINDOWS:
    import ctypes
    from tools.agent_loop import fs_transport_windows as _win


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _agac(tmp_path):
    kok = tmp_path / "kok"
    (kok / "alt").mkdir(parents=True)
    (kok / "a.py").write_bytes(b"ICERIDE\n")
    (kok / "alt" / "b.py").write_bytes(b"BB\n")
    return kok


def _kayitlar(directory):
    return {kayit.name: kayit for kayit in
            fs_transport.list_directory(directory)}


def _baglanti_kurulabiliyor(tmp_path) -> bool:
    hedef = tmp_path / "deneme-hedefi"
    hedef.write_bytes(b"X")
    baglanti = tmp_path / "deneme-baglantisi"
    try:
        os.symlink(str(hedef), str(baglanti))
    except OSError:
        return False
    baglanti.unlink()
    return True


def _kaynak_sayisi() -> int:
    """Open kernel objects. Windows counts handles; POSIX counts the
    next free descriptor, which only moves if one leaked."""
    if _WINDOWS:
        sayi = ctypes.wintypes.DWORD()
        ctypes.WinDLL("kernel32").GetProcessHandleCount(
            ctypes.c_void_p(-1), ctypes.byref(sayi))
        return sayi.value
    fd = os.open(os.devnull, os.O_RDONLY)
    os.close(fd)
    return fd


# ---------------------------------------------------------------------
# THE ORDINARY CASE -- or every refusal below is just a broken opener
# ---------------------------------------------------------------------

def test_a_root_opens_and_lists_its_children(tmp_path):
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        kayitlar = _kayitlar(directory)
        assert set(kayitlar) == {"a.py", "alt"}, "'.' ve '..' sizdi ya da eksik"
        assert kayitlar["a.py"].kind == "file"
        assert kayitlar["alt"].kind == "dir"
        assert kayitlar["a.py"].size == 8
        assert kayitlar["a.py"].file_id and kayitlar["alt"].file_id
        assert kayitlar["a.py"].file_id != kayitlar["alt"].file_id
    finally:
        fs_transport.close_directory(directory)


def test_a_child_file_opens_relative_and_reads(tmp_path):
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        descriptor = fs_transport.open_child_file(directory,
                                                  _kayitlar(directory)["a.py"])
        try:
            assert os.read(descriptor, 1 << 20) == b"ICERIDE\n"
        finally:
            os.close(descriptor)
    finally:
        fs_transport.close_directory(directory)


def test_a_child_directory_opens_relative_and_lists(tmp_path):
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        alt = fs_transport.open_child_directory(directory,
                                                _kayitlar(directory)["alt"])
        try:
            assert set(_kayitlar(alt)) == {"b.py"}
        finally:
            fs_transport.close_directory(alt)
    finally:
        fs_transport.close_directory(directory)


def test_closing_a_directory_twice_is_harmless_and_then_refuses(tmp_path):
    directory = fs_transport.open_root(_agac(tmp_path))
    fs_transport.close_directory(directory)
    fs_transport.close_directory(directory)
    with pytest.raises(fs_transport.TransportError):
        fs_transport.list_directory(directory)


# ---------------------------------------------------------------------
# THE RECORD IS AUTHORITATIVE -- the measurement that changed the design
# ---------------------------------------------------------------------

@pytest.mark.parametrize("degisiklik", ["ayni-boyut", "atomik-replace"])
def test_a_listing_reflects_a_change_the_parent_index_would_hide(
        tmp_path, degisiklik):
    """The two changes a lazily flushed index entry was measured to
    miss: a same-size in-place rewrite (117 of 200 invisible) and an
    atomic replace (137 of 200 invisible, still carrying the OLD file
    identity). Nothing here stats the child, because `os.stat` is what
    flushes the entry and would hide the very defect being tested."""
    kok = tmp_path / "kok"
    kok.mkdir()
    hedef = kok / "a.py"
    hedef.write_bytes(b"AAAA\n")

    directory = fs_transport.open_root(kok)
    try:
        once = _kayitlar(directory)["a.py"]
        # the measured timestamp quantum, which the production seam
        # always waits out between a baseline and anything writing --
        # without it a same-size rewrite lands in the same tick and no
        # metadata field can carry it
        time.sleep(0.010)
        if degisiklik == "ayni-boyut":
            hedef.write_bytes(b"BBBB\n")
        else:
            gecici = kok / "a.tmp"
            gecici.write_bytes(b"CCCC\n")
            os.replace(gecici, hedef)
        sonra = _kayitlar(directory)["a.py"]
        assert sonra != once, \
            "listeleme degisikligi kaciriyor -- ebeveyn indeksine guveniliyor"
    finally:
        fs_transport.close_directory(directory)


def test_a_listing_of_an_untouched_directory_is_stable(tmp_path):
    """The other side of the same coin: no false positives, or the
    metadata class is noise."""
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        assert _kayitlar(directory) == _kayitlar(directory)
    finally:
        fs_transport.close_directory(directory)


# ---------------------------------------------------------------------
# NAMES FROM A LISTING ARE DATA
# ---------------------------------------------------------------------

@pytest.mark.parametrize("ad", ["..", ".", "", "alt\\b.py", "alt/b.py",
                                "a\0b"])
def test_a_name_with_a_separator_or_traversal_is_refused(ad):
    with pytest.raises(fs_transport.TransportError):
        fs_transport.validate_child_name(ad)


@pytest.mark.parametrize("ad", [b"a.py", 3, None])
def test_a_name_that_is_not_exactly_str_is_refused(ad):
    with pytest.raises(fs_transport.TransportError):
        fs_transport.validate_child_name(ad)


def test_a_forged_record_name_cannot_reach_a_grandchild(tmp_path):
    """MEASURED: `NtCreateFile` accepted `alt\\b.py` as a relative name
    and opened the grandchild. The refusal is what stops a listing from
    becoming a path."""
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        gercek = _kayitlar(directory)["a.py"]
        sahte = fs_transport.Record(
            name="alt\\b.py" if _WINDOWS else "alt/b.py", kind=gercek.kind,
            attributes=gercek.attributes, reparse_tag=gercek.reparse_tag,
            mode=gercek.mode, size=gercek.size, mtime_ns=gercek.mtime_ns,
            file_id=gercek.file_id, nlink=gercek.nlink)
        with pytest.raises(fs_transport.TransportError):
            fs_transport.open_child_file(directory, sahte)
    finally:
        fs_transport.close_directory(directory)


def test_an_opened_child_that_is_not_the_listed_one_is_refused(tmp_path):
    """The binding, exercised directly: the identity in the record is
    what the opened object has to match."""
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        gercek = _kayitlar(directory)["a.py"]
        sahte = fs_transport.Record(
            name=gercek.name, kind=gercek.kind, attributes=gercek.attributes,
            reparse_tag=gercek.reparse_tag, mode=gercek.mode,
            size=gercek.size, mtime_ns=gercek.mtime_ns,
            file_id="0000:" + "00" * 16, nlink=gercek.nlink)
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.open_child_file(directory, sahte)
        assert "listelenen" in str(ret.value)
    finally:
        fs_transport.close_directory(directory)


# ---------------------------------------------------------------------
# WHAT MUST NOT OPEN
# ---------------------------------------------------------------------

def test_a_root_that_is_not_a_directory_is_refused(tmp_path):
    hedef = tmp_path / "a.py"
    hedef.write_bytes(b"X")
    with pytest.raises(fs_transport.TransportError):
        fs_transport.open_root(hedef)


def test_a_missing_root_is_refused_with_no_raw_os_error(tmp_path):
    with pytest.raises(fs_transport.TransportError) as ret:
        fs_transport.open_root(tmp_path / f"{SENTINEL}-yok")
    metin = str(ret.value) + repr(ret.value)
    assert SENTINEL not in metin, "isletim sistemi metni yolu tasidi"
    assert "/" not in metin and "\\" not in metin


def test_a_link_is_listed_as_a_link_and_never_opened_as_a_file(tmp_path):
    kok = _agac(tmp_path)
    disarida = tmp_path / "disarida.txt"
    disarida.write_bytes(SENTINEL.encode("ascii"))
    try:
        os.symlink(str(disarida), str(kok / "baglanti.py"))
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    directory = fs_transport.open_root(kok)
    try:
        kayit = _kayitlar(directory)["baglanti.py"]
        assert kayit.kind == "link"
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.open_child_file(directory, kayit)
        assert SENTINEL not in str(ret.value) + repr(ret.value)
    finally:
        fs_transport.close_directory(directory)


def test_a_junction_is_listed_as_a_link(tmp_path):
    if not _WINDOWS:
        pytest.skip("kavsak noktasi Windows'a ozgu")
    import _winapi

    kok = _agac(tmp_path)
    disari = tmp_path / "kavsak-hedefi"
    disari.mkdir()
    (disari / "icerde.txt").write_bytes(SENTINEL.encode("ascii"))
    _winapi.CreateJunction(str(disari), str(kok / "kavsak"))
    directory = fs_transport.open_root(kok)
    try:
        kayit = _kayitlar(directory)["kavsak"]
        assert kayit.kind == "link" and kayit.reparse_tag
        with pytest.raises(fs_transport.TransportError):
            fs_transport.open_child_directory(directory, kayit)
    finally:
        fs_transport.close_directory(directory)


def test_a_fifo_is_refused_without_blocking(tmp_path):
    """`O_RDONLY` on a FIFO blocks until a writer appears, so the POSIX
    open carries `O_NONBLOCK`. The alarm is the point: without it a
    regression here does not fail, it hangs."""
    if not hasattr(os, "mkfifo"):
        pytest.skip("bu platformda FIFO yaratilamiyor")
    import signal

    kok = tmp_path / "kok"
    kok.mkdir()
    os.mkfifo(str(kok / "boru"))
    assert stat.S_ISFIFO(os.lstat(kok / "boru").st_mode), "senaryo kurulmadi"

    directory = fs_transport.open_root(kok)

    def zaman_asimi(signum, frame):
        raise AssertionError("acilis blokladi -- O_NONBLOCK dusmus")

    onceki = signal.signal(signal.SIGALRM, zaman_asimi)
    signal.setitimer(signal.ITIMER_REAL, 5.0)
    try:
        kayit = _kayitlar(directory)["boru"]
        with pytest.raises(fs_transport.TransportError):
            fs_transport.open_child_file(directory, kayit)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, onceki)
        fs_transport.close_directory(directory)


def test_a_refused_open_leaks_no_kernel_object(tmp_path):
    """The Windows path opens a handle FIRST and only then discovers
    the mismatch. A refusal that keeps the handle exhausts the process
    on a tree full of them."""
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        gercek = _kayitlar(directory)["a.py"]
        sahte = fs_transport.Record(
            name=gercek.name, kind=gercek.kind, attributes=gercek.attributes,
            reparse_tag=gercek.reparse_tag, mode=gercek.mode,
            size=gercek.size, mtime_ns=gercek.mtime_ns,
            file_id="0000:" + "00" * 16, nlink=gercek.nlink)
        once = _kaynak_sayisi()
        for _ in range(50):
            with pytest.raises(fs_transport.TransportError):
                fs_transport.open_child_file(directory, sahte)
        assert _kaynak_sayisi() <= once, \
            "reddedilen acilis cekirdek nesnesi sizdirdi"
    finally:
        fs_transport.close_directory(directory)


def test_a_successful_walk_leaks_no_kernel_object(tmp_path):
    kok = _agac(tmp_path)
    once = _kaynak_sayisi()
    for _ in range(20):
        directory = fs_transport.open_root(kok)
        try:
            for kayit in fs_transport.list_directory(directory):
                if kayit.kind == "dir":
                    alt = fs_transport.open_child_directory(directory, kayit)
                    fs_transport.close_directory(alt)
                elif kayit.kind == "file":
                    os.close(fs_transport.open_child_file(directory, kayit))
        finally:
            fs_transport.close_directory(directory)
    assert _kaynak_sayisi() <= once, "basarili yuruyus nesne sizdirdi"


# ---------------------------------------------------------------------
# THE ENUMERATION BUFFER IS UNTRUSTED DATA (Windows)
# ---------------------------------------------------------------------

def _girdi(next_offset: int, ad: str, *, ad_bayt=None, oznitelik=0x20):
    """One FILE_ID_EXTD_DIR_INFO, byte for byte. Header is 88."""
    kodlu = ad.encode("utf-16-le")
    uzunluk = len(kodlu) if ad_bayt is None else ad_bayt
    govde = struct.pack("<II", next_offset, 0)
    govde += struct.pack("<6q", 0, 0, 0, 0, 0, 0)
    govde += struct.pack("<IIII", oznitelik, uzunluk, 0, 0)
    govde += b"\0" * 16
    return govde + kodlu


def _tampon(*parcalar, boy=4096):
    ham = b"".join(parcalar)
    assert len(ham) <= boy, "test tamponu tasti"
    return ctypes.create_string_buffer(ham + b"\0" * (boy - len(ham)), boy)


@pytest.mark.skipif(not _WINDOWS, reason="enumerasyon tamponu Windows'a ozgu")
def test_a_well_formed_buffer_parses():
    """The positive control. Without it every refusal below could be a
    parser that rejects everything."""
    ilk = _girdi(96, "a.py")
    tampon = _tampon(ilk + b"\0" * (96 - len(ilk)), _girdi(0, "b.py"))
    assert _win._page_names(tampon) == ["a.py", "b.py"]


@pytest.mark.skipif(not _WINDOWS, reason="enumerasyon tamponu Windows'a ozgu")
def test_dot_entries_are_dropped_not_refused():
    ilk = _girdi(96, ".")
    ikinci = _girdi(96, "..")
    tampon = _tampon(ilk + b"\0" * (96 - len(ilk)),
                     ikinci + b"\0" * (96 - len(ikinci)),
                     _girdi(0, "a.py"))
    assert _win._page_names(tampon) == ["a.py"]


@pytest.mark.skipif(not _WINDOWS, reason="enumerasyon tamponu Windows'a ozgu")
@pytest.mark.parametrize("bozukluk", [
    "ilerlemeyen-ofset", "geri-giden-ofset", "tampon-disi-ofset",
    "tek-sayili-ad", "sifir-ad", "tasan-ad", "ayirici-iceren-ad",
])
def test_a_malformed_buffer_is_refused(bozukluk):
    """Offsets and lengths inside this buffer index into memory. It is
    validated the way a network packet is, and every one of these is a
    refusal rather than a best-effort parse."""
    if bozukluk == "ilerlemeyen-ofset":
        tampon = _tampon(_girdi(8, "a.py"))
    elif bozukluk == "geri-giden-ofset":
        ilk = _girdi(96, "a.py")
        tampon = _tampon(ilk + b"\0" * (96 - len(ilk)), _girdi(1, "b.py"))
    elif bozukluk == "tampon-disi-ofset":
        tampon = _tampon(_girdi(4000, "a.py"))
    elif bozukluk == "tek-sayili-ad":
        tampon = _tampon(_girdi(0, "a.py", ad_bayt=7))
    elif bozukluk == "sifir-ad":
        tampon = _tampon(_girdi(0, "a.py", ad_bayt=0))
    elif bozukluk == "tasan-ad":
        tampon = _tampon(_girdi(0, "a.py", ad_bayt=8000))
    else:
        tampon = _tampon(_girdi(0, "alt\\b.py"))
    with pytest.raises(fs_transport.TransportError):
        _win._page_names(tampon)


@pytest.mark.skipif(not _WINDOWS, reason="enumerasyon tamponu Windows'a ozgu")
def test_a_cyclic_buffer_ends_as_a_refusal_not_a_hang():
    """A page whose offsets form a cycle must not spin. The cap is what
    turns a hang into a verdict."""
    parcalar, ofset = [], 0
    while ofset + 96 <= 4096:
        parcalar.append(_girdi(96, "a.py").ljust(96, b"\0"))
        ofset += 96
    parcalar[-1] = _girdi(96, "a.py").ljust(96, b"\0")     # son kayit da ilerler
    with pytest.raises(fs_transport.TransportError):
        _win._page_names(_tampon(*parcalar))


@pytest.mark.skipif(not _WINDOWS, reason="sayfalama Windows'a ozgu")
def test_a_listing_paginates_over_many_buffers(tmp_path):
    """500 names were measured spanning 18 buffers of 4 KiB, ending on
    ERROR_NO_MORE_FILES."""
    kok = tmp_path / "kok"
    kok.mkdir()
    for i in range(500):
        (kok / f"dosya-{i:04d}-uzunca-bir-ad.py").write_bytes(b"X")
    directory = fs_transport.open_root(kok)
    try:
        kayitlar = fs_transport.list_directory(directory)
    finally:
        fs_transport.close_directory(directory)
    adlar = [kayit.name for kayit in kayitlar]
    assert len(adlar) == 500 and len(set(adlar)) == 500


@pytest.mark.skipif(not _WINDOWS, reason="sayfalama Windows'a ozgu")
def test_only_no_more_files_ends_a_listing(tmp_path, monkeypatch):
    """An incomplete inventory that looks complete is the failure this
    package has been bitten by more than once."""
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    gercek = _win._k32.GetFileInformationByHandleEx

    def yarim(handle, sinif, tampon, boy):
        if sinif in (19, 20):
            ctypes.set_last_error(1784)          # ERROR_INVALID_USER_BUFFER
            return 0
        return gercek(handle, sinif, tampon, boy)

    monkeypatch.setattr(_win._k32, "GetFileInformationByHandleEx", yarim)
    try:
        with pytest.raises(fs_transport.TransportError):
            fs_transport.list_directory(directory)
    finally:
        fs_transport.close_directory(directory)


def test_nothing_here_runs_git_or_a_subprocess():
    """The authority moved off git because git's state was reachable by
    the model. A transport that shells out would put it back."""
    from pathlib import Path
    for modul in (fs_transport,
                  _win if _WINDOWS else
                  __import__("tools.agent_loop.fs_transport_posix",
                             fromlist=["x"])):
        kaynak = Path(modul.__file__).read_text(encoding="utf-8").lower()
        for yasak in ("subprocess", "import git", "popen", "system("):
            assert yasak not in kaynak, f"{modul.__name__} {yasak} tasiyor"


# =====================================================================
# WHAT IS ACTUALLY ASKED FOR (Windows)
#
# The normative invariant is not "no handle". It is that the metadata
# path never asks for the right to read data. That is an argument to
# `NtCreateFile`, so it is asserted at that argument.
# =====================================================================

@pytest.fixture
def nt_kaydi(monkeypatch):
    if not _WINDOWS:
        pytest.skip("NtCreateFile Windows'a ozgu")
    kayitlar = []
    gercek = _win._ntdll.NtCreateFile

    def izleyen(handle_ptr, erisim, oa, iosb, alloc, attrs, share,
                disposition, options, ea, ea_len):
        # `byref()` yields a CArgObject: no `.contents`, but it does
        # keep `_obj`, which is the structure that was passed
        nesne = getattr(oa, "_obj", oa)
        kayitlar.append({"erisim": erisim, "secenek": options,
                         "kok": bool(nesne.RootDirectory),
                         "disposition": disposition})
        return gercek(handle_ptr, erisim, oa, iosb, alloc, attrs, share,
                      disposition, options, ea, ea_len)

    monkeypatch.setattr(_win._ntdll, "NtCreateFile", izleyen)
    return kayitlar


def _kok_ile(tmp_path):
    kok = tmp_path / "kok"
    (kok / "alt").mkdir(parents=True)
    (kok / "a.py").write_bytes(b"ICERIDE\n")
    return kok


def test_a_metadata_query_asks_for_attributes_and_nothing_else(
        tmp_path, nt_kaydi):
    kok = _kok_ile(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        fs_transport.list_directory(directory)
    finally:
        fs_transport.close_directory(directory)

    assert nt_kaydi, "NtCreateFile hic cagrilmadi -- test olcum yapmiyor"
    for cagri in nt_kaydi:
        assert cagri["erisim"] == _win.METADATA_ACCESS, \
            f"metadata erisimi 0x{cagri['erisim']:08X}"
        assert not cagri["erisim"] & _win.FILE_READ_DATA
        assert not cagri["erisim"] & _win.GENERIC_READ
        assert not cagri["erisim"] & _win.FORBIDDEN_ACCESS
        assert cagri["kok"], "goreli acilis degil -- RootDirectory bos"
        assert cagri["secenek"] & _win._FILE_OPEN_REPARSE_POINT
        assert cagri["disposition"] == _win._FILE_OPEN


def test_a_content_open_asks_for_data_and_a_directory_open_for_listing(
        tmp_path, nt_kaydi):
    kok = _kok_ile(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        kayitlar = {k.name: k for k in fs_transport.list_directory(directory)}
        nt_kaydi.clear()
        os.close(fs_transport.open_child_file(directory, kayitlar["a.py"]))
        icerik = list(nt_kaydi)
        nt_kaydi.clear()
        alt = fs_transport.open_child_directory(directory, kayitlar["alt"])
        fs_transport.close_directory(alt)
        dizin = list(nt_kaydi)
    finally:
        fs_transport.close_directory(directory)

    assert len(icerik) == 1 and len(dizin) == 1
    assert icerik[0]["erisim"] == _win.CONTENT_ACCESS
    assert icerik[0]["erisim"] & _win.FILE_READ_DATA
    assert icerik[0]["secenek"] & _win._FILE_NON_DIRECTORY_FILE
    assert not icerik[0]["erisim"] & _win.FORBIDDEN_ACCESS

    assert dizin[0]["erisim"] == _win.DIRECTORY_ACCESS
    assert dizin[0]["erisim"] & _win.FILE_LIST_DIRECTORY
    assert dizin[0]["secenek"] & _win._FILE_DIRECTORY_FILE
    assert not dizin[0]["erisim"] & _win.FORBIDDEN_ACCESS


def test_a_reparse_fingerprint_asks_for_attributes_only(tmp_path, nt_kaydi):
    kok = _kok_ile(tmp_path)
    disari = tmp_path / "disarida"
    disari.mkdir()
    try:
        os.symlink(str(disari), str(kok / "baglanti"),
                   target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    directory = fs_transport.open_root(kok)
    try:
        kayit = {k.name: k
                 for k in fs_transport.list_directory(directory)}["baglanti"]
        nt_kaydi.clear()
        ham = fs_transport.link_evidence(directory, kayit)
    finally:
        fs_transport.close_directory(directory)

    assert isinstance(ham, bytes) and 0 < len(ham) <= \
        _win.MAXIMUM_REPARSE_DATA_BUFFER_SIZE
    assert nt_kaydi, "olcum yapilmadi"
    for cagri in nt_kaydi:
        assert cagri["erisim"] == _win.METADATA_ACCESS
        assert cagri["secenek"] & _win._FILE_OPEN_REPARSE_POINT


def test_no_data_descriptor_is_made_on_the_metadata_path(tmp_path,
                                                         monkeypatch):
    """`msvcrt.open_osfhandle` is the ONLY door from a handle to a data
    descriptor, so it is counted directly."""
    if not _WINDOWS:
        pytest.skip("open_osfhandle Windows'a ozgu")
    import msvcrt as _msvcrt

    kok = _kok_ile(tmp_path)
    cevrimler = []
    gercek = _msvcrt.open_osfhandle

    def izleyen(handle, flags):
        cevrimler.append(handle)
        return gercek(handle, flags)

    monkeypatch.setattr(_win.msvcrt, "open_osfhandle", izleyen)
    directory = fs_transport.open_root(kok)
    try:
        fs_transport.list_directory(directory)
        assert not cevrimler, "metadata yolu veri descriptor'i uretti"
    finally:
        fs_transport.close_directory(directory)


def test_a_reparse_fingerprint_is_bound_to_the_listing_first(tmp_path):
    """The control code must not run against an object the listing did
    not name."""
    kok = _kok_ile(tmp_path)
    disari = tmp_path / "disarida"
    disari.mkdir()
    try:
        os.symlink(str(disari), str(kok / "baglanti"),
                   target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")
    directory = fs_transport.open_root(kok)
    try:
        gercek = {k.name: k
                  for k in fs_transport.list_directory(directory)}["baglanti"]
        sahte = fs_transport.Record(
            name=gercek.name, kind=gercek.kind, attributes=gercek.attributes,
            reparse_tag=gercek.reparse_tag, mode=gercek.mode,
            size=gercek.size, mtime_ns=gercek.mtime_ns,
            file_id="0000:" + "00" * 16, nlink=gercek.nlink)
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.link_evidence(directory, sahte)
        assert "listelenen" in str(ret.value)
    finally:
        fs_transport.close_directory(directory)


def test_a_reparse_fingerprint_refuses_a_non_link(tmp_path):
    kok = _kok_ile(tmp_path)
    directory = fs_transport.open_root(kok)
    try:
        kayit = {k.name: k
                 for k in fs_transport.list_directory(directory)}["a.py"]
        with pytest.raises(fs_transport.TransportError):
            fs_transport.link_evidence(directory, kayit)
    finally:
        fs_transport.close_directory(directory)


def test_a_failed_device_control_is_a_typed_refusal(tmp_path, monkeypatch):
    if not _WINDOWS:
        pytest.skip("DeviceIoControl Windows'a ozgu")
    kok = _kok_ile(tmp_path)
    disari = tmp_path / "disarida"
    disari.mkdir()
    try:
        os.symlink(str(disari), str(kok / "baglanti"),
                   target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")

    monkeypatch.setattr(_win._k32, "DeviceIoControl",
                        lambda *args: 0)
    directory = fs_transport.open_root(kok)
    try:
        kayit = {k.name: k
                 for k in fs_transport.list_directory(directory)}["baglanti"]
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.link_evidence(directory, kayit)
        assert "ayrisma" in str(ret.value)
    finally:
        fs_transport.close_directory(directory)


@pytest.mark.parametrize("donen", [0, _WIN_MAX + 1])
def test_an_out_of_range_reparse_length_is_refused(tmp_path, monkeypatch,
                                                   donen):
    if not _WINDOWS:
        pytest.skip("DeviceIoControl Windows'a ozgu")
    kok = _kok_ile(tmp_path)
    disari = tmp_path / "disarida"
    disari.mkdir()
    try:
        os.symlink(str(disari), str(kok / "baglanti"),
                   target_is_directory=True)
    except OSError:
        pytest.skip("bu ortamda sembolik baglanti kurulamiyor")

    def yalanci(handle, kod, gir, gir_boy, cik, cik_boy, donen_ptr, ovl):
        getattr(donen_ptr, "_obj", donen_ptr).value = donen
        return 1

    monkeypatch.setattr(_win._k32, "DeviceIoControl", yalanci)
    directory = fs_transport.open_root(kok)
    try:
        kayit = {k.name: k
                 for k in fs_transport.list_directory(directory)}["baglanti"]
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.link_evidence(directory, kayit)
        assert "tutarsiz" in str(ret.value)
    finally:
        fs_transport.close_directory(directory)


def test_a_close_that_fails_is_a_refusal_not_a_success(tmp_path,
                                                       monkeypatch):
    """`closed` must mean closed. The first version set the flag before
    calling, so a handle that refused to close was reported released."""
    kok = _kok_ile(tmp_path)
    directory = fs_transport.open_root(kok)
    if _WINDOWS:
        monkeypatch.setattr(_win._k32, "CloseHandle", lambda handle: 0)
    else:
        def patlayan(fd):
            raise OSError(9, "kurgu")
        monkeypatch.setattr(os, "close", patlayan)

    with pytest.raises(fs_transport.TransportError):
        fs_transport.close_directory(directory)
    assert directory.closed is False, \
        "kapanmayan nesne kapanmis sayildi"

    monkeypatch.undo()
    fs_transport.close_directory(directory)
    assert directory.closed is True


def test_closing_an_already_closed_directory_is_a_no_op(tmp_path):
    directory = fs_transport.open_root(_kok_ile(tmp_path))
    fs_transport.close_directory(directory)
    fs_transport.close_directory(directory)
    assert directory.closed is True


# =====================================================================
# MECHANICAL PROOF, not a promise
#
# "The cleanup result is consumed" is the kind of claim that is true on
# the day it is written and quietly false three edits later. Measured
# before this test existed: fourteen `*_quietly(...)` calls stood as
# bare statements across the three modules, every one of them dropping
# a close failure on the floor.
# =====================================================================

def _agent_loop_kaynaklari():
    import tools.agent_loop.fs_evidence as _ev
    import tools.agent_loop.fs_transport_posix as _px
    from pathlib import Path as _P
    moduller = [fs_transport, _ev, _px]
    if _WINDOWS:
        moduller.append(_win)
    return [(m.__name__, _P(m.__file__).read_text(encoding="utf-8"))
            for m in moduller]


def test_no_quiet_cleanup_result_is_discarded():
    """Every `*_quietly(...)` value must be USED -- as a condition, an
    argument, an assignment or a return. A bare call statement means the
    close failure had nowhere to go."""
    import ast

    atilan = []
    for ad, kaynak in _agent_loop_kaynaklari():
        for dugum in ast.walk(ast.parse(kaynak)):
            if not isinstance(dugum, ast.Expr):
                continue
            if not isinstance(dugum.value, ast.Call):
                continue
            islev = dugum.value.func
            isim = getattr(islev, "id", getattr(islev, "attr", ""))
            if isim.endswith("_quietly"):
                atilan.append(f"{ad}:{dugum.lineno} {isim}")
    assert not atilan, f"sessiz kapanis sonucu atilan yerler: {atilan}"


def test_every_quiet_cleanup_helper_actually_returns_a_verdict():
    """A helper that returns nothing cannot have its result consumed,
    so the check above would pass for the wrong reason."""
    import ast

    sessizler = 0
    for ad, kaynak in _agent_loop_kaynaklari():
        for dugum in ast.walk(ast.parse(kaynak)):
            if not isinstance(dugum, ast.FunctionDef):
                continue
            if not dugum.name.endswith("_quietly"):
                continue
            sessizler += 1
            donusler = [d for d in ast.walk(dugum)
                        if isinstance(d, ast.Return) and d.value is not None]
            assert donusler, f"{ad}:{dugum.name} deger dondurmuyor"
    assert sessizler >= 3, f"yalniz {sessizler} sessiz yardimci bulundu"


def test_no_walker_seam_lets_a_raw_os_error_through():
    """Structural companion to the behavioural tests: every call the
    walker makes into the transport goes through the one guard."""
    import ast
    import tools.agent_loop.fs_evidence as _ev
    from pathlib import Path as _P

    kaynak = _P(_ev.__file__).read_text(encoding="utf-8")
    agac = ast.parse(kaynak)

    # A `*_quietly` helper is allowed to call the transport directly:
    # catching both error kinds is its entire job, and it is the one
    # place where that is written down.
    muaf = set()
    for dugum in ast.walk(agac):
        if isinstance(dugum, ast.FunctionDef) and \
                dugum.name.endswith("_quietly"):
            muaf.update(id(alt) for alt in ast.walk(dugum))

    # Pure helpers that touch no filesystem and cannot raise `OSError`.
    # Named one by one on purpose: anything NEW appearing on
    # `fs_transport` is a seam until someone says otherwise, which is
    # the direction this check should fail in.
    SAF = {"mark_cleanup_failed", "cleanup_failed", "validate_child_name"}

    korumasiz = []
    for dugum in ast.walk(agac):
        if not isinstance(dugum, ast.Call) or id(dugum) in muaf:
            continue
        islev = dugum.func
        if not isinstance(islev, ast.Attribute):
            continue
        if not isinstance(islev.value, ast.Name):
            continue
        if islev.value.id != "fs_transport" or islev.attr in SAF:
            continue
        # `_guard` receives the callable itself and never calls it here,
        # so a direct call means a seam that bypassed the guard
        korumasiz.append(f"{islev.attr}:{dugum.lineno}")
    assert not korumasiz, \
        f"guard disinda dogrudan transport cagrisi: {korumasiz}"


# =====================================================================
# DESCRIPTOR SETUP: PRIMARY ERROR x CLEANUP ERROR
#
# Between `open_osfhandle` and the returned descriptor there are three
# refusals, and each one used to call a close that raised. Measured on
# all three: PRIMARY_SURVIVED=False -- the caller was told the
# descriptor would not close, and never told why the file was refused.
# =====================================================================

@pytest.mark.skipif(not _WINDOWS, reason="descriptor kurulumu Windows'a ozgu")
@pytest.mark.parametrize("yol,beklenen", [
    ("fstat", "acik nesne durumu okunamadi"),
    ("reparse", "son bilesen yeniden ayrisma noktasi"),
    ("tur", "son bilesen siradan bir dosya degil"),
])
@pytest.mark.parametrize("kapanis_bozuk", [False, True])
def test_a_descriptor_setup_refusal_keeps_its_primary_error(
        tmp_path, monkeypatch, yol, beklenen, kapanis_bozuk):
    """Six cases: three refusals, each with a working and a failing
    close. The primary error is the same in all six.

    THE TEST CLEANS UP AFTER ITS OWN SABOTAGE. In the failing-close
    cases `sahte_close` raises INSTEAD of closing, which is the whole
    point -- and it means the descriptor is genuinely still open when
    the patch is undone. Leaving it there would have this package's
    handle-lifetime test leaking three descriptors per run, so the
    sabotaged ones are closed for real afterwards, and only those."""
    taban = _kaynak_sayisi()
    kok = _agac(tmp_path)
    directory = fs_transport.open_root(kok)
    kayit = _kayitlar(directory)["a.py"]

    class SahteStat:
        st_mode = 0o040000 if yol == "tur" else 0o100644
        st_file_attributes = 0x400 if yol == "reparse" else 0x20

    gercek_close = os.close
    kapatilan = []
    sabote_edilen = []          # still open BECAUSE this test broke it

    def sahte_fstat(fd):
        if yol == "fstat":
            raise OSError(f"{SENTINEL} C:" + chr(92) + "gizli")
        return SahteStat()

    def sahte_close(fd):
        kapatilan.append(fd)
        if kapanis_bozuk:
            sabote_edilen.append(fd)
            raise OSError(f"{SENTINEL} kapatma C:" + chr(92) + "gizli")
        return gercek_close(fd)

    monkeypatch.setattr(os, "fstat", sahte_fstat)
    monkeypatch.setattr(os, "close", sahte_close)
    try:
        with pytest.raises(fs_transport.TransportError) as ret:
            fs_transport.open_child_file(directory, kayit)
    finally:
        monkeypatch.undo()
        # only the sabotaged ones: a healthy case already closed its
        # descriptor through the real call, and closing again would be
        # a double close on a number the process may have reused
        for fd in sabote_edilen:
            gercek_close(fd)
        fs_transport.close_directory(directory)

    assert kapatilan, "senaryo kurulmadi: descriptor hic kapatilmaya calisilmadi"
    assert bool(sabote_edilen) is kapanis_bozuk, \
        "senaryo kurulmadi: sabotaj gercek durumla ortusmuyor"
    assert _kaynak_sayisi() <= taban, \
        "test kendi actigi kaynagi birakti"
    assert str(ret.value) == beklenen, \
        f"birincil hata ezildi: {str(ret.value)!r}"
    assert fs_transport.cleanup_failed(ret.value) is kapanis_bozuk, \
        "temizlik isareti gercek duruma uymuyor"
    metin = str(ret.value) + repr(ret.value) + \
        " ".join(getattr(ret.value, "__notes__", []))
    assert SENTINEL not in metin, "ham isletim sistemi metni sizdi"
    assert "\\" not in metin and "/" not in metin


def test_no_close_is_swallowed_anywhere_in_these_modules():
    """`_unused_close_descriptor` was dead code doing `except OSError:
    pass`, and the earlier mechanical test could not see it because that
    test only looked at names ending in `_quietly`. The rule is now
    about the SHAPE, so a differently named helper cannot hide."""
    import ast

    yutan = []
    for ad, kaynak in _agent_loop_kaynaklari():
        for dugum in ast.walk(ast.parse(kaynak)):
            if not isinstance(dugum, ast.ExceptHandler):
                continue
            if len(dugum.body) == 1 and isinstance(dugum.body[0], ast.Pass):
                yutan.append(f"{ad}:{dugum.lineno}")
    assert not yutan, f"sessizce yutulan hata yollari: {yutan}"
