"""PACKAGE B2B-A-D3A / R1B.1 -- bounded and contained Git transport.

WHAT THIS FILE IS ABOUT. Not what git said, but whether the process
that said it is finished, its readers are finished, and its container
is empty. MEASURED on the transport this replaced: `capture_output=True`
read 50,331,648 bytes from a 48 MiB producer with no cap, and a run
whose grandchild held the pipe never returned at all.

NO REAL GIT IS NEEDED. Every program below is a disposable stand-in
written into `tmp_path`. The producer announces how much it intends to
write and only writes its finish marker at the very end, so a transport
that stopped it early can be told apart from one that read everything
and complained afterwards.

EVERY NEGATIVE TEST PROVES ITS INJECTION POINT WAS REACHED. A refusal
that came from an unrelated setup failure is a green test measuring
nothing, which is how three earlier rounds ended.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time

import pytest

from tools.agent_loop import git_transport
from tools.agent_loop import process as process_mod

SENTINEL = "KURGU-GIZLI-NOBETCI-" + "z" * 8
# a raw message shaped like the ones that actually leak: a sentinel and
# a path, both of which belong to the machine and not to a caller
HAM_METIN = SENTINEL + " C:" + chr(92) + "gizli" + chr(92) + "yol"

_URETICI_KAYNAK = "\n".join([
    '# A disposable stand-in for git. It announces how much it will',
    '# write, then writes it -- so a transport that stops it early can',
    '# be told apart from one that let it finish.',
    'import os',
    'import sys',
    'import time',
    '',
    "NEREYE = os.environ['URETICI_ISARET']",
    "BOYUT = int(os.environ['URETICI_BOYUT'])",
    "AKIS = os.environ.get('URETICI_AKIS', 'stdout')",
    "COCUK = os.environ.get('URETICI_COCUK') == '1'",
    "BEKLE = float(os.environ.get('URETICI_BEKLE', '0'))",
    "CIKIS = int(os.environ.get('URETICI_CIKIS', '0'))",
    '',
    'if COCUK:',
    '    import subprocess',
    "    subprocess.Popen([sys.executable, '-c',",
    "                      'import time; time.sleep(120)'])",
    '',
    "hedef = sys.stdout.buffer if AKIS == 'stdout' else sys.stderr.buffer",
    "parca = b'K' * 65536",
    'yazilan = 0',
    'while yazilan < BOYUT:',
    '    n = min(len(parca), BOYUT - yazilan)',
    '    hedef.write(parca[:n])',
    '    yazilan += n',
    'hedef.flush()',
    '',
    'if BEKLE:',
    '    time.sleep(BEKLE)',
    '',
    '# only reached if nobody stopped us',
    "with open(NEREYE, 'w', encoding='ascii') as akis:",
    '    akis.write(str(yazilan))',
    'sys.exit(CIKIS)',
])


def _uretici(tmp_path, *, boyut, akis="stdout", cocuk=False, bekle=0.0,
             cikis=0):
    """A disposable fake git, plus the environment that drives it."""
    betik = tmp_path / "sahte_git.py"
    betik.write_text(_URETICI_KAYNAK, encoding="utf-8")
    isaret = tmp_path / "uretici-bitti.txt"
    ortam = {
        "URETICI_ISARET": str(isaret), "URETICI_BOYUT": str(boyut),
        "URETICI_AKIS": akis, "URETICI_COCUK": "1" if cocuk else "0",
        "URETICI_BEKLE": str(bekle), "URETICI_CIKIS": str(cikis),
    }
    return [sys.executable, str(betik)], ortam, isaret


# The producer above never reads stdin, which is exactly right for every
# read that asks git a question. PACKAGE B6-R1 added the one call that
# HANDS git bytes -- `hash-object --stdin` -- so the stand-in below is
# its counterpart: it consumes stdin and records the DIGEST of what
# arrived, which is the only way to tell a payload that was delivered
# whole from one that was quietly cut in half.
_TUKETICI_KAYNAK = "\n".join([
    '# A disposable stand-in that CONSUMES stdin and reports what it got.',
    'import hashlib',
    'import os',
    'import sys',
    'import time',
    '',
    "NEREYE = os.environ['TUKETICI_ISARET']",
    "OKU = os.environ.get('TUKETICI_OKU', '1') == '1'",
    "BEKLE = float(os.environ.get('TUKETICI_BEKLE', '0'))",
    "CIKIS = int(os.environ.get('TUKETICI_CIKIS', '0'))",
    "YAZ = os.environ.get('TUKETICI_YAZ', '')",
    '',
    'if BEKLE:',
    '    time.sleep(BEKLE)',
    '',
    'alinan = b""',
    'if OKU:',
    '    alinan = sys.stdin.buffer.read()',
    '',
    'if YAZ:',
    "    sys.stdout.buffer.write(YAZ.encode('ascii'))",
    'else:',
    "    sys.stdout.buffer.write(hashlib.sha256(alinan).hexdigest()",
    "                            .encode('ascii'))",
    'sys.stdout.buffer.flush()',
    '',
    "with open(NEREYE, 'w', encoding='ascii') as akis:",
    "    akis.write(f'{len(alinan)}:{hashlib.sha256(alinan).hexdigest()}')",
    'sys.exit(CIKIS)',
])


def _tuketici(tmp_path, *, oku=True, bekle=0.0, cikis=0, yaz=""):
    betik = tmp_path / "sahte_git_tuketici.py"
    betik.write_text(_TUKETICI_KAYNAK, encoding="utf-8")
    isaret = tmp_path / "tuketici-bitti.txt"
    ortam = {"TUKETICI_ISARET": str(isaret), "TUKETICI_OKU":
             "1" if oku else "0", "TUKETICI_BEKLE": str(bekle),
             "TUKETICI_CIKIS": str(cikis), "TUKETICI_YAZ": yaz}
    return [sys.executable, str(betik)], ortam, isaret


@pytest.fixture
def kur(tmp_path, monkeypatch):
    """Install the producer's environment on the transport's own seam.

    The transport builds the child's environment itself, so this is the
    only place a test can reach it -- and doing it through `monkeypatch`
    means a failing test cannot leave the seam replaced."""
    def kurulum(**kwargs):
        argv, ortam, isaret = _uretici(tmp_path, **kwargs)
        monkeypatch.setattr(git_transport, "_git_env",
                            lambda: {**os.environ, **ortam})
        return argv, isaret
    return kurulum


@pytest.fixture
def kur_tuketici(tmp_path, monkeypatch):
    """The consumer's environment, installed on the same seam as `kur`."""
    def kurulum(**kwargs):
        argv, ortam, isaret = _tuketici(tmp_path, **kwargs)
        monkeypatch.setattr(git_transport, "_git_env",
                            lambda: {**os.environ, **ortam})
        return argv, isaret
    return kurulum


@pytest.fixture(autouse=True)
def izlenen_surecler(monkeypatch):
    """Every process this file starts, and proof that none survived.

    The transport hands its `Popen` to nobody, so the pipes are captured
    here as well: an injection that stops a reader from ever being built
    leaves two open descriptors that belong to this test, not to the
    module under test."""
    kayit = []
    gercek = process_mod.launch_contained

    def sarmalayan(argv, *, cwd, env=None):
        proc, container = gercek(argv, cwd=cwd, env=env)
        kayit.append(proc)
        return proc, container

    monkeypatch.setattr(process_mod, "launch_contained", sarmalayan)
    yield kayit
    for proc in kayit:
        for akis in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if akis is not None:
                    akis.close()
            except (OSError, ValueError):
                pass
    canli = [proc.pid for proc in kayit if proc.poll() is None]
    assert canli == [], f"testten sonra yasayan surec: {canli}"


def _tum_metin(hata):
    """Everything a caller could print: the message, the repr, the notes
    and the whole chained cause/context chain."""
    parcalar, dugum, derinlik = [], hata, 0
    while dugum is not None and derinlik < 10:
        parcalar.append(str(dugum))
        parcalar.append(repr(dugum))
        parcalar.extend(getattr(dugum, "__notes__", None) or [])
        dugum = dugum.__cause__ or dugum.__context__
        derinlik += 1
    return " ".join(parcalar)


# =====================================================================
# THE CEILING IS A MEMORY BOUND, NOT A REPORT
# =====================================================================

def test_output_exactly_at_the_ceiling_is_accepted(kur, tmp_path):
    """The boundary belongs to the accepted side. An off-by-one here is
    a transport that refuses correct output, which is the failure mode
    nobody writes a test for."""
    tavan = 4096
    argv, isaret = kur(boyut=tavan)
    cikti = git_transport.run_git_bounded(argv, cwd=tmp_path,
                                          stdout_limit=tavan, timeout=60)
    assert len(cikti) == tavan
    assert cikti == b"K" * tavan
    assert isaret.exists(), "senaryo kurulmadi: uretici bitmedi"


def test_one_byte_past_the_ceiling_is_refused(kur, tmp_path):
    tavan = 4096
    argv, _ = kur(boyut=tavan + 1)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=tavan, timeout=60)
    assert "tavan" in str(ret.value)


def test_a_producer_beyond_the_ceiling_is_stopped_and_refused(kur, tmp_path,
                                                              monkeypatch):
    """THE measurement this round exists for.

    MEASURED on the old transport: 8,388,608 bytes entered memory
    against a 1 MiB ceiling, because `capture_output=True` reads
    everything before any application limit is consulted. Bounded now:
    the retained buffer cannot pass the ceiling, the reader may see at
    most one chunk beyond it, and the producer never reaches its
    finish marker."""
    tavan = 1 << 20
    argv, isaret = kur(boyut=16 << 20)
    olcum = {}

    class Olcen(process_mod.BoundedStream):
        def run(self):
            super().run()
            if self.label == "stdout":
                olcum["gorulen"] = self.total
                olcum["tutulan"] = len(self.buffer)
                olcum["sonuc"] = self.outcome

    monkeypatch.setattr(process_mod, "BoundedStream", Olcen)
    basla = time.monotonic()
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=tavan, timeout=60)
    gecen = time.monotonic() - basla

    assert olcum.get("sonuc") == process_mod.READ_OVERFLOWED, \
        "senaryo kurulmadi: tavan asilmadi"
    assert olcum["tutulan"] <= tavan, \
        f"tutulan {olcum['tutulan']} > tavan {tavan}"
    assert olcum["gorulen"] <= tavan + process_mod.READ_CHUNK_BYTES, \
        f"gorulen {olcum['gorulen']} bir parcadan fazla asti"
    assert not isaret.exists(), \
        "uretici planladigi ciktiyi bitirdi -- erken durdurulmadi"
    assert "tavan" in str(ret.value)
    assert gecen < 60


def test_a_noisy_stderr_is_bounded_and_never_returned(kur, tmp_path):
    """stderr exists here only to classify a failure. MEASURED on the
    old transport: 4,194,304 bytes retained."""
    argv, _ = kur(boyut=4 << 20, akis="stderr")
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=60)
    assert "K" * 64 not in _tum_metin(ret.value), "stderr icerigi disari cikti"
    assert git_transport.STDERR_CEILING <= 64 << 10


@pytest.mark.parametrize("kotu", [0, -1, True, float("nan"),
                                  float("inf"), "64", None])
def test_an_invalid_ceiling_is_refused_before_launch(kur, tmp_path, kotu):
    """Validated BEFORE anything starts: `True` is an `int` and `NaN`
    compares false to every bound, so either would silently become a
    ceiling nobody chose."""
    argv, isaret = kur(boyut=16)
    with pytest.raises(git_transport.FlatWorkspaceError):
        git_transport.run_git_bounded(argv, cwd=tmp_path, stdout_limit=kotu,
                                      timeout=10)
    assert not isaret.exists(), "gecersiz tavana ragmen program calisti"


# =====================================================================
# A READER THAT FAILED IS NOT A SHORT ANSWER
# =====================================================================

class _KopukAkis:
    """A pipe that breaks at the END of the read, in raw machine words.

    Deliberately not at the first read: closing the pipe under a
    producer that has not written yet kills the producer too, and the
    scenario would then be "git died", which the transport already
    refuses for an unrelated reason. Here git finishes, exits ZERO, and
    only the reader fails -- which is the case that used to pass a
    truncated buffer off as the whole output."""

    def __init__(self, gercek):
        self._gercek = gercek
        self.patladi = False

    def read(self, n):
        parca = self._gercek.read(n)
        if parca:
            return parca
        self.patladi = True
        raise OSError(HAM_METIN)

    def close(self):
        self._gercek.close()


def test_a_failed_reader_is_refused_even_when_git_exits_zero(kur, tmp_path,
                                                             monkeypatch):
    """The gap the outcome exists to close. A pipe that broke mid-read
    left a truncated buffer beside exit code 0, and the truncation
    became the object's bytes."""
    argv, isaret = kur(boyut=16, cikis=0)
    akislar, okuyucular = [], []

    class Kopuk(process_mod.BoundedStream):
        def __init__(self, label, stream, limit, tripped):
            if label == "stdout":
                stream = _KopukAkis(stream)
                akislar.append(stream)
            super().__init__(label, stream, limit, tripped)
            okuyucular.append(self)

    monkeypatch.setattr(process_mod, "BoundedStream", Kopuk)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=60)

    assert akislar and akislar[0].patladi, "kirik okumaya hic ulasilmadi"
    assert isaret.exists(), \
        "senaryo kurulmadi: uretici sifirla bitmedi, ret baska sebepten"
    stdout_okuyucu = [o for o in okuyucular if o.label == "stdout"][0]
    assert stdout_okuyucu.outcome == process_mod.READ_FAILED
    assert bytes(stdout_okuyucu.buffer) == b"K" * 16, \
        "senaryo kurulmadi: kirilma once degil, okumanin sonunda olmali"
    assert "eksiksiz okunamadi" in str(ret.value)
    assert HAM_METIN not in _tum_metin(ret.value)


def test_a_reader_that_raises_leaves_no_traceback_and_a_closed_outcome():
    """Directly on the reader, with no process at all: an exception that
    escaped this thread would print a traceback carrying paths and STILL
    leave the reader looking like a normal short read."""
    import threading

    class Patlayan:
        def read(self, n):
            raise RuntimeError(HAM_METIN)

        def close(self):
            pass

    okuyucu = process_mod.BoundedStream("stdout", Patlayan(), 1024,
                                        threading.Event())
    okuyucu.start()
    okuyucu.join(timeout=10)
    assert not okuyucu.is_alive()
    assert okuyucu.outcome == process_mod.READ_FAILED
    assert okuyucu.overflowed is False
    assert bytes(okuyucu.buffer) == b""


# =====================================================================
# CLEANUP IS COMPLETE, NON-ABORTABLE, AND OUTRANKS THE OUTPUT
# =====================================================================

def test_a_timeout_leaves_no_parent_and_no_child(kur, tmp_path):
    """MEASURED on the old transport: the call never returned at all --
    the grandchild held the pipe open until the 120-second timeout.

    That the refusal is the TIMEOUT one and not `git sureci
    temizlenemedi` is the proof the container was shown empty; the
    autouse fixture is the proof for the parent."""
    argv, isaret = kur(boyut=1024, cocuk=True, bekle=90.0)
    basla = time.monotonic()
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=5)
    gecen = time.monotonic() - basla
    assert gecen < 40, f"kapsayici agaci birakmadi: {gecen:.1f}s"
    assert "zaman asimina" in str(ret.value)
    assert not isaret.exists(), "uretici bitirdi -- durdurulmadi"


@pytest.mark.parametrize("nerede", ["bosaltma", "birlestirme"])
def test_an_unproven_cleanup_step_refuses_the_whole_run(kur, tmp_path,
                                                        monkeypatch, nerede):
    """A run whose output is perfect is still refused if the container
    or the readers cannot be shown finished. The real operation still
    happens underneath -- only its VERDICT is forced -- so the injection
    tests the decision and not the machine."""
    argv, isaret = kur(boyut=64)
    ulasildi = {"evet": False}

    if nerede == "bosaltma":
        gercek_drain = process_mod.Container.drain

        def sahte_drain(self, deadline=None):
            ulasildi["evet"] = True
            gercek_drain(self, deadline)
            return False

        monkeypatch.setattr(process_mod.Container, "drain", sahte_drain)
    else:
        gercek_join = process_mod.join_within

        def sahte_join(threads, deadline):
            ulasildi["evet"] = True
            gercek_join(threads, deadline)
            return False

        monkeypatch.setattr(process_mod, "join_within", sahte_join)

    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=60)
    assert ulasildi["evet"], "hedef temizlik adimina ulasilmadi"
    assert isaret.exists(), \
        "senaryo kurulmadi: uretici bitmedi, ret baska sebepten"
    assert str(ret.value) == "git sureci temizlenemedi"


def test_a_throwing_cleanup_step_does_not_skip_the_ones_after_it(kur,
                                                                 tmp_path,
                                                                 monkeypatch):
    """The gap that made the previous envelope only look complete. The
    steps a `finally` chain would have skipped are precisely the ones
    that empty the container and join the readers -- so an exception in
    the first step used to leave a live process behind and report a
    tidy refusal."""
    argv, _ = kur(boyut=64, bekle=90.0)
    gorulen = []

    def patlayan_stop(process, deadline):
        gorulen.append("durdur")
        raise OSError(HAM_METIN)

    gercek_drain = process_mod.Container.drain

    def izlenen_drain(self, deadline=None):
        gorulen.append("bosalt")
        return gercek_drain(self, deadline)

    gercek_join = process_mod.join_within

    def izlenen_join(threads, deadline):
        gorulen.append("birlestir")
        return gercek_join(threads, deadline)

    gercek_wait = subprocess.Popen.wait

    def izlenen_wait(self, timeout=None):
        gorulen.append("bekle")
        return gercek_wait(self, timeout=timeout)

    monkeypatch.setattr(process_mod, "stop", patlayan_stop)
    monkeypatch.setattr(process_mod.Container, "drain", izlenen_drain)
    monkeypatch.setattr(process_mod, "join_within", izlenen_join)
    monkeypatch.setattr(subprocess.Popen, "wait", izlenen_wait)

    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=2)

    assert gorulen[0] == "durdur", "senaryo kurulmadi: enjeksiyona ulasilmadi"
    assert gorulen == ["durdur", "bosalt", "birlestir", "bekle"], \
        f"temizlik adimlari atlandi: {gorulen}"
    assert str(ret.value) == "git sureci temizlenemedi"


@pytest.mark.parametrize("nerede", ["tasima", "temizlik"])
def test_no_raw_process_text_survives_a_refusal(kur, tmp_path, monkeypatch,
                                                nerede):
    """Both directions of the leak. `tasima` raises the raw error while
    the transport is reading; `temizlik` raises it while cleanup is
    running, which is the case that used to travel out through
    `__context__` because the refusal was raised inside the handler."""
    ulasildi = {"evet": False}

    if nerede == "tasima":
        argv, _ = kur(boyut=16)

        class Kurulamayan(process_mod.BoundedStream):
            def __init__(self, label, stream, limit, tripped):
                ulasildi["evet"] = True
                raise OSError(HAM_METIN)

        monkeypatch.setattr(process_mod, "BoundedStream", Kurulamayan)
        zaman = 60
    else:
        argv, _ = kur(boyut=64, bekle=90.0)

        def patlayan_stop(process, deadline):
            ulasildi["evet"] = True
            raise OSError(HAM_METIN)

        monkeypatch.setattr(process_mod, "stop", patlayan_stop)
        zaman = 2

    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=zaman)

    assert ulasildi["evet"], "hedef enjeksiyona ulasilmadi"
    metin = _tum_metin(ret.value)
    assert SENTINEL not in metin, "ham surec metni istisnaya tasindi"
    assert "/" not in metin and chr(92) not in metin, "yol disari cikti"
    assert ret.value.__cause__ is None
    assert ret.value.__context__ is None, "ham hata baglam olarak tasindi"


def test_a_nonzero_exit_is_refused_without_its_stderr(kur, tmp_path):
    """The command failed; what it had to say about why is git's own
    text and carries paths, remotes and credential-helper complaints."""
    argv, isaret = kur(boyut=256, akis="stderr", cikis=3)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20, timeout=60)
    assert isaret.exists(), "senaryo kurulmadi: uretici cikisa ulasmadi"
    assert str(ret.value) == "git komutu basarisiz"
    assert "K" not in _tum_metin(ret.value)


# =====================================================================
# BOUNDED STDIN. PACKAGE B6-R1.
#
# The transport used to close stdin unconditionally, which is right for
# every call that ASKS git something. `hash-object --stdin` is the first
# that TELLS git something, and a payload delivered in part is not a
# short answer -- it is an answer about different bytes than the caller
# asked about. Every test below therefore proves what the child actually
# received, not merely that the call returned.
# =====================================================================

def test_bounded_stdin_arrives_at_the_child_whole(kur_tuketici, tmp_path):
    """Deliberately larger than any pipe buffer.

    A single `write` on a raw stream moves what it can and returns that
    count without raising -- the defect `PromptWriter` was rewritten for.
    A payload that fits in the buffer would pass either way, so this one
    does not fit."""
    yuk = bytes(range(256)) * 16384                      # 4 MiB
    argv, isaret = kur_tuketici()
    cikti = git_transport.run_git_bounded(argv, cwd=tmp_path,
                                          stdout_limit=1 << 20,
                                          stdin_bytes=yuk, timeout=60)
    beklenen = hashlib.sha256(yuk).hexdigest()
    assert cikti.decode("ascii") == beklenen
    assert isaret.read_text(encoding="ascii") == f"{len(yuk)}:{beklenen}"


def test_stdin_stays_closed_when_no_payload_is_given(kur_tuketici, tmp_path):
    """The behaviour every existing caller depends on. A child that reads
    stdin must see EOF immediately, not block waiting for a writer that
    was never asked for."""
    argv, isaret = kur_tuketici()
    cikti = git_transport.run_git_bounded(argv, cwd=tmp_path,
                                          stdout_limit=1 << 20, timeout=60)
    bos = hashlib.sha256(b"").hexdigest()
    assert cikti.decode("ascii") == bos
    assert isaret.read_text(encoding="ascii") == f"0:{bos}"


def test_an_empty_payload_is_a_payload_and_not_a_closed_pipe(kur_tuketici,
                                                             tmp_path):
    """`b""` and `None` are different requests. MEASURED in Task A: the
    empty blob is a legal object with a real id, so an empty payload must
    reach git as an empty stream rather than be treated as "no stdin"."""
    argv, isaret = kur_tuketici()
    cikti = git_transport.run_git_bounded(argv, cwd=tmp_path,
                                          stdout_limit=1 << 20,
                                          stdin_bytes=b"", timeout=60)
    assert cikti.decode("ascii") == hashlib.sha256(b"").hexdigest()
    assert isaret.exists(), "senaryo kurulmadi: tuketici bitmedi"


class _KisaAkis:
    """A raw stream that accepts a fixed prefix and then stops accepting.

    THE EXACT MACHINE BEHAVIOUR the writer was rewritten for: `bufsize=0`
    makes stdin raw, and a raw write moves what it can and RETURNS THAT
    COUNT without raising. Nothing about it looks like an error."""

    def __init__(self, gercek, kabul):
        self._gercek = gercek
        self._kalan = kabul

    def write(self, data):
        if self._kalan <= 0:
            return 0                       # the pipe stopped accepting
        n = min(self._kalan, len(data))
        self._gercek.write(bytes(data[:n]))
        self._kalan -= n
        return n

    def flush(self):
        self._gercek.flush()

    def close(self):
        self._gercek.close()


def test_a_partially_written_stdin_is_refused_even_when_git_exits_zero(
        kur_tuketici, tmp_path, monkeypatch):
    """THE gap this parameter opens, closed.

    The pipe takes 64 bytes of a 4096-byte payload and then stops. The
    child reads those 64 bytes, computes a perfectly valid digest OF THE
    WRONG BYTES and exits ZERO. Nothing in the return code, the output
    ceiling or the container says anything is wrong -- and the answer
    would have been recorded as the object id of a file nobody has."""
    yuk = b"A" * 4096
    kabul = 64
    argv, isaret = kur_tuketici()
    kesildi = {"evet": False}
    gercek = process_mod.PromptWriter

    class Yarim(gercek):
        def __init__(self, stream, payload):
            kesildi["evet"] = True
            super().__init__(_KisaAkis(stream, kabul), payload)

    monkeypatch.setattr(process_mod, "PromptWriter", Yarim)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=yuk, timeout=60)
    assert kesildi["evet"], "senaryo kurulmadi: yazici degistirilmedi"
    # the child really did finish, and really did see only the prefix
    assert isaret.read_text(encoding="ascii").startswith(f"{kabul}:")
    assert str(ret.value) == "git girdisi tam yazilamadi"


def test_a_writer_that_never_completes_is_refused(kur_tuketici, tmp_path,
                                                  monkeypatch):
    """The flag is asked AFTER the join, so a writer that is merely slow
    is not mistaken for one that failed -- and one that truly never
    finished is not accepted because the return code looked fine."""
    argv, isaret = kur_tuketici()
    gercek = process_mod.PromptWriter

    class HicYazmaz(gercek):
        def run(self):
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass
            # `completed` deliberately left False

    monkeypatch.setattr(process_mod, "PromptWriter", HicYazmaz)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=b"X" * 32, timeout=60)
    assert isaret.exists(), "senaryo kurulmadi: tuketici hic calismadi"
    assert str(ret.value) == "git girdisi tam yazilamadi"


@pytest.mark.parametrize("kotu", ["metin", bytearray(b"ab"),
                                  memoryview(b"ab"), 5, [1, 2]])
def test_a_payload_that_is_not_bytes_is_refused_before_launch(
        kur_tuketici, tmp_path, kotu):
    """A `str` would be encoded by somebody's default and a `memoryview`
    would make the payload's lifetime somebody else's problem. Checked
    before the container exists, so nothing runs."""
    argv, isaret = kur_tuketici()
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=kotu, timeout=60)
    assert not isaret.exists(), "gecersiz yuke ragmen program calisti"
    assert str(ret.value) == "git girdisi bayt dizisi degil"


def test_a_child_that_ignores_a_large_payload_neither_hangs_nor_survives(
        kur_tuketici, tmp_path):
    """The writer parks in `write` until the pipe closes, which is fine
    HERE and was fatal on the main path. What must hold is that the
    deadline still governs, the run is refused rather than accepted on a
    partial write, and nothing is left alive -- the autouse fixture
    proves the last part."""
    argv, isaret = kur_tuketici(oku=False)
    basla = time.monotonic()
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=b"B" * (8 << 20), timeout=20)
    gecen = time.monotonic() - basla
    assert gecen < 40, f"cagri deadline'i asti: {gecen:.1f}s"
    assert isaret.exists(), "senaryo kurulmadi: tuketici hic calismadi"
    assert str(ret.value) == "git girdisi tam yazilamadi"


def test_a_timeout_with_a_payload_leaves_no_parent_and_no_child(
        kur_tuketici, tmp_path):
    """A stalled child must not become a call that never returns just
    because there is now a writer thread in the picture."""
    argv, isaret = kur_tuketici(bekle=90.0)
    basla = time.monotonic()
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=b"C" * 64, timeout=5)
    gecen = time.monotonic() - basla
    assert gecen < 40, f"kapsayici birakmadi: {gecen:.1f}s"
    assert "zaman asimina" in str(ret.value)
    assert not isaret.exists(), "tuketici bitirdi -- durdurulmadi"


def test_the_output_ceiling_still_governs_while_stdin_is_written(
        kur_tuketici, tmp_path):
    """Two bounds, both live at once. A payload going in must not buy the
    child an unbounded channel coming out."""
    argv, _ = kur_tuketici(yaz="D" * 4096)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path, stdout_limit=64,
                                      stdin_bytes=b"E" * 128, timeout=60)
    assert "tavan" in str(ret.value)


def test_a_nonzero_exit_with_a_payload_is_refused_without_its_stderr(
        kur_tuketici, tmp_path):
    argv, isaret = kur_tuketici(cikis=4)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=b"F" * 32, timeout=60)
    assert isaret.exists(), "senaryo kurulmadi: tuketici cikisa ulasmadi"
    assert str(ret.value) == "git komutu basarisiz"


def test_no_payload_byte_survives_a_stdin_refusal(kur_tuketici, tmp_path,
                                                  monkeypatch):
    """The payload is a file of the operator's. It is the LAST thing that
    may appear in a refusal's text, its cause or its context."""
    yuk = SENTINEL.encode("ascii") * 8
    argv, _ = kur_tuketici(cikis=5)
    with pytest.raises(git_transport.FlatWorkspaceError) as ret:
        git_transport.run_git_bounded(argv, cwd=tmp_path,
                                      stdout_limit=1 << 20,
                                      stdin_bytes=yuk, timeout=60)
    metin = _tum_metin(ret.value)
    assert SENTINEL not in metin, "yuk disari cikti"
    assert ret.value.__cause__ is None
    assert ret.value.__context__ is None, "ham hata baglam olarak tasindi"


def test_git_bytes_forwards_the_payload_and_defaults_to_no_stdin(monkeypatch,
                                                                 tmp_path):
    """The named helper every caller reaches for, not just the low-level
    seam. A wrapper that dropped `stdin_bytes` on the floor would send
    git an EMPTY stream and hash the wrong thing -- silently, because the
    empty blob is a perfectly valid object."""
    gorulen = []

    def sahte(argv, *, cwd, stdout_limit, stdin_bytes=None,
              timeout=git_transport._GIT_TIMEOUT_SECONDS):
        gorulen.append((tuple(argv), stdin_bytes))
        return b"tamam"

    monkeypatch.setattr(git_transport, "run_git_bounded", sahte)
    git_transport.git_bytes(tmp_path, "cat-file", "-t", "x", stdout_limit=8)
    git_transport.git_bytes(tmp_path, "hash-object", "--stdin",
                            stdout_limit=41, stdin_bytes=b"YUK")

    assert gorulen[0][1] is None, "girdisiz cagri bir yuk uydurdu"
    assert gorulen[1][1] == b"YUK", "yuk tasimaya gecmedi"
    assert gorulen[1][0][:3] == ("git", "-C", str(tmp_path))


def test_the_git_environment_carries_its_isolation_flags():
    """The flags exist for measured reasons: replacement objects would
    let a ref decide which bytes a baseline resolves to, and the prompt
    settings keep a credential helper from ever blocking a scan."""
    ortam = git_transport._git_env()
    assert ortam["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert ortam["GIT_TERMINAL_PROMPT"] == "0"
    assert ortam["GIT_ASKPASS"] == ""
    assert ortam["GCM_INTERACTIVE"] == "never"
    assert ortam["GIT_CONFIG_NOSYSTEM"] == "1"
