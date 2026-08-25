"""The tracked leak scanner must find planted leaks, refuse blind spots,
and never echo a secret.

Everything here is synthetic: an invented corpus, invented repo files, and
a secret assembled at RUNTIME from halves -- writing a complete credential
pattern into a test fixture would trip the very battery this tool encodes.
"""
import pytest

from eval.tools import leak_scan
from eval.tools.leak_scan import build_corpora, fold, scan

CORPUS_SENTENCE = ("Kurgu Omega Kurumu kurgusal donem ozetinde Zeta Endeksi "
                   "47 000 birim olarak yazilmistir.")


@pytest.fixture
def veri(tmp_path):
    data_dir = tmp_path / "veri"
    data_dir.mkdir()
    (data_dir / "kurgu-belge.txt").write_text(CORPUS_SENTENCE,
                                              encoding="utf-8")
    (data_dir / "ikinci.txt").write_text("Ikinci kurgu icerik parcasi.",
                                         encoding="utf-8")
    return data_dir


def _corpora(data_dir, output_dir=None):
    documents, full, contributed = build_corpora(data_dir, output_dir)[:3]
    return documents, full, contributed


def _repo(tmp_path, monkeypatch, files: dict):
    root = tmp_path / "depo"
    root.mkdir(exist_ok=True)
    for name, body in files.items():
        (root / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(leak_scan, "REPO_ROOT", root)
    return root


def test_the_corpus_is_enumerated_not_listed(veri):
    documents, full, contributed = _corpora(veri)
    assert contributed == 2
    assert fold(CORPUS_SENTENCE)[:30] in documents
    assert documents == full  # no output dir given


def test_a_planted_fragment_is_found(tmp_path, monkeypatch, veri):
    documents, full, _ = _corpora(veri)
    # the digit-bearing slice around "47 000": fact-shaped, so it must land
    # in the HARD bucket, not the prose triage list
    start = CORPUS_SENTENCE.index("Zeta")
    planted = CORPUS_SENTENCE[start:start + 20]
    _repo(tmp_path, monkeypatch, {
        "temiz.py": "value = 1\n",
        "sizintili.py": f"# aciklama: {planted}\n",
    })
    report = scan(["temiz.py", "sizintili.py"], documents, full, width=14)
    assert report["files"] == 2
    assert "sizintili.py" in report["document_hits"]
    assert "temiz.py" not in report["document_hits"]


def test_hits_are_locations_never_content(tmp_path, monkeypatch, veri):
    """Round 15: the tool used to print the matched windows -- document
    fragments -- into its own report. Hits now carry a file, a window
    count and line numbers; the matched text appears nowhere."""
    documents, full, _ = _corpora(veri)
    start = CORPUS_SENTENCE.index("Zeta")
    planted = CORPUS_SENTENCE[start:start + 20]
    _repo(tmp_path, monkeypatch, {
        "sizintili.py": f"deger = 1\n# aciklama: {planted}\n"})
    report = scan(["sizintili.py"], documents, full, width=14)
    entry = report["document_hits"]["sizintili.py"]
    assert set(entry) == {"pencere", "satirlar"}
    assert entry["pencere"] >= 1
    assert entry["satirlar"] == [2]      # points AT the leak
    import json as _json
    blob = _json.dumps(report, ensure_ascii=False)
    assert planted not in blob
    assert fold(planted)[:14] not in blob


def test_a_textless_pdf_is_unreadable_not_a_contribution(veri):
    """Round 15: a VALID pdf with no text layer sailed through as a
    successful contribution -- the exact shape of a scanned document
    hiding from a text scan. It now fails the data-side corpus."""
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page()
    doc.save(str(veri / "metinsiz.pdf"))
    doc.close()
    with pytest.raises(RuntimeError):
        build_corpora(veri, None)


def test_letters_only_overlaps_are_triage_not_hard_findings(
        tmp_path, monkeypatch, veri):
    """Ordinary vocabulary shared with the documents ("kurgusal donem
    ozetinde" here) is the class twelve audit rounds triaged by hand every
    time: visible, never fatal. A digit-bearing window stays fatal."""
    documents, full, _ = _corpora(veri)
    prose = CORPUS_SENTENCE[25:43]  # letters-only mid-sentence slice
    _repo(tmp_path, monkeypatch, {"duzyazi.py": f"# not: {prose}\n"})
    report = scan(["duzyazi.py"], documents, full, width=14)
    assert "duzyazi.py" not in report["document_hits"]
    assert "duzyazi.py" in report["document_prose_hits"]


def test_width_is_a_parameter_because_floors_are_blind_spots(
        tmp_path, monkeypatch, veri):
    documents, full, _ = _corpora(veri)
    # a digit-bearing 13-char slice ("Endeksi 47 00" area): the historic
    # leak was exactly one character under a fixed floor
    start = fold(CORPUS_SENTENCE).index("endeksi 47")
    short_leak = fold(CORPUS_SENTENCE)[start:start + 13]
    assert len(short_leak) == 13
    # flush against a letter: a leading space would hand the scan a free
    # 14th character and reconstruct the wider window by accident
    _repo(tmp_path, monkeypatch, {"kisa.py": f"# x{short_leak}\n"})
    wide = scan(["kisa.py"], documents, full, width=14)
    assert "kisa.py" not in wide["document_hits"]
    narrow = scan(["kisa.py"], documents, full, width=13)
    assert "kisa.py" in narrow["document_hits"]


def test_output_echoes_are_triage_not_document_leaks(
        tmp_path, monkeypatch, veri):
    """The two-haystack rule: our own saved output echoing back is triage,
    a document hit is a leak candidate -- the report must keep them apart."""
    out_dir = tmp_path / "cikti"
    out_dir.mkdir()
    (out_dir / "sonuc.json").write_text(
        '{"cevap": "kurgu cikti cumlesi burada durur"}', encoding="utf-8")
    documents, full, _ = _corpora(veri, out_dir)
    _repo(tmp_path, monkeypatch, {
        "yankili.py": "# kurgu cikti cumlesi burada durur\n"})
    report = scan(["yankili.py"], documents, full, width=14)
    assert "yankili.py" not in report["document_hits"]
    assert "yankili.py" in report["full_only_hits"]


def test_an_unreadable_pdf_fails_the_scan_instead_of_shrinking_it(veri):
    """Round 14: eight real PDFs contributed zero characters because the
    reader library was missing, and the scan blessed the repo anyway. A
    DOCUMENT that cannot be read is a place a leak can hide -- hard error."""
    (veri / "bozuk.pdf").write_bytes(b"bu bir pdf degil")
    with pytest.raises(RuntimeError):
        build_corpora(veri, None)


def test_output_litter_is_a_warning_not_a_death(tmp_path, veri):
    """Our own artifact tree accumulates audit litter (fixture "PDFs" that
    are not PDFs). Dying on our own byproducts makes the scan get skipped
    -- the historic failure -- so output-side unreadables come back as a
    visible list while the scan continues."""
    out_dir = tmp_path / "cikti"
    out_dir.mkdir()
    (out_dir / "kurgu-artik.pdf").write_bytes(b"pdf degil")
    (out_dir / "gercek.json").write_text('{"a": "kurgu"}', encoding="utf-8")
    documents, full, _, unreadable = build_corpora(veri, out_dir)[:4]
    assert len(unreadable) == 1
    assert "kurgu-artik" in unreadable[0]
    assert "kurgu" in full  # the readable neighbour still contributed


def test_sqlite_content_reaches_the_corpus(veri):
    import sqlite3

    db_path = veri / "kurgu.db"
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE kayit (metin text)")
    connection.execute("INSERT INTO kayit VALUES ('kurgu veritabani satiri')")
    connection.commit()
    connection.close()
    documents, _, contributed = _corpora(veri)
    assert contributed == 3
    assert "kurgu veritabani satiri" in documents


def test_xlsx_content_reaches_the_corpus(veri):
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.append(["kurgu hucre degeri", 47])
    workbook.save(veri / "kurgu.xlsx")
    documents, _, contributed = _corpora(veri)
    assert contributed == 3
    assert "kurgu hucre degeri" in documents


def test_a_secret_is_reported_as_a_label_never_as_text(
        tmp_path, monkeypatch, veri):
    """Round 14: the tool echoed the first 40 characters of what it caught,
    putting password material into its own report. Labels only now -- and
    documented placeholders are not findings, so the tool does not cry
    wolf on .env.example forever."""
    documents, full, _ = _corpora(veri)
    parola = "gizli" + "Parola7"
    dsn = "postgresql" + f"://birim7:{parola}@konak/veritabani"
    placeholder = "postgresql" + "://birim7:CHANGE_ME@konak/veritabani"
    fixture = "postgresql" + "://birim7:sahte@konak/veritabani"
    root = _repo(tmp_path, monkeypatch, {
        "sirli.py": f'DSN = "{dsn}"\n',
        "yertutucu.py": f'DSN = "{placeholder}"\n',
    })
    # round 16: fixture exemption comes from LIVING UNDER tests/, never
    # from vocabulary -- a broad word list could mask a real secret
    (root / "tests").mkdir()
    (root / "tests" / "fikstur.py").write_text(f'DSN = "{fixture}"\n',
                                              encoding="utf-8")
    report = scan(["sirli.py", "yertutucu.py", "tests/fikstur.py"],
                  documents, full, width=14)
    assert report["secrets"]["sirli.py"] == ["dsn_kimlikli"]
    assert parola not in str(report)          # the text never comes back
    assert "yertutucu.py" not in report["secrets"]
    # round 17: the tests/ path routes to TRIAGE, it is no longer exempt --
    # a real credential pasted into a test used to be invisible by path
    assert "tests/fikstur.py" not in report["secrets"]
    assert report["test_secrets"]["tests/fikstur.py"] == ["dsn_kimlikli"]


def test_invisible_characters_are_flagged(tmp_path, monkeypatch, veri):
    documents, full, _ = _corpora(veri)
    _repo(tmp_path, monkeypatch, {
        "hayaletli.py": "x = 'kurgu" + chr(0xA0) + "deger'\n"})
    report = scan(["hayaletli.py"], documents, full, width=14)
    assert report["invisible"]["hayaletli.py"] == ["NBSP"]


def test_an_empty_scan_is_a_failure_not_a_clean_verdict(
        tmp_path, monkeypatch, veri):
    documents, full, _ = _corpora(veri)
    with pytest.raises(RuntimeError):
        scan([], documents, full, width=14)
    _repo(tmp_path, monkeypatch, {})
    with pytest.raises(RuntimeError):
        scan(["yok.bin"], documents, full, width=14)


def test_a_data_name_piece_inside_content_is_flagged_masked(
        tmp_path, monkeypatch, veri):
    """Round 16, the real finding's shape: two source files' NAME PIECES
    sat inside test fixtures -- content scanning missed them because they
    are names, not content. Pieces are now hunted inside bodies, and
    reported MASKED, because the stem itself must not ride out in the
    report."""
    # a SOURCE-document name (image suffix): stems come from pdf/image
    # names only -- our own eval artifacts' names are project vocabulary
    (veri / "gizlikok-2077-rapor.png").write_bytes(b"\x89PNG kurgu")
    documents, full, _ = _corpora(veri)
    stems = build_corpora(veri, None)[4]
    _repo(tmp_path, monkeypatch, {
        "koklu.py": "# not: gizlikok degeri ile 2077 yilinda\n",
        "temiz.py": "deger = 1\n",
    })
    report = scan(["koklu.py", "temiz.py"], documents, full, width=14,
                  data_stems=stems)
    hits = report["stem_hits"]["koklu.py"]
    assert any(h.startswith("g") and h.endswith("k") for h in hits)
    assert any(h.startswith("2") and h.endswith("7") for h in hits)
    assert "gizlikok" not in str(hits)     # masked, never verbatim
    assert "temiz.py" not in report["stem_hits"]


def test_an_unknown_data_suffix_fails_the_corpus(veri):
    """A file type the collector does not understand is a place a leak can
    hide -- 12 of 34 files were once outside the haystack in silence."""
    (veri / "bilinmez.xyz").write_bytes(b"taninmayan tur")
    with pytest.raises(RuntimeError):
        build_corpora(veri, None)


def test_images_are_counted_as_a_declared_limit(veri):
    (veri / "gorsel.png").write_bytes(b"\x89PNG kurgu")
    corpora = build_corpora(veri, None)
    contributed, images = corpora[2], corpora[6]
    assert images == 1
    assert contributed == 2  # the text files still contribute


def test_a_real_password_containing_dots_is_not_a_placeholder(
        tmp_path, monkeypatch, veri):
    """Round 17: "..." sat in the generic placeholder list, and a real
    password CONTAINING three dots was cleared by them. The DSN's
    clearance is judged on the whole password field now."""
    documents, full, _ = _corpora(veri)
    parola = "ger" + "..." + "cek7"                 # dots INSIDE a password
    dsn = "postgresql" + f"://birim7:{parola}@konak/veritabani"
    sablon = "postgresql" + "://birim7:...@konak/veritabani"
    _repo(tmp_path, monkeypatch, {
        "noktali.py": f'DSN = "{dsn}"\n',
        "sablonlu.py": f'DSN = "{sablon}"\n',
    })
    report = scan(["noktali.py", "sablonlu.py"], documents, full, width=14)
    assert report["secrets"]["noktali.py"] == ["dsn_kimlikli"]  # a FINDING
    assert "sablonlu.py" not in report["secrets"]   # whole-field ellipsis


def test_a_source_sheet_wearing_a_tool_name_is_still_content(veri):
    """Round 17: the rapor/karsilastirma sheet skip existed for OUR output
    workbooks, but it also ran against SOURCE workbooks -- a data sheet
    named "Rapor" was excluded from the haystack while the file counted
    as contributed. Data-side sheets are read whole now."""
    openpyxl = pytest.importorskip("openpyxl")
    workbook = openpyxl.Workbook()
    workbook.active.title = "Rapor"
    workbook.active.append(["kurgu rapor sayfasi icerigi", 47])
    workbook.save(veri / "kurgu-kaynak.xlsx")
    documents, _, contributed = _corpora(veri)
    assert contributed == 3
    assert "kurgu rapor sayfasi icerigi" in documents


def test_the_output_side_keeps_the_tool_sheet_skip(tmp_path, veri):
    openpyxl = pytest.importorskip("openpyxl")
    out_dir = tmp_path / "cikti"
    out_dir.mkdir()
    workbook = openpyxl.Workbook()
    workbook.active.title = "Rapor"
    workbook.active.append(["arac mesaji burada"])
    workbook.save(out_dir / "sonuc.xlsx")
    _documents, full, _ = _corpora(veri, out_dir)
    assert "arac mesaji burada" not in full


def test_an_unsupported_repo_file_is_reported_not_silently_skipped(
        tmp_path, monkeypatch, veri):
    """Round 17: a tracked file with an unsupported extension was simply
    not scanned -- any binary-shaped file could opt out of the battery by
    its suffix. It cannot vanish any more."""
    documents, full, _ = _corpora(veri)
    root = _repo(tmp_path, monkeypatch, {"temiz.py": "deger = 1\n"})
    (root / "ikili.bin").write_bytes(b"\x00\x01kurgu")
    report = scan(["temiz.py", "ikili.bin"], documents, full, width=14)
    assert report["kapsam_disi"] == ["ikili.bin"]
    assert report["files"] == 1


def test_a_typescript_source_is_scanned_as_text(
        tmp_path, monkeypatch, veri):
    """Generated clients are part of the repository's leak surface."""
    documents, full, _ = _corpora(veri)
    start = CORPUS_SENTENCE.index("Zeta")
    planted = CORPUS_SENTENCE[start:start + 20]
    _repo(tmp_path, monkeypatch, {
        "istemci.ts": f"// aciklama: {planted}\n",
    })
    report = scan(["istemci.ts"], documents, full, width=14)
    assert report["files"] == 1
    assert report["kapsam_disi"] == []
    assert "istemci.ts" in report["document_hits"]


def test_a_stem_shared_with_repo_vocabulary_is_triaged_not_dropped(
        tmp_path, monkeypatch, veri):
    """Round 17: a piece that also names a repo file used to be EXCLUDED
    from the stem battery -- so a repo file named after the source
    document allowlisted its own name, and the filename layer could never
    fire by construction. Overlaps land in their own triage bucket now."""
    (veri / "ortakkokx-rapor.png").write_bytes(b"\x89PNG kurgu")
    monkeypatch.setattr(leak_scan, "tracked_files",
                        lambda commits=None: ["ortakkokx_modul.py"])
    corpora = build_corpora(veri, None)
    hard_stems, vocab_stems = corpora[4], corpora[5]
    assert "ortakkokx" in vocab_stems      # returned, not swallowed
    assert "ortakkokx" not in hard_stems
    documents, full = corpora[0], corpora[1]
    _repo(tmp_path, monkeypatch, {
        "isaretli.py": "# ortakkokx degerleri burada\n"})
    report = scan(["isaretli.py"], documents, full, width=14,
                  data_stems=hard_stems, vocab_stems=vocab_stems)
    assert "isaretli.py" not in report["stem_hits"]
    hits = report["stem_vocab_hits"]["isaretli.py"]
    assert any(h.startswith("o") and h.endswith("x") for h in hits)
    assert "ortakkokx" not in str(hits)    # masked here too


def test_a_binary_only_commit_cannot_scan_clean(tmp_path, monkeypatch, veri):
    """Round 18: commit mode git-showed ANYTHING as text, and a commit
    holding only binary files scanned its own garbage to a TEMIZ exit.
    Both modes use one suffix policy now; the unscannable files land
    out-of-scope, and the commit message is still real scan input."""
    documents, full, _ = _corpora(veri)
    _repo(tmp_path, monkeypatch, {})
    report = scan(["gorsel.png", "veri.xlsx"], documents, full, width=14,
                  commit="kurgusha404",
                  extra_bodies=[("commit-mesaji:kurgusha404",
                                 "kurgu commit mesaji")])
    assert report["kapsam_disi"] == ["gorsel.png", "veri.xlsx"]
    assert report["files"] == 1  # only the message was actually scanned


def test_a_placeholder_word_inside_a_real_token_is_still_a_finding(
        tmp_path, monkeypatch, veri):
    """Round 18: placeholder clearance was substring search over the whole
    match -- a real-looking token CARRYING the word cleared itself. The
    token must now BE the placeholder, entirely."""
    documents, full, _ = _corpora(veri)
    tainted = "api" + '_key = "gercekCHANGE_MEgibi12345"'
    pure = "api" + '_key = "CHANGE_ME"'
    _repo(tmp_path, monkeypatch, {
        "lekelix.py": tainted + "\n",
        "temizsablon.py": pure + "\n",
    })
    report = scan(["lekelix.py", "temizsablon.py"], documents, full, width=14)
    assert report["secrets"]["lekelix.py"] == ["api_anahtari"]
    assert "temizsablon.py" not in report["secrets"]


def test_a_renamed_separator_multi_piece_name_is_still_caught(
        tmp_path, monkeypatch, veri):
    """Round 18: every piece of "kurgu-kok-adi" sits under the word floor,
    so writing the name with ANOTHER separator escaped the stem layer.
    The whole name, re-joined every common way, now travels too."""
    (veri / "kurgu-kok-adi.png").write_bytes(b"\x89PNG kurgu")
    documents, full, _ = _corpora(veri)
    stems = build_corpora(veri, None)[4]
    _repo(tmp_path, monkeypatch, {
        "ayracli.py": "# deger: kurgu_kok_adi uzerinden\n",
        "bitisik.py": "# deger: kurgukokadi uzerinden\n",
        "temiz.py": "deger = 1\n",
    })
    report = scan(["ayracli.py", "bitisik.py", "temiz.py"], documents, full,
                  width=14, data_stems=stems)
    assert "ayracli.py" in report["stem_hits"]
    assert "bitisik.py" in report["stem_hits"]
    assert "temiz.py" not in report["stem_hits"]
    assert "kurgu_kok_adi" not in str(report)   # masked, as ever


def test_error_output_masks_broken_source_paths(veri):
    """Round 18: the unreadable-file error echoed full data paths -- the
    very names the battery exists to keep out of print."""
    (veri / "gizliyolx-bozuk.pdf").write_bytes(b"pdf degil")
    with pytest.raises(RuntimeError) as error:
        build_corpora(veri, None)
    message = str(error.value)
    assert "gizliyolx" not in message
    assert "g" + "*" * 13 + "k.pdf" in message  # stem masked to its ends


def test_high_entropy_literals_are_triaged_without_echo(
        tmp_path, monkeypatch, veri):
    """Round 18: a secret assembled or pasted as one uniform token is
    invisible to every named pattern. Key-shaped literals above the
    entropy floor surface as counts and lines; hex hash pins (entropy
    exactly 4.0) stay quiet, and the token itself never prints."""
    documents, full, _ = _corpora(veri)
    hot = "aB3dE5gH7jK9mN1pQ2sT4vW6"
    hex_pin = "0123456789abcdef" * 4
    _repo(tmp_path, monkeypatch, {
        "sicak.py": f'DEGER = "{hot}"\n',
        "soguk.py": f'PIN = "{hex_pin}"\nAD = "aaaabbbbccccddddeeee"\n',
    })
    report = scan(["sicak.py", "soguk.py"], documents, full, width=14)
    assert report["yuksek_entropi"]["sicak.py"] == {"adet": 1,
                                                    "satirlar": [1]}
    assert "soguk.py" not in report["yuksek_entropi"]
    assert hot not in str(report)


def test_the_tool_carries_no_invisible_characters_itself():
    from pathlib import Path

    body = Path(leak_scan.__file__).read_text(encoding="utf-8")
    for ch in (chr(0xA0), chr(0x202F), chr(0x200B), chr(0xFEFF)):
        assert ch not in body
