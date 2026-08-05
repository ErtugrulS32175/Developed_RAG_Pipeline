"""The mutation harness must be deterministic, dev-only and honestly labelled.

Every fixture below is invented and matches the invented vocabulary already
cleared in earlier tests (Zeta/Beta, 47 000 / 88 000, page 42). What is locked
here: each mutator's structural invariant, the refusal to touch locked
clusters, determinism, and that the replay reproduces the PREREGISTERED
expectation -- the constructed classes the guard must catch are caught, and
the wrong-binding classes it is structurally blind to pass through.
"""
import pytest

from eval.answer.adversarial_feasibility import BaseCase, EvidenceClaim
from eval.answer.adversarial_mutate import (
    CANDIDATE_ROLE,
    CONFIRMED_ROLE,
    EXPECTED_DIAGNOSTIC,
    build_mutants,
    development_cases,
    mutate_arithmetic_restatement,
    mutate_corrupted_quote,
    mutate_forced_answer_unanswerable,
    mutate_invented_provenance,
    mutate_label_value_swap,
    mutate_question_answer_mismatch,
    mutate_unsupported_figure,
    mutate_wrong_row,
    population_exclusions,
    build_report,
    contract_metadata,
    replay,
    replay_flags,
)
from pipeline.lang.tr_notation import fold, normalize, numbers
from pipeline.retrieval.context import Passage, RagContext

DEV_ID = "0" * 64
DEV_ID_2 = "1" + "0" * 63
LOCKED_ID = "f" + "0" * 63

ZETA = "Zeta uretimi 47 000 birimdir. Beta uretimi 88 000 birimdir."
OTHER = "Baska bir konu hakkindaki bu metin hicbir rakam tasimaz."
# A genuine sibling-row relation: the SAME label over two records. This is
# what wrong_row must require -- and the two attack passages below are what
# it must refuse: numbers that merely sit near each other, with no shared
# label (BARE) or no label at all (BARE_PAIR).
# label "Zeta Endeksi" verified absent from both haystacks before use; the
# first choice was an ordinary Turkish phrase that DID occur in the documents
SIRA = ("Alfa Sirketi, Zeta Endeksi = 47 000. "
        "Beta Sirketi, Zeta Endeksi = 88 000.")
BARE = "Deger listesi soyle: 47 000 sonra 12 345 sonra 67 890 gelir."
BARE_PAIR = "Toplam 47 000 ile 88 000 burada."


def _case(stable_id, *, key="47 000", answer="Sayfa 42'ye gore 47 000 birim.",
          question="zeta uretimi nedir?", passages=None, evidence=None,
          set_name="kurgu", question_type="sayisal"):
    passages = passages or [Passage(1, 42, ZETA, "kurgu-belge s.42")]
    evidence = evidence if evidence is not None else [(1, ZETA)]
    return BaseCase(
        stable_id=stable_id,
        locked=stable_id[0] not in "01234",
        set_name=set_name,
        question=question,
        key=key,
        answer=answer,
        question_type=question_type,
        context=RagContext(passages=tuple(passages), numbered=True),
        evidence_shape_valid=True,
        evidence=tuple(EvidenceClaim(h, q) for h, q in evidence),
        source_row_fingerprint="0" * 64,
    )


def _row(mutant):
    return {"cevap": mutant.answer, "dayanak": list(mutant.evidence)}


ALL_MUTATORS = [
    mutate_unsupported_figure, mutate_arithmetic_restatement, mutate_wrong_row,
    mutate_label_value_swap, mutate_question_answer_mismatch,
    mutate_forced_answer_unanswerable, mutate_corrupted_quote,
    mutate_invented_provenance,
]


@pytest.mark.parametrize("mutator", ALL_MUTATORS, ids=lambda f: f.__name__)
def test_every_mutator_refuses_a_locked_case(mutator):
    with pytest.raises(ValueError):
        mutator(_case(LOCKED_ID), ())


def test_development_filter_drops_locked_cases():
    kept = development_cases((_case(DEV_ID), _case(LOCKED_ID)))
    assert [case.stable_id for case in kept] == [DEV_ID]


def test_a_forged_dev_flag_on_a_locked_id_is_refused():
    """The auditor's attack: locked identity with locked=False must not pass.
    The split is recomputed from the id; a lying flag is an error, not an
    instruction."""
    import dataclasses

    forged = dataclasses.replace(_case(LOCKED_ID), locked=False)
    with pytest.raises(ValueError):
        mutate_unsupported_figure(forged, ())
    with pytest.raises(ValueError):
        development_cases((forged,))


def test_a_malformed_stable_id_is_refused():
    """The auditor's attack: a 63-character id must be rejected outright."""
    broken = _case(DEV_ID)
    broken = type(broken)(**{**broken.__dict__, "stable_id": "0" * 63})
    with pytest.raises(ValueError):
        mutate_unsupported_figure(broken, ())
    with pytest.raises(ValueError):
        development_cases((broken,))


def test_a_forged_peer_cannot_be_smuggled_into_a_mutation():
    """The auditor's attack: peers are validated at the same gate as cases."""
    import dataclasses

    case = _case(DEV_ID)
    forged_peer = dataclasses.replace(
        _case(LOCKED_ID, key="88 000",
              answer="Sayfa 42'ye gore 88 000 birim.",
              question="beta uretimi nedir?"),
        locked=False)
    with pytest.raises(ValueError):
        mutate_label_value_swap(case, (forged_peer,))
    with pytest.raises(ValueError):
        mutate_question_answer_mismatch(case, (forged_peer,))


def test_unsupported_figure_invents_an_absent_value_and_is_caught():
    case = _case(DEV_ID)
    mutant = mutate_unsupported_figure(case, ())
    assert mutant is not None and mutant.answer != case.answer
    mutated_values = numbers(normalize(mutant.answer)) - {42.0}
    context_values = numbers(normalize(case.context.model_text))
    assert mutated_values and mutated_values.isdisjoint(context_values)
    flags = replay_flags(_row(mutant), mutant.context, "plain")
    assert any(code == "kaynaksiz_sayi" for code, _ in flags)


def test_mutation_is_deterministic():
    first = mutate_unsupported_figure(_case(DEV_ID), ())
    second = mutate_unsupported_figure(_case(DEV_ID), ())
    assert first.answer == second.answer


def test_arithmetic_restatement_shifts_magnitude_and_is_caught():
    case = _case(DEV_ID)
    mutant = mutate_arithmetic_restatement(case, ())
    assert mutant is not None
    shifted = numbers(normalize(mutant.answer))
    assert 470000.0 in shifted or 4700.0 in shifted
    flags = replay_flags(_row(mutant), mutant.context, "plain")
    assert any(code == "kaynaksiz_sayi" for code, _ in flags)


def test_wrong_row_requires_a_shared_label_frame_and_passes_unseen():
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, SIRA, "kurgu-belge s.42")],
                 evidence=[(1, SIRA)])
    mutant = mutate_wrong_row(case, ())
    assert mutant is not None
    assert "88 000" in mutant.answer
    # the sibling value really is in the cited passage: nothing to catch
    assert replay_flags(_row(mutant), mutant.context, "structured_derived") == []


def test_wrong_row_refuses_numbers_without_a_row_relation():
    """The auditor's attack: any nearby figure must NOT count as a sibling."""
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, BARE, "kurgu-belge s.42")],
                 evidence=[(1, BARE)])
    assert mutate_wrong_row(case, ()) is None


def test_wrong_row_refuses_a_footnote_marker_as_sibling_value():
    """The diagnosed escape: '(birim) 1 = ...' footnote markers share the
    key's label frame but are not record values. A mutant built from one has
    a figure that appears in every segment -- uncatchable and unadjudicable."""
    text = "Alfa Sirketi, Zeta Endeksi (birim) 1 = 47 000."
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, text, "kurgu-belge s.42")],
                 evidence=[(1, text)])
    assert mutate_wrong_row(case, ()) is None


def test_wrong_row_skips_figures_from_the_keys_own_record():
    """The diagnosed escape: a column-header year repeated in every row
    shares the key's label frame after digit-stripping, but it also sits in
    the key's own record -- so it is not a sibling VALUE. The real sibling
    value must be chosen instead."""
    text = ("Alfa Sirketi, Zeta Endeksi 2024 = 47 000. "
            "Beta Sirketi, Zeta Endeksi 2024 = 88 000.")
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, text, "kurgu-belge s.42")],
                 evidence=[(1, text)])
    mutant = mutate_wrong_row(case, ())
    assert mutant is not None
    assert "88 000" in mutant.answer
    assert "2024" not in mutant.answer.replace("Sayfa 42", "")


def test_a_single_digit_key_is_a_declared_gap_not_a_silent_one():
    """MUT-02 across three audit rounds, and the limit we settled on.

    A single-digit answer has single-digit siblings, so admitting the shape
    means admitting footnote markers with it. Two attempts to separate them
    failed -- the marker was let in outright, then a rule about how much
    prose follows the token was walked through by putting the marker at the
    end of its record. The document writes "Zeta Sayisi 2" for a value and
    for a reference alike; nothing local tells them apart. So the shape is
    OUT of the population and the harness says so by producing nothing,
    rather than producing something no one can adjudicate.

    Both fixtures below are the shape: one with a clean sibling, one with a
    marker. Neither yields a mutant, and that is the intended behaviour."""
    clean = "Alfa Sirketi, Zeta Sayisi = 3. Beta Sirketi, Zeta Sayisi = 7."
    marker = ("Alfa Sirketi, Zeta Sayisi = 3. "
              "Gama Sirketi, Zeta Sayisi 2. "
              "Beta Sirketi, Zeta Sayisi = 7.")
    for text in (clean, marker):
        case = _case(DEV_ID, key="3", answer="Sayfa 42'ye gore 3 adettir.",
                     passages=[Passage(1, 42, text, "kurgu-belge s.42")],
                     evidence=[(1, text)])
        assert mutate_wrong_row(case, ()) is None


def test_a_lower_case_passage_still_yields_a_sibling_mutant():
    """Auditor finding MUT-R5-01: the mutator's segmenter demanded an
    uppercase start after the boundary, so a lower-cased passage never
    split -- every figure landed inside the key's "own record" and a valid
    sibling produced no mutant. The validator had already learned this
    shape; the instrument had not, and an instrument blind to a shape the
    subject handles under-measures exactly there."""
    text = ("alfa sirketi, zeta endeksi = 47 000. "
            "beta sirketi, zeta endeksi = 88 000.")
    case = _case(DEV_ID, key="47 000",
                 answer="Sayfa 42'ye gore 47 000 birimdir.",
                 passages=[Passage(1, 42, text, "kurgu-belge s.42")],
                 evidence=[(1, text)])
    mutant = mutate_wrong_row(case, ())
    assert mutant is not None
    assert "88 000" in mutant.answer


def test_population_exclusions_count_the_declared_gap():
    """Auditor finding MUT-R5-02: a policy exclusion that only shows up as
    "produced nothing" is a silent denominator change. The report must say
    how many cases each declared rule removed."""
    text = "Alfa Sirketi, Zeta Sayisi = 3. Beta Sirketi, Zeta Sayisi = 7."
    single = _case(DEV_ID, key="3", answer="Sayfa 42'ye gore 3 adettir.",
                   passages=[Passage(1, 42, text, "kurgu-belge s.42")],
                   evidence=[(1, text)])
    multi = _case(DEV_ID, key="47 000",
                  answer="Sayfa 42'ye gore 47 000 birimdir.",
                  passages=[Passage(1, 42, SIRA, "kurgu-belge s.42")],
                  evidence=[(1, SIRA)])
    counts = population_exclusions((single, multi))
    assert counts["wrong_row"] == {"single_digit_key": 1}


def test_contract_metadata_reads_version_hash_and_addenda(tmp_path):
    """Round-9 P2: the SHA linkage existed but nothing tested it. Pinned:
    missing directory is all-null and incomplete; a base contract yields
    its declared protocol, exact hash and a COMPLETE chain; an addendum
    appears by name with its own hash and the effective version follows its
    declaration, so a scorer reads one field instead of inferring a version
    from a filename map."""
    import hashlib

    assert contract_metadata(tmp_path / "yok") == {
        "contract_version": None, "contract_sha256": None,
        "effective_protocol_version": None, "contract_complete": False,
        "addenda": {}}

    base = tmp_path / "adversarial_contract.md"
    base.write_text("# Kurgu sozlesme\n\nProtocol: `kurgu_protokol_v9`\n",
                    encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_version"] == "kurgu_protokol_v9"
    assert meta["effective_protocol_version"] == "kurgu_protokol_v9"
    assert meta["contract_sha256"] == hashlib.sha256(
        base.read_bytes()).hexdigest()
    assert meta["contract_complete"] is True
    assert meta["addenda"] == {}

    addendum = tmp_path / "adversarial_contract_ek.md"
    addendum.write_text("Protocol: `kurgu_protokol_v9.1`\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_version"] == "kurgu_protokol_v9"
    assert meta["effective_protocol_version"] == "kurgu_protokol_v9.1"
    assert meta["contract_complete"] is True
    assert meta["addenda"] == {"adversarial_contract_ek.md": {
        "sha256": hashlib.sha256(addendum.read_bytes()).hexdigest(),
        "protocol": "kurgu_protokol_v9.1"}}

    # an addendum that declares nothing changes the effective version not
    # at all -- the owner's cue to declare, not this module's cue to guess
    silent = tmp_path / "adversarial_contract_not.md"
    silent.write_text("Sadece aciklama.\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["effective_protocol_version"] == "kurgu_protokol_v9.1"
    assert meta["contract_complete"] is True
    assert meta["addenda"]["adversarial_contract_not.md"]["protocol"] is None


def test_contract_chain_fails_closed(tmp_path):
    """Auditor finding, round 10: the chain was fail-open in two ways. An
    addendum standing WITHOUT its base contract filled the effective
    version while the hash stayed null -- a scorer reading only that field
    would run a locked measurement bound to nothing. And two declaring
    addenda elected a winner by filename sort, which is not version order
    ("v2.10" sorts before "v2.2"). Now: no complete base means no effective
    version at all, and two declarers raise instead of choosing."""
    import pytest as _pytest

    # addendum only, no base: recorded, but never effective
    orphan = tmp_path / "adversarial_contract_ek.md"
    orphan.write_text("Protocol: `kurgu_protokol_v9.1`\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_sha256"] is None
    assert meta["effective_protocol_version"] is None
    assert meta["contract_complete"] is False
    assert meta["addenda"]["adversarial_contract_ek.md"]["protocol"] == (
        "kurgu_protokol_v9.1")

    # base present but declaring no Protocol line: hashed, still incomplete
    base = tmp_path / "adversarial_contract.md"
    base.write_text("# Kurgu sozlesme, surum satiri yok\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_sha256"] is not None
    assert meta["contract_version"] is None
    assert meta["effective_protocol_version"] is None
    assert meta["contract_complete"] is False

    # two declaring addenda: refuse, never silently pick
    second = tmp_path / "adversarial_contract_iki.md"
    second.write_text("Protocol: `kurgu_protokol_v9.2`\n", encoding="utf-8")
    with _pytest.raises(ValueError):
        contract_metadata(tmp_path)


def test_two_protocol_lines_inside_one_file_are_refused(tmp_path):
    """Auditor finding, round 11: the cross-file ambiguity check could not
    see INSIDE a file -- two Protocol lines in one document silently
    resolved to the first, chain still reported complete. Ambiguity must
    refuse identically wherever it lives: a double-declaring base and a
    double-declaring lone addendum both raise."""
    import pytest as _pytest

    base = tmp_path / "adversarial_contract.md"
    base.write_text("Protocol: `kurgu_protokol_v9`\n"
                    "Protocol: `kurgu_protokol_v8`\n", encoding="utf-8")
    with _pytest.raises(ValueError):
        contract_metadata(tmp_path)

    base.write_text("Protocol: `kurgu_protokol_v9`\n", encoding="utf-8")
    addendum = tmp_path / "adversarial_contract_ek.md"
    addendum.write_text("Protocol: `kurgu_protokol_v9.1`\n"
                        "Protocol: `kurgu_protokol_v9.2`\n", encoding="utf-8")
    with _pytest.raises(ValueError):
        contract_metadata(tmp_path)

    # and the repaired addendum restores a complete chain
    addendum.write_text("Protocol: `kurgu_protokol_v9.1`\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_complete"] is True
    assert meta["effective_protocol_version"] == "kurgu_protokol_v9.1"


def test_whitespace_or_bom_cannot_hide_a_protocol_declaration(tmp_path):
    """Auditor finding, round 12: the match required the line to START with
    the keyword, so a second declaration behind a space or tab went
    uncounted -- and a BOM hid the FIRST one, silently electing the second:
    the worse direction, because the invisible byte chose the version. All
    three now count and the ambiguity refuses. The declared cost: an
    indented example inside a code block counts too, and refusing an
    example beats electing a wrong version."""
    import pytest as _pytest

    base = tmp_path / "adversarial_contract.md"
    for body in (
        "Protocol: `kurgu_protokol_v9`\n Protocol: `kurgu_protokol_v8`\n",
        "Protocol: `kurgu_protokol_v9`\n\tProtocol: `kurgu_protokol_v8`\n",
        "\ufeffProtocol: `kurgu_protokol_v9`\nProtocol: `kurgu_protokol_v8`\n",
    ):
        base.write_text(body, encoding="utf-8")
        with _pytest.raises(ValueError):
            contract_metadata(tmp_path)

    # a BOM on a file with ONE declaration is an encoding artefact, not an
    # ambiguity: the declaration is seen and the chain completes
    base.write_text("\ufeffProtocol: `kurgu_protokol_v9`\n", encoding="utf-8")
    meta = contract_metadata(tmp_path)
    assert meta["contract_version"] == "kurgu_protokol_v9"
    assert meta["contract_complete"] is True


def test_contract_metadata_ignores_the_working_directory(tmp_path, monkeypatch):
    """Round-9 P2: the directory was a relative path, so calling from
    anywhere but the repo root silently produced all-null metadata -- the
    silent-zero shape again. Anchoring goes through __file__; the same
    answer must come back from any working directory."""
    at_root = contract_metadata()
    monkeypatch.chdir(tmp_path)
    assert contract_metadata() == at_root


def test_the_report_carries_the_contract_metadata():
    """The linkage the locked run will refuse to score without."""
    report = build_report((), {})
    assert report["contract"] == contract_metadata()
    assert "effective_protocol_version" in report["contract"]


def test_the_replayed_row_carries_the_question():
    """Policies that check BINDING need the question; a row without it
    scores zero for them silently -- which an early version reported as a
    real 0/2 result."""
    from eval.answer.adversarial_mutate import _row

    row = _row("cevap", [], "kurgu soru?")
    assert row["soru"] == "kurgu soru?"

    report = replay(_small_population())
    binding = report["wrong_row"]["policies"].get("plain_binding")
    assert binding is not None
    assert binding["counts"]["n"] > 0


def test_the_harness_does_not_borrow_the_validators_segmenter():
    """Auditor finding MUT-01: instrument and subject must be independent,
    or a segmentation bug hides itself in both."""
    import inspect

    import eval.answer.adversarial_mutate as harness
    from pipeline.validation.rag import binding_guard

    assert harness._segments is not binding_guard._segments
    source = inspect.getsource(harness)
    assert "from pipeline.validation.rag.binding_guard import" not in source


def test_label_value_swap_refuses_two_bare_numbers():
    """The auditor's attack: a swap needs two LABELLED relations, not two
    numbers standing next to each other."""
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, BARE_PAIR, "kurgu-belge s.42")],
                 evidence=[(1, BARE_PAIR)])
    peer = _case(DEV_ID_2, key="88 000",
                 answer="Sayfa 42'ye gore 88 000 birim.",
                 question="beta uretimi nedir?",
                 passages=[Passage(1, 42, BARE_PAIR, "kurgu-belge s.42")],
                 evidence=[(1, BARE_PAIR)])
    assert mutate_label_value_swap(case, (peer,)) is None


def test_label_value_swap_uses_the_peer_key_and_passes_unseen():
    case = _case(DEV_ID)
    peer = _case(DEV_ID_2, key="88 000",
                 answer="Sayfa 42'ye gore 88 000 birim.",
                 question="beta uretimi nedir?")
    mutant = mutate_label_value_swap(case, (peer,))
    assert mutant is not None and "88 000" in mutant.answer
    assert replay_flags(_row(mutant), mutant.context, "plain") == []


def test_question_answer_mismatch_takes_the_peer_answer_verbatim():
    case = _case(DEV_ID)
    peer = _case(DEV_ID_2, key="88 000",
                 answer="Sayfa 42'ye gore 88 000 birim.",
                 question="beta uretimi nedir?")
    mutant = mutate_question_answer_mismatch(case, (peer,))
    assert mutant is not None and mutant.answer == peer.answer
    assert replay_flags(_row(mutant), mutant.context, "structured_derived") == []


def test_forced_answer_strips_every_key_passage_and_is_caught():
    case = _case(DEV_ID, passages=[
        Passage(1, 42, ZETA, "kurgu-belge s.42"),
        Passage(2, 7, OTHER, "kurgu-belge s.7"),
    ])
    mutant = mutate_forced_answer_unanswerable(case, ())
    assert mutant is not None
    assert [p.handle for p in mutant.context.passages] == [2]
    flags = replay_flags(_row(mutant), mutant.context, "plain")
    assert any(code == "kaynaksiz_sayi" for code, _ in flags)


def test_corrupted_quote_leaves_the_passage_and_is_caught():
    case = _case(DEV_ID)
    mutant = mutate_corrupted_quote(case, ())
    assert mutant is not None and mutant.answer == case.answer
    corrupted = mutant.evidence[0]["alinti"]
    assert fold(corrupted) not in fold(ZETA)
    flags = replay_flags(_row(mutant), mutant.context, "structured_derived")
    assert any(code == "uydurma_alinti" for code, _ in flags)


def test_expected_diagnostic_map_is_frozen_to_the_contract():
    """Only three classes carry a universal diagnostic code. An earlier draft
    required kaynaksiz_sayi for arithmetic/forced too, which redefined
    diagnostic recall outside the frozen contract -- this locks the rollback."""
    universal = {name: code for name, code in EXPECTED_DIAGNOSTIC.items() if code}
    assert universal == {
        "unsupported_figure": "kaynaksiz_sayi",
        "corrupted_quote": "uydurma_alinti",
        "invented_provenance": "uydurma_pasaj",
    }


def _small_population():
    case = _case(DEV_ID,
                 passages=[Passage(1, 42, SIRA, "kurgu-belge s.42")],
                 evidence=[(1, SIRA)])
    peer = _case(DEV_ID_2, key="88 000",
                 answer="Sayfa 42'ye gore 88 000 birim.",
                 question="beta uretimi nedir?",
                 passages=[Passage(1, 42, ZETA, "kurgu-belge s.42")],
                 evidence=[(1, ZETA)])
    return build_mutants((case, peer))


def test_relation_classes_are_reported_as_candidates_not_confirmed():
    """A label frame screens degenerate constructions; it cannot PROVE a real
    record or label relation. Whatever survives it must be labelled a
    candidate awaiting human confirmation, outside any recall denominator."""
    report = replay(_small_population())
    for name in ("wrong_row", "label_value_swap", "question_answer_mismatch"):
        assert report[name]["population_role"] == CANDIDATE_ROLE
        assert report[name]["structural_recall_eligible"] is False
    assert report["unsupported_figure"]["population_role"] == CONFIRMED_ROLE
    assert report["unsupported_figure"]["structural_recall_eligible"] is True


def test_replay_report_serializes_deterministically():
    """An unsorted flag dict once made two identical runs hash differently.

    A same-process double replay CANNOT catch that bug: set iteration order is
    stable within one process, so the broken code passed it too (verified by
    the auditor under four PYTHONHASHSEED values). The real lock is the ORDER
    PROPERTY itself -- every serialized flags dict must already be sorted --
    with the double-replay equality kept only as a cheap sanity check."""
    import json

    report = replay(_small_population())
    for body in report.values():
        for result in body["policies"].values():
            codes = list(result["flags"])
            assert codes == sorted(codes)
    assert json.dumps(report) == json.dumps(replay(_small_population()))


def test_invented_provenance_cites_a_handle_the_context_never_supplied():
    case = _case(DEV_ID)
    mutant = mutate_invented_provenance(case, ())
    assert mutant is not None
    assert mutant.evidence[0]["pasaj"] == 2
    flags = replay_flags(_row(mutant), mutant.context, "structured_derived")
    assert any(code == "uydurma_pasaj" for code, _ in flags)
