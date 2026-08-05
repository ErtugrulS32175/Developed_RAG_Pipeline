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
