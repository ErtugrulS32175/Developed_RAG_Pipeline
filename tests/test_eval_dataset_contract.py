import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
import uuid

import pytest

from pipeline.evaluation import datasets


KEY_A = "11111111-1111-4111-8111-111111111111"
KEY_B = "22222222-2222-4222-8222-222222222222"
SENTINELS = (
    "PRIVATE_QUESTION_SENTINEL",
    "PRIVATE_KEY_SENTINEL",
    "PRIVATE_ANSWER_SENTINEL",
)


def _case(case_key=KEY_A, **overrides):
    value = {
        "case_key": case_key,
        "q": "kurgu soru",
        "key": "kurgu anahtar",
        "answer": "kurgu cevap",
        "pages": [7, 19],
        "type": "metin",
    }
    value.update(overrides)
    return value


def _version(*cases, version=1):
    return {"version": version, "cases": list(cases or (_case(),))}


def test_one_case_and_version_have_stable_canonical_digests():
    version = datasets.validate_version(_version(_case()))
    case = version.cases[0]

    expected_case = (
        b'{"answer":"kurgu cevap","case_key":"11111111-1111-4111-8111-'
        b'111111111111","key":"kurgu anahtar","pages":[7,19],'
        b'"q":"kurgu soru","type":"metin"}'
    )
    expected_version = b'{"cases":[' + expected_case + b'],"version":1}'
    assert case.canonical_bytes == expected_case
    assert version.canonical_bytes == expected_version
    assert case.sha256 == hashlib.sha256(expected_case).hexdigest()
    assert version.sha256 == hashlib.sha256(expected_version).hexdigest()


def test_object_member_order_does_not_change_the_digest():
    original = _case()
    reordered = {name: original[name] for name in reversed(tuple(original))}
    left = datasets.validate_version(_version(original))
    right = datasets.validate_version({"cases": [reordered], "version": 1})
    assert left.canonical_bytes == right.canonical_bytes
    assert left.sha256 == right.sha256


def test_validation_does_not_mutate_or_alias_the_caller():
    offered = _version(_case())
    before = copy.deepcopy(offered)
    validated = datasets.validate_version(offered)
    offered["cases"][0]["pages"].append(99)
    offered["cases"][0]["q"] = "changed"
    assert before == _version(_case())
    assert validated.cases[0].pages == (7, 19)
    assert validated.cases[0].q == "kurgu soru"
    with pytest.raises(FrozenInstanceError):
        validated.version = 2


@pytest.mark.parametrize("case_type", ["metin", "sayisal", "tablo"])
def test_the_three_closed_case_types_are_accepted(case_type):
    assert datasets.validate_case(_case(type=case_type)).type == case_type


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(extra="x"),
    lambda value: value.pop("answer"),
    lambda value: value.update(type="image"),
    lambda value: value.update(pages=[]),
    lambda value: value.update(pages=[7, 7]),
    lambda value: value.update(pages=[19, 7]),
    lambda value: value.update(pages=[True]),
    lambda value: value.update(pages=[0]),
])
def test_case_shape_type_and_pages_fail_closed(mutation):
    offered = _case()
    mutation(offered)
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_case(offered)


@pytest.mark.parametrize("field", ["q", "key", "answer"])
@pytest.mark.parametrize("value", ["", " ", " leading", "trailing ", "a\nline", None])
def test_text_fields_are_nonempty_bounded_clean_utf8(field, value):
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_case(_case(**{field: value}))


@pytest.mark.parametrize("field, size", [
    ("q", datasets.TEXT_BYTE_LIMITS["q"] + 1),
    ("key", datasets.TEXT_BYTE_LIMITS["key"] + 1),
    ("answer", datasets.TEXT_BYTE_LIMITS["answer"] + 1),
])
def test_text_limits_count_utf8_bytes_not_characters(field, size):
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_case(_case(**{field: "z" * size}))
    # This has fewer characters than the byte limit but exceeds it in UTF-8.
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_case(_case(**{field: "\u011f" * (size // 2 + 1)}))


@pytest.mark.parametrize("case_key", [
    "11111111-1111-1111-8111-111111111111",
    "11111111-1111-4111-8111-11111111111A",
    "{11111111-1111-4111-8111-111111111111}",
    "not-a-uuid",
    7,
])
def test_case_keys_are_canonical_uuid4_values(case_key):
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_case(_case(case_key=case_key))


def test_new_case_keys_are_random_canonical_uuid4_values():
    first = datasets.new_case_key()
    second = datasets.new_case_key()
    assert first != second
    assert first == str(uuid.UUID(first))
    assert uuid.UUID(first).version == 4


def test_duplicate_and_noncanonical_case_order_are_refused():
    with pytest.raises(datasets.EvalDatasetError, match="^eval_case_key_duplicate$"):
        datasets.validate_version(_version(_case(), _case()))
    with pytest.raises(datasets.EvalDatasetError, match="^eval_order_invalid$"):
        datasets.validate_version(_version(_case(KEY_B), _case(KEY_A)))
    accepted = datasets.validate_version(_version(_case(KEY_A), _case(KEY_B)))
    assert tuple(case.case_key for case in accepted.cases) == (KEY_A, KEY_B)


@pytest.mark.parametrize("value", [
    {"version": 1, "cases": [], "extra": True},
    {"version": 1, "cases": []},
    {"version": True, "cases": [_case()]},
    {"version": 0, "cases": [_case()]},
    {"version": datasets.MAX_VERSION + 1, "cases": [_case()]},
])
def test_version_shape_number_and_minimum_count_are_closed(value):
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_version(value)


def test_a_version_contains_at_most_five_hundred_cases():
    cases = []
    for number in range(datasets.MAX_CASES):
        raw = uuid.UUID(int=number + 1, version=4)
        cases.append(_case(str(raw)))
    cases.sort(key=lambda case: case["case_key"])
    assert len(datasets.validate_version(_version(*cases)).cases) == 500
    extra = _case("ffffffff-ffff-4fff-bfff-ffffffffffff")
    with pytest.raises(datasets.EvalDatasetError):
        datasets.validate_version(_version(*(cases + [extra])))


def test_strict_json_rejects_duplicate_keys_bom_nonfinite_and_bad_utf8():
    valid = json.dumps(_version(_case()), separators=(",", ":")).encode()
    assert datasets.load_version_json(valid).version == 1
    duplicate = valid.replace(b'{"version":1', b'{"version":1,"version":1')
    attacks = [duplicate, b"\xef\xbb\xbf" + valid, b'{"version":NaN}', b"\xff"]
    for attack in attacks:
        with pytest.raises(datasets.EvalDatasetError):
            datasets.load_version_json(attack)


def test_legacy_projection_is_exact_and_fresh():
    case = datasets.validate_case(_case())
    projected = datasets.project_legacy_case(case)
    assert projected == {
        "q": "kurgu soru",
        "key": "kurgu anahtar",
        "answer": "kurgu cevap",
        "pages": [7, 19],
        "type": "metin",
    }
    assert "case_key" not in projected
    projected["pages"].append(99)
    assert case.pages == (7, 19)


def test_stable_db_api_interfaces_compose_without_aliasing():
    offered = [_case(KEY_A), _case(KEY_B, pages=[23], type="tablo")]
    normalized = datasets.normalize_cases(offered)
    assert type(normalized) is tuple
    assert tuple(normalized[0]) == (
        "case_key", "q", "key", "answer", "pages", "type")
    assert normalized[0]["pages"] == (7, 19)
    assert len(datasets.case_digest(normalized[0])) == 64
    assert len(datasets.version_digest(normalized)) == 64
    assert datasets.case_digest(normalized[0]) == datasets.case_digest(offered[0])
    assert datasets.version_digest(normalized) == datasets.version_digest(offered)

    legacy = datasets.project_legacy(normalized)
    assert type(legacy) is tuple
    assert set(legacy[0]) == {"q", "key", "answer", "pages", "type"}
    assert legacy[0]["pages"] == [7, 19]
    legacy[0]["pages"].append(101)
    assert normalized[0]["pages"] == (7, 19)


def test_content_digest_is_independent_of_the_storage_version_number():
    cases = [_case()]
    content = datasets.version_digest(cases)
    first = datasets.validate_version(_version(*cases, version=1))
    second = datasets.validate_version(_version(*cases, version=2))
    assert content == datasets.version_digest(datasets.normalize_cases(cases))
    assert first.sha256 != second.sha256


def test_direct_dataclass_construction_cannot_bypass_the_authority():
    forged = datasets.EvalCase(
        case_key=KEY_A,
        q="",
        key="kurgu anahtar",
        answer="kurgu cevap",
        pages=(7,),
        type="metin",
    )
    with pytest.raises(datasets.EvalDatasetError):
        datasets.canonical_case_bytes(forged)
    with pytest.raises(datasets.EvalDatasetError):
        datasets.project_legacy_case(forged)

    forged_version = datasets.EvalDatasetVersion(version=1, cases=(forged,))
    with pytest.raises(datasets.EvalDatasetError):
        datasets.canonical_version_bytes(forged_version)


def test_errors_never_echo_private_dataset_material():
    cases = [
        _case(q=SENTINELS[0] + "\n"),
        _case(key=SENTINELS[1] + "\n"),
        _case(answer=SENTINELS[2] + "\n"),
    ]
    rendered = []
    for case in cases:
        with pytest.raises(datasets.EvalDatasetError) as caught:
            datasets.validate_case(case)
        rendered.append(repr(caught.value))
        assert caught.value.args in {
            ("eval_text_invalid",),
            ("eval_case_invalid",),
        }
    report = " ".join(rendered)
    assert all(sentinel not in report for sentinel in SENTINELS)
