"""PACKAGE B2B-B2A1 -- the state binding's execution identity.

ONE question: which execution root does a run's binding name, and can a
binding be made to name two, or none.

WHY A BRIDGE AND NOT A SWITCH. Two execution surfaces are live at once:
the legacy disposable git worktree and the D3A flat workspace. Making
`workspace_id` the only accepted identity would have invalidated every
binding `worktree.create` writes -- and that module, plus four legacy
test files, are outside this package's authorised list. So the schema
accepts EXACTLY ONE of the two until B2B-B2C removes the worktree arm.

THE RULE IS IN THE SCHEMA, not in a later hand-written check. A binding
carrying neither identity names no place to run; one carrying both
leaves the choice to whichever reader looks first. Both are refused
before any caller sees the record.
"""
from __future__ import annotations

import pytest

from tools.agent_loop import contract, state

REPO_ID = "a" * 32
WORKTREE_ID = "b" * 32
WORKSPACE_ID = "c" * 32
BASELINE = "d" * 40
DIGEST = "e" * 64
RUN = "kosu-1"


def _binding(**overrides):
    kayit = {"protocol_version": contract.PROTOCOL_VERSION,
             "run_id": RUN, "repo_id": REPO_ID, "baseline_sha": BASELINE,
             "manifest_digest": DIGEST}
    kayit.update(overrides)
    return kayit


@pytest.fixture
def state_dir(tmp_path):
    yol = tmp_path / "durum"
    yol.mkdir()
    return yol


@pytest.mark.parametrize("alan,deger", [
    ("worktree_id", WORKTREE_ID),
    ("workspace_id", WORKSPACE_ID),
])
def test_exactly_one_execution_identity_is_accepted(state_dir, alan, deger):
    """Both arms of the bridge round-trip. The legacy one stays valid
    until B2B-B2C, and the flat one is what `changes.py` moves to."""
    state.write_binding(state_dir, _binding(**{alan: deger}))
    okunan = state.read_binding(state_dir)
    assert okunan[alan] == deger
    assert len(set(okunan) & {"worktree_id", "workspace_id"}) == 1


@pytest.mark.parametrize("nasil", ["ikisi-birden", "hicbiri"])
def test_a_binding_without_exactly_one_identity_is_refused(state_dir, nasil):
    """Neither shape has an answer to "where does this run execute".
    Refused at the WRITE, so a record like this never reaches disk."""
    if nasil == "ikisi-birden":
        kayit = _binding(worktree_id=WORKTREE_ID, workspace_id=WORKSPACE_ID)
    else:
        kayit = _binding()

    with pytest.raises(state.StateError):
        state.write_binding(state_dir, kayit)
    assert not (state_dir / state.BINDING_FILENAME).exists(), \
        "reddedilen baglama yine de diske yazildi"


@pytest.mark.parametrize("kotu", ["", "z" * 32, "a" * 31, "A" * 32, 5, None])
def test_a_workspace_identity_outside_the_opaque_grammar_is_refused(
        state_dir, kotu):
    """The same grammar the rest of the contract already owns; nothing
    is re-spelled here."""
    with pytest.raises(state.StateError):
        state.write_binding(state_dir, _binding(workspace_id=kotu))
    assert not (state_dir / state.BINDING_FILENAME).exists()


def test_the_workspace_grammar_is_the_contracts_opaque_id():
    """Structural pin: a second spelling of this pattern is a second
    place to get it wrong."""
    ozellikler = state.BINDING_SCHEMA["properties"]
    assert ozellikler["workspace_id"]["pattern"] == contract.OPAQUE_ID_PATTERN
    assert ozellikler["worktree_id"]["pattern"] == contract.OPAQUE_ID_PATTERN
    assert state.BINDING_SCHEMA["oneOf"] == [{"required": ["worktree_id"]},
                                             {"required": ["workspace_id"]}]
    # the identity is NOT in `required` -- `oneOf` is what enforces it,
    # and having both would make the neither-case impossible to express
    assert "worktree_id" not in state.BINDING_SCHEMA["required"]
    assert "workspace_id" not in state.BINDING_SCHEMA["required"]


def test_the_common_bonds_are_unchanged(state_dir):
    """The identity moved; nothing else did."""
    for alan in ("protocol_version", "run_id", "repo_id", "baseline_sha",
                 "manifest_digest"):
        assert alan in state.BINDING_SCHEMA["required"], alan
    state.write_binding(state_dir, _binding(workspace_id=WORKSPACE_ID))
    for alan, deger in (("repo_id", "f" * 32), ("baseline_sha", "0" * 40),
                        ("manifest_digest", "1" * 64)):
        with pytest.raises(state.IncompatibleState):
            state.assert_binding(
                state_dir,
                **{"repo_id": REPO_ID, "baseline_sha": BASELINE,
                   "manifest_digest": DIGEST, alan: deger})


@pytest.mark.parametrize("alan,dogru,yanlis", [
    ("worktree_id", WORKTREE_ID, "9" * 32),
    ("workspace_id", WORKSPACE_ID, "9" * 32),
])
def test_assert_binding_compares_the_execution_identity(state_dir, alan,
                                                        dogru, yanlis):
    ortak = {"repo_id": REPO_ID, "baseline_sha": BASELINE,
             "manifest_digest": DIGEST}
    state.write_binding(state_dir, _binding(**{alan: dogru}))
    assert state.assert_binding(state_dir, **ortak, **{alan: dogru})
    with pytest.raises(state.IncompatibleState) as ret:
        state.assert_binding(state_dir, **ortak, **{alan: yanlis})
    assert alan in str(ret.value)


def test_one_identity_can_never_be_answered_by_the_other(state_dir):
    """The trap this bridge creates and has to close. The schema lets
    exactly one key exist, so `binding.get("workspace_id")` on a legacy
    record is `None` -- and a caller who asked about a workspace must
    not be told yes just because the field is absent."""
    ortak = {"repo_id": REPO_ID, "baseline_sha": BASELINE,
             "manifest_digest": DIGEST}
    state.write_binding(state_dir, _binding(worktree_id=WORKTREE_ID))
    with pytest.raises(state.IncompatibleState) as ret:
        state.assert_binding(state_dir, **ortak, workspace_id=WORKSPACE_ID)
    assert "workspace_id" in str(ret.value)
    # and the mirror direction
    state.write_binding(state_dir, _binding(workspace_id=WORKSPACE_ID))
    with pytest.raises(state.IncompatibleState):
        state.assert_binding(state_dir, **ortak, worktree_id=WORKTREE_ID)


def test_a_refusal_carries_no_record_content_or_path(state_dir):
    """What may leave is the field NAME. Never the value, never the
    file's own location."""
    state.write_binding(state_dir, _binding(workspace_id=WORKSPACE_ID))
    with pytest.raises(state.IncompatibleState) as ret:
        state.assert_binding(state_dir, repo_id="f" * 32,
                             baseline_sha=BASELINE, manifest_digest=DIGEST)
    metin = str(ret.value) + repr(ret.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"
    for gizli in (REPO_ID, WORKSPACE_ID, BASELINE, DIGEST, str(state_dir)):
        assert gizli not in metin, "ret metni kayit icerigi tasiyor"
