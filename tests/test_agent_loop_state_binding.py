"""PACKAGE B2B-B2A1 / B2B-B2C -- the state binding's execution identity.

ONE question: which execution root does a run's binding name, and can a
binding be made to name none, or one that no longer exists.

THE BRIDGE IS GONE. B2B-B2A1 let the schema accept EXACTLY ONE of
`worktree_id` and `workspace_id`, because two execution surfaces were
live at once and making `workspace_id` the only identity would have
invalidated every binding the old `worktree.create` wrote. B2B-B2C
removed the worktree surface and the module behind it, so the rule
collapsed into the simplest one available: `workspace_id` is REQUIRED,
`worktree_id` is not an optional field but an UNKNOWN one, and
`additionalProperties: False` is what refuses a stale document carrying
it.

THE RULE IS IN THE SCHEMA, not in a later hand-written check. A binding
carrying no identity names no place to run, and it is refused before any
caller sees the record.

WHY `assert_binding` STILL TAKES `workspace_id=None`. That is the
GENERIC state-directory check -- shape plus the identities the caller
actually knows -- not an optional identity. Nothing reaches that
comparison without a real `workspace_id` on disk, because the schema
already required one; and every identity-bound caller passes the value.
Both halves of that distinction are pinned below.
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
             "manifest_digest": DIGEST, "workspace_id": WORKSPACE_ID}
    kayit.update(overrides)
    return kayit


@pytest.fixture
def state_dir(tmp_path):
    yol = tmp_path / "durum"
    yol.mkdir()
    return yol


def test_a_workspace_binding_round_trips(state_dir):
    state.write_binding(state_dir, _binding())
    okunan = state.read_binding(state_dir)
    assert okunan["workspace_id"] == WORKSPACE_ID
    assert "worktree_id" not in okunan


@pytest.mark.parametrize("nasil", ["kimliksiz", "yalniz-worktree",
                                   "ikisi-birden"])
def test_a_binding_without_exactly_this_identity_is_refused(state_dir, nasil):
    """Three shapes with no answer to "where does this run execute", and
    each is refused at the WRITE so a record like it never reaches disk.

    The last two are the same refusal for a new reason: `worktree_id` is
    an unknown property now, so a stale legacy document and a
    both-identities document both die on `additionalProperties`."""
    kayit = _binding()
    if nasil == "kimliksiz":
        kayit.pop("workspace_id")
    elif nasil == "yalniz-worktree":
        kayit.pop("workspace_id")
        kayit["worktree_id"] = WORKTREE_ID
    else:
        kayit["worktree_id"] = WORKTREE_ID

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


def test_the_schema_names_one_identity_and_requires_it():
    """Structural pin: a second spelling of this pattern is a second
    place to get it wrong, and an identity that is merely ALLOWED is one
    a binding can be written without."""
    sema = state.BINDING_SCHEMA
    assert sema["properties"]["workspace_id"]["pattern"] == \
        contract.OPAQUE_ID_PATTERN
    assert "workspace_id" in sema["required"]
    assert "worktree_id" not in sema["properties"], \
        "eski yurutme kimligi hala semada"
    assert "oneOf" not in sema, "iki kimlik koprusu hala duruyor"
    assert sema["additionalProperties"] is False


def test_the_common_bonds_are_unchanged(state_dir):
    """The identity changed; nothing else did."""
    for alan in ("protocol_version", "run_id", "repo_id", "baseline_sha",
                 "manifest_digest"):
        assert alan in state.BINDING_SCHEMA["required"], alan
    state.write_binding(state_dir, _binding())
    for alan, deger in (("repo_id", "f" * 32), ("baseline_sha", "0" * 40),
                        ("manifest_digest", "1" * 64)):
        with pytest.raises(state.IncompatibleState):
            state.assert_binding(
                state_dir,
                **{"repo_id": REPO_ID, "baseline_sha": BASELINE,
                   "manifest_digest": DIGEST, alan: deger})


def test_assert_binding_compares_the_execution_identity(state_dir):
    ortak = {"repo_id": REPO_ID, "baseline_sha": BASELINE,
             "manifest_digest": DIGEST}
    state.write_binding(state_dir, _binding())
    assert state.assert_binding(state_dir, **ortak,
                                workspace_id=WORKSPACE_ID)
    with pytest.raises(state.IncompatibleState) as ret:
        state.assert_binding(state_dir, **ortak, workspace_id="9" * 32)
    assert "workspace_id" in str(ret.value)


def test_the_generic_check_verifies_shape_without_naming_an_identity(
        state_dir):
    """The distinction this package had to keep. A caller that does not
    know the workspace id still gets the document validated and the
    common bonds compared -- and it is never told "yes" ABOUT an identity
    it did not name.

    That is not the same as the identity being optional: the record on
    disk always carries a real one, which is why the generic call below
    can be answered at all."""
    state.write_binding(state_dir, _binding())
    binding = state.assert_binding(state_dir, repo_id=REPO_ID,
                                   baseline_sha=BASELINE,
                                   manifest_digest=DIGEST)
    assert binding["workspace_id"] == WORKSPACE_ID, \
        "genel kontrol kimliksiz bir kaydi kabul etti"


def test_the_old_execution_identity_cannot_be_asked_about():
    """The bridge's mirror-direction trap, closed by removal rather than
    by a check: there is no parameter left to ask it with."""
    import inspect

    parametreler = inspect.signature(state.assert_binding).parameters
    assert "workspace_id" in parametreler
    assert "worktree_id" not in parametreler
    with pytest.raises(TypeError):
        state.assert_binding("kurgu", repo_id=REPO_ID, baseline_sha=BASELINE,
                             manifest_digest=DIGEST, worktree_id=WORKTREE_ID)


def test_a_refusal_carries_no_record_content_or_path(state_dir):
    """What may leave is the field NAME. Never the value, never the
    file's own location."""
    state.write_binding(state_dir, _binding())
    with pytest.raises(state.IncompatibleState) as ret:
        state.assert_binding(state_dir, repo_id="f" * 32,
                             baseline_sha=BASELINE, manifest_digest=DIGEST)
    metin = str(ret.value) + repr(ret.value)
    assert "/" not in metin and chr(92) not in metin, "ret metni yol tasiyor"
    for gizli in (REPO_ID, WORKSPACE_ID, BASELINE, DIGEST, str(state_dir)):
        assert gizli not in metin, "ret metni kayit icerigi tasiyor"
