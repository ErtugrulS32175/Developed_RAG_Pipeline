"""Mutation run over every mechanism B1 relies on -- IN A COPY.

The previous harness mutated the LIVE working tree and restored after
each battery. An evaluator observed a source file changing under them,
and when the run was interrupted the tree was left carrying a mutation:
a `check=False` that had replaced `check=True`, with the battery
reporting 116 green against it. That is a false green produced by the
verification tool itself, which is the worst place to have one.

So nothing here touches the real repository. The tree is copied once,
every mutation is applied inside the copy, and the originals are
hash-verified before and after. If the copy is destroyed mid-run,
nothing of value was in it.

A KILL IS NOT A RED. `-x` makes the battery stop at the first failure,
which is fast, but a mutation that happens to break an unrelated test
would look identical to one the target test caught. So each mutation
declares the test that is SUPPOSED to fail, and the run records which
test actually failed. A mismatch is reported as MISDIRECTED, never as a
kill.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE_REPO = Path(__file__).resolve().parents[2]
MODULES = ("state", "locking", "worktree", "preflight", "execution",
           "cli", "schemas")
NL = chr(10)

# The execution-binding block in `run_implementer`, verbatim: two R2A
# mutations rewrite it and both must keep matching the real source.
BINDING_TRY = (
    "    try:" + NL
    + "        cwd = worktree.assert_execution_binding(" + NL
    + "            call.repo, state_dir=call.state_dir, run_id=call.run_id,"
    + NL
    + "            worktree_id=call.worktree_id,"
    + " baseline_sha=call.baseline_sha)" + NL
    + "    except worktree.WorktreeError as refused:" + NL
    + "        raise WorktreeNotBound(str(refused)) from None")
BS = chr(92)

# (label, module, old, new, expected_test_substring)
MUTATIONS = [
    ("kilit-dislama", "locking",
     "    if not _try_lock(fd):" + NL + "        os.close(fd)",
     "    if False:" + NL + "        os.close(fd)",
     "first_owner_takes_the_lock"),
    ("kilit-ofseti", "locking",
     "    _LOCK_OFFSET = 1 << 30", "    _LOCK_OFFSET = 0",
     "unreadable_lock_record"),
    ("cift-birakma", "locking",
     "    if lock.released:" + NL + "        raise LockNotOwned",
     "    if False:" + NL + "        raise LockNotOwned",
     "releasing_twice"),
    ("inspect-tip-kapisi", "locking",
     "    if not isinstance(pid, int) or isinstance(pid, bool) " + BS,
     "    if False and isinstance(pid, bool) " + BS,
     "well_formed_but_wrong_typed_lock_record"),
    ("degismez-alanlar", "state",
     'IMMUTABLE_STATE_FIELDS = ("protocol_version", "run_id", "started_at",'
     + NL + '                          "baseline_sha")',
     "IMMUTABLE_STATE_FIELDS = ()",
     "legal_transition_cannot_rewrite"),
    ("sonlu-sayi", "state",
     "    _assert_finite(payload, what)" + NL
     + '    budget = payload.get("budget")',
     '    budget = payload.get("budget")',
     "non_finite_number_never_reaches_disk"),
    ("butce-degismezi", "state",
     "                and spent > ceiling:", "                and False:",
     "budget_invariant_is_enforced"),
    ("butce-siniri", "state",
     "                and spent > ceiling:",
     "                and spent >= ceiling:",
     "spending_exactly_the_ceiling"),
    ("yarim-cift", "state",
     "    if has_state != has_binding:" + NL + "        raise CorruptState(",
     "    if False:" + NL + "        raise CorruptState(",
     "half_a_state_directory"),
    ("run-id-caprazi", "state",
     '    if current["run_id"] != binding["run_id"]:', "    if False:",
     "two_documents_must_describe_the_same_run"),
    ("bitmemis-kosu", "state",
     '    if current["state"] not in contract.TERMINAL_STATES '
     "and not allow_resume:", "    if False:",
     "unfinished_previous_run"),
    ("zorunlu-worktree-id", "state",
     '                 "manifest_digest", "worktree_id"],',
     '                 "manifest_digest"],',
     "real_worktree_id_can_actually_be_written"),
    ("dizin-fsync", "state",
     "        fsync_directory(target.parent)", "        pass",
     "atomic_write_flushes_the_directory_entry"),
    ("yol-siniri", "preflight",
     '    return any(path == entry or path.startswith(entry + "/")',
     "    return any(path == entry or path.startswith(entry)",
     "dirty_allowlist_stops_at_a_path_segment"),
    ("durum-kosulsuz", "preflight",
     "    if state_dir is not None:",
     "    if state_dir is not None and (Path(state_dir)" + NL
     + "                                  / state_module.BINDING_FILENAME"
     + ").exists():",
     "half_a_state_directory"),
    ("kontrol-duzlemi-kapisi", "preflight",
     "    tampered = _control_plane_changes(repo_path)" + NL
     + "    if tampered:",
     "    tampered = _control_plane_changes(repo_path)" + NL + "    if False:",
     "modified_control_plane_stops_the_run"),
    ("kontrol-duzlemi-yol-listesi", "preflight",
     '    for field_name in ("allowed_paths", "dirty_tree_allowlist"):',
     '    for field_name in ("allowed_paths",):',
     "no_manifest_path_list_may_name_the_control_plane"),
    ("kontrol-duzlemi-dogrudan", "preflight",
     "    if _touches_control_plane(dirty_allowlist):", "    if False:",
     "caller_cannot_allowlist_the_control_plane"),
    ("kontrol-duzlemi-test-ailesi", "preflight",
     "    return tuple(contract.CONTROL_PLANE_PATHS) + tuple(" + NL
     + "        contract.CONTROL_PLANE_GLOBS)",
     "    return tuple(contract.CONTROL_PLANE_PATHS)",
     "whole_agent_loop_test_family_is_control_plane"),
    ("allowlist-test-ailesi", "preflight",
     "        if any(fnmatch.fnmatch(normalised, pattern)" + NL
     + "               for pattern in contract.CONTROL_PLANE_GLOBS):",
     "        if False:",
     "no_manifest_path_list_may_name_the_control_plane"),
    ("git-donus-kodu", "preflight",
     "    if done.returncode != 0:", "    if False:",
     "git_command_that_fails_is_not_a_clean_tree"),
    ("git-stderr-sizintisi", "preflight",
     'raise GitUnavailable(f"git {args[0]} (rc={done.returncode})")',
     'raise GitUnavailable(f"git {args[0]} {done.stderr}")',
     "failure_detail_never_carries_git_stderr"),
    ("handshake-baslatma", "preflight",
     "        subprocess.run([str(path), *HANDSHAKE_ARGV], capture_output=True,",
     "        subprocess.run([sys.executable, '-c', 'pass'], capture_output=True,",
     "file_that_cannot_be_launched"),
    ("handshake-timeout", "preflight",
     "                       timeout=HANDSHAKE_TIMEOUT_SECONDS,",
     "                       timeout=None,",
     "binary_that_never_returns"),
    ("kimlik-deseni", "worktree",
     "    if not WORKTREE_ID.match(str(worktree_id)):", "    if False:",
     "id_that_could_name_another_directory"),
    ("kayit-varligi", "worktree",
     "    if record is None:" + NL
     + '        raise WorktreeError("calisma agaci kaydi yok; silinmedi")',
     "    if record is None:" + NL + "        return None",
     "record_from_another_state_directory"),
    ("kayit-depo-yetkisi", "worktree",
     '    if record["repo_id"] != state_module.repo_identity(repo):',
     "    if False:",
     "repository_check_refuses_even_when_git_would_agree"),
    ("git-kayitli-mi", "worktree",
     "    if _registered_here(repo, path):", "    if True:",
     "killed_process_always_leaves_recoverable_residue"),
    ("git-kaldirma-kontrolu", "worktree",
     '        _git(repo, "worktree", "remove", "--force", str(path))',
     '        _git(repo, "worktree", "remove", "--force", str(path),'
     " check=False)",
     "failed_git_removal_is_not_stepped_over"),
    ("kurtarma-depo-filtresi", "worktree",
     '        if record is None or record["repo_id"] != repo_id:',
     "        if record is None:",
     "recovery_does_not_report_another_repositorys"),
    ("kap-dislayici", "worktree",
     "        holder.mkdir()", "        holder.mkdir(exist_ok=True)",
     "existing_holder_is_never_reused"),
    ("registry-sorgusu-kontrolu", "worktree",
     '    listing = _git(repo, "worktree", "list", "--porcelain").stdout',
     '    listing = _git(repo, "worktree", "list", "--porcelain",'
     " check=False).stdout",
     "unanswerable_git_query_deletes_nothing"),
    ("holder-silindi-mi", "worktree",
     "    if holder.exists():" + NL
     + '        raise WorktreeError("calisma agaci kabi silinemedi; '
     'kayit korundu")',
     "    if False:" + NL
     + '        raise WorktreeError("calisma agaci kabi silinemedi; '
     'kayit korundu")',
     "record_outlives_a_holder_that_could_not_be_removed"),
    ("dizin-zinciri", "state",
     "    ensure_directory(target.parent)",
     "    target.parent.mkdir(parents=True, exist_ok=True)",
     "brand_new_directory_chain_is_made_durable"),
    ("yaz-once-kaydi", "worktree",
     "    state_module.write_json_atomically(" + NL
     + '        _record_path(state_dir, record["worktree_id"]), record,' + NL
     + '        RECORD_SCHEMA, "calisma agaci kaydi")' + NL
     + "    return record",
     "    return record",
     "record_is_written_before_anything_exists"),
    # ----------------------------------------------------------------
    # R2A -- the execution binding. Each mutation deletes exactly one
    # refusal and its target test must be the one that notices.
    # ----------------------------------------------------------------
    ("r2a-hazir-kontrolu", "worktree",
     '        raise WorktreeError("yurutme bagi: kayit hazir durumda degil")',
     "        pass",
     "test_a_planned_record_is_refused_even_with_the_tree_on_disk"),
    ("r2a-kayit-depo", "worktree",
     '        raise WorktreeError("yurutme bagi: kayit bu depoya ait degil")',
     "        pass",
     "test_a_record_issued_to_a_different_repository_identity_is_refused"),
    ("r2a-kayit-kosu", "worktree",
     '        raise WorktreeError("yurutme bagi: kayit bu kosuya ait degil")',
     "        pass",
     "test_a_record_whose_run_was_rewritten_is_refused"),
    ("r2a-kayit-taban", "worktree",
     '        raise WorktreeError("yurutme bagi: kayit taban surumle '
     'uyusmuyor")',
     "        pass",
     "test_a_record_whose_baseline_was_rewritten_is_refused"),
    ("r2a-gomulu-kimlik", "worktree",
     '        raise WorktreeError("yurutme bagi: kayit baska bir kimligi '
     'adliyor")',
     "        pass",
     "test_a_record_copied_under_another_id_is_refused"),
    ("r2a-bag-varligi", "worktree",
     "    except state_module.CorruptState:" + NL
     + "        raise WorktreeError(" + NL
     + '            "yurutme bagi: kosu baglamasi yok ya da bozuk") from None',
     "    except state_module.CorruptState:" + NL
     + '        binding = {"worktree_id": worktree_id, "repo_id": repo_id,'
     + NL
     + '                   "run_id": run_id, "baseline_sha": baseline_sha}',
     "test_a_missing_run_binding_is_refused"),
    ("r2a-bag-agac", "worktree",
     "        raise WorktreeError(" + NL
     + '            "yurutme bagi: kosu bu calisma agacina bagli degil")',
     "        pass",
     "test_a_second_ready_worktree_the_binding_does_not_name_is_refused"),
    ("r2a-bag-depo", "worktree",
     "        raise WorktreeError(" + NL
     + '            "yurutme bagi: kosu baglamasi bu depoya ait degil")',
     "        pass",
     "test_a_binding_issued_to_a_different_repository_is_refused"),
    ("r2a-bag-kosu", "worktree",
     "        raise WorktreeError(" + NL
     + '            "yurutme bagi: kosu baglamasi bu kosuya ait degil")',
     "        pass",
     "test_a_binding_for_a_different_run_is_refused"),
    ("r2a-bag-taban", "worktree",
     "        raise WorktreeError(" + NL
     + '            "yurutme bagi: kosu baglamasi taban surumle uyusmuyor")',
     "        pass",
     "test_a_binding_at_a_different_baseline_is_refused"),
    ("r2a-bas-kontrolu", "worktree",
     '    head = _git(derived, "rev-parse", "HEAD").stdout.strip()' + NL
     + "    if head != baseline_sha:" + NL
     + '        raise WorktreeError("yurutme bagi: calisma agaci taban '
     'surumde degil")' + NL
     + "    return derived",
     "    return derived",
     "test_a_worktree_whose_head_moved_is_refused"),
    ("r2a-git-kutugu", "worktree",
     "        raise WorktreeError(" + NL
     + '            "yurutme bagi: calisma agaci bu depoda kayitli degil")',
     "        pass",
     "test_a_copied_tree_that_git_does_not_register_is_refused"),
    ("r2a-ana-agac", "worktree",
     '        raise WorktreeError("yurutme bagi: ana calisma agaci '
     'yurutulemez")',
     "        pass",
     "test_the_repository_argument_may_never_be_the_execution_target"),
    ("r2a-kap-sinirlama", "worktree",
     '        raise WorktreeError("yurutme bagi: yol kabin disina '
     'cozuluyor")',
     "        pass",
     "test_a_link_that_resolves_outside_the_holder_is_refused"),
    ("r2a-bag-cagrisi-yok", "execution",
     BINDING_TRY,
     "    cwd = worktree.holder_for(worktree_id) / worktree.WORKTREE_DIRNAME",
     "test_a_missing_record_is_refused"),
    ("r2a-cagiran-yolu", "execution",
     BINDING_TRY,
     BINDING_TRY.replace("cwd = worktree.assert_execution_binding(",
                         "worktree.assert_execution_binding(")
     + NL + "    cwd = Path(repo)",
     "test_the_model_runs_exactly_in_the_derived_recorded_worktree"),
    # ----------------------------------------------------------------
    # R2B -- one canonical inline schema: bytes, hash, argv, validator
    # ----------------------------------------------------------------
    ("r2b-yol-degeri", "cli",
     '        "--json-schema", schemas.IMPLEMENTER_SCHEMA_BINDING.'
     'canonical_json,',
     '        "--json-schema", str(binary),',
     "test_the_argv_schema_is_inline_canonical_and_hashed"),
    ("r2b-kanonik-sirasiz", "schemas",
     '    return json.dumps(document, sort_keys=True, '
     'separators=(",", ":"),',
     '    return json.dumps(document, sort_keys=False, '
     'separators=(",", ":"),',
     "test_canonical_json_is_order_independent_compact_and_deterministic"),
    ("r2b-ham-sozluk", "execution",
     "        validator.validate(payload)",
     "        Draft202012Validator(" + NL
     + "            schemas.IMPLEMENTER_RESULT_SCHEMA).validate(payload)",
     "test_the_validator_cannot_be_the_raw_module_dictionary"),
    ("r2b-esitlik-kontrolu", "execution",
     "    if text != binding.canonical_json " + BS + NL
     + "            or hashlib.sha256(payload).hexdigest() != "
     "binding.sha256:" + NL
     + '        raise SchemaNotBound("argv\'deki sema kanonik baglamayla '
     'eslesmiyor")',
     "    if False:" + NL
     + '        raise SchemaNotBound("argv\'deki sema kanonik baglamayla '
     'eslesmiyor")',
     "test_a_builder_that_smuggles_a_different_schema_is_refused_before_"
     "launch"),
    # ----------------------------------------------------------------
    # R2B-R1 -- the binding is frozen and every divergence is TYPED
    # ----------------------------------------------------------------
    ("r2br1-dondurma", "schemas",
     "    def __setattr__(self, name, value):" + NL
     + '        raise AttributeError("baglama donduruldu; alan yeniden '
     'yazilamaz")',
     "    def __setattr__(self, name, value):" + NL
     + "        object.__setattr__(self, name, value)",
     "test_the_binding_refuses_attribute_rewrites"),
    ("r2br1-tip-kontrolu", "execution",
     "    if type(text) is not str:" + NL
     + '        raise SchemaNotBound("argv\'deki sema degeri metin degil")',
     "    if False:" + NL
     + '        raise SchemaNotBound("argv\'deki sema degeri metin degil")',
     "test_a_malformed_schema_token_is_the_same_typed_refusal"),
    # The exact-type check, weakened to the isinstance form that let a
    # lying `str` subclass through. Only the impersonation test sees it.
    ("r2br11-alt-sinif", "execution",
     "    if type(text) is not str:",
     "    if not isinstance(text, str):",
     "test_a_lying_str_subclass_cannot_impersonate_the_schema"),
    ("r2br1-kodlama-sarici", "execution",
     "    except UnicodeEncodeError:" + NL
     + "        raise SchemaNotBound(" + NL
     + '            "argv\'deki sema UTF-8\'e kodlanamiyor") from None',
     "    except UnicodeEncodeError:" + NL
     + "        raise",
     "test_a_malformed_schema_token_is_the_same_typed_refusal"),
    ("r2b-yanlis-sha", "execution",
     "    return Draft202012Validator(json.loads(text)), binding.sha256",
     "    return Draft202012Validator(json.loads(text)), "
     "binding.sha256[::-1]",
     "test_the_argv_schema_is_inline_canonical_and_hashed"),
    # ----------------------------------------------------------------
    # B2A -- the call boundary: validate once, canonicalize once, use
    # only the canonical value. Each mutation reopens exactly one of
    # the check/use divergences the package closed.
    # ----------------------------------------------------------------
    ("b2a-arac-tipi", "cli",
     "    if any(type(tool) is not str for tool in requested):" + NL
     + '        raise UnsafeInvocation("arac adlari tam metin olmalidir")',
     "    if False:" + NL
     + '        raise UnsafeInvocation("arac adlari tam metin olmalidir")',
     "test_a_deceptive_tool_object_can_never_reach_the_argv"),
    ("b2a-ikili-donusu", "execution",
     "        binary=_usable_binary(binary),",
     "        binary=(_usable_binary(binary), Path(str(binary)))[1],",
     "test_the_checked_binary_is_the_launched_binary"),
    # B2A-R1 moved the budget rule into `cli`, the one authority both
    # roads use; these two follow it there.
    ("b2a-butce-tavani", "cli",
     "    if budget_usd > MAX_BUDGET_USD:" + NL
     + '        raise UnsafeInvocation("butce sozlesme tavanini asiyor")',
     "    if False:" + NL
     + '        raise UnsafeInvocation("butce sozlesme tavanini asiyor")',
     "test_a_budget_above_the_schema_maximum_is_refused"),
    ("b2a-butce-tipi", "cli",
     "    if type(budget_usd) not in (int, float):",
     "    if not isinstance(budget_usd, (int, float)):",
     "test_a_deceptive_budget_never_reaches_the_argv"),
    # ----------------------------------------------------------------
    # B2A-R1 -- the three contract gaps
    # ----------------------------------------------------------------
    ("b2ar1-builder-butce", "cli",
     "    budget_usd = exact_budget(budget_usd)",
     "    exact_budget",
     "test_the_public_builder_enforces_the_budget_bounds_itself"),
    ("b2ar1-tipli-ret", "execution",
     "    except cli.UnsafeInvocation as refused:" + NL
     + "        raise CallInputRefused(str(refused)) from None",
     "    except cli.UnsafeInvocation:" + NL
     + "        raise",
     "test_every_cli_refusal_reaches_the_caller_as_an_adapter_error"),
    ("b2ar1-arac-yansimasi", "cli",
     '            f"implementer izinli olmayan {len(forbidden)} arac '
     'istedi; "',
     '            f"implementer bu araclari alamaz: {sorted(forbidden)}; "',
     "test_a_refused_tool_name_is_never_echoed_back"),
    ("b2a-sure-tipi", "execution",
     "    if type(timeout_seconds) is not int:",
     "    if not isinstance(timeout_seconds, int):",
     "test_a_deceptive_integer_bound_is_refused_before_any_process"),
    ("b2a-istem-tipi", "execution",
     "    if type(prompt) is not str:",
     "    if not isinstance(prompt, str):",
     "test_a_deceptive_prompt_never_reaches_stdin"),
    ("b2a-argv-token-tipi", "cli",
     "    tokens = []" + NL
     + "    for token in argv:" + NL
     + "        if type(token) is not str:" + NL
     + '            raise UnsafeInvocation("argv tam metin olmayan bir oge '
     'tasiyor")' + NL
     + "        tokens.append(token)",
     "    tokens = [str(token) for token in argv]",
     "test_assert_safe_argv_refuses_a_token_that_is_not_an_exact_string"),
    ("b2a-model-tipi", "cli",
     "    if type(model) is not str or len(model) > MODEL_MAX_LENGTH " + BS
     + NL + "            or not MODEL_PATTERN.fullmatch(model):" + NL
     + '        raise UnsafeInvocation("model adi sozlesme desenine uymuyor")',
     "    if False:" + NL
     + '        raise UnsafeInvocation("model adi sozlesme desenine uymuyor")',
     "test_the_builder_refuses_a_model_outside_the_frozen_schema"),
    ("b2a-kimlik-tipi", "execution",
     "    if type(value) is not str or not pattern.fullmatch(value):" + NL
     + '        raise IdentityRefused(f"{what} sozlesme desenine uymuyor")',
     "    if False:" + NL
     + '        raise IdentityRefused(f"{what} sozlesme desenine uymuyor")',
     "test_a_deceptive_identity_never_reaches_the_worktree_binding"),
]


def verdict_exit_code(baseline_rc, results, intact):
    """The process exit code, derived from EVERYTHING that can go wrong.

    This used to be implicit: every path fell off the end of `main` and
    the process exited 0 -- a red baseline, a missed mutation, a
    misdirected kill and even a modified live tree all LOOKED like
    success to any automation reading the exit code. Printed text is
    not a verdict; the exit code is.

      0  every mutation YAKALANDI, baseline green, live tree untouched
      1  at least one verdict is not YAKALANDI
      2  the baseline battery was red, or nothing was judged at all
      3  the live tree changed while the harness ran
    """
    if baseline_rc != 0:
        return 2
    if not intact:
        return 3
    if not results:
        return 2
    if any(entry["hukum"] != "YAKALANDI" for entry in results):
        return 1
    return 0


def digest(path):
    return hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def originals():
    return {m: digest(SOURCE_REPO / "tools" / "agent_loop" / (m + ".py"))
            for m in MODULES}


def _pytest(workdir, extra):
    # R2B mutation targets live in the CONTRACT battery too (the
    # canonicalization rules are contract tests), so it runs here.
    argv = [sys.executable, "-m", "pytest", "tests/test_agent_loop_b1.py",
            "tests/test_agent_loop_b2_execution.py",
            "tests/test_agent_loop_contract.py",
            "-q", "--no-header", "-p", "no:cacheprovider", "-rf"] + extra
    # a temp directory of the HARNESS's own. The battery derives its
    # worktree root from the process temp directory, so a run that
    # inherited the shared one would build and delete holders in the
    # same place a real agent loop keeps its own.
    private_temp = workdir.parent / "tmp"
    private_temp.mkdir(exist_ok=True)
    environment = dict(os.environ)
    environment.update({"TMPDIR": str(private_temp), "TEMP": str(private_temp),
                        "TMP": str(private_temp)})
    done = subprocess.run(argv, cwd=str(workdir), capture_output=True,
                          text=True, env=environment)
    failing = [l.split("::")[-1].split(" ")[0]
               for l in done.stdout.splitlines() if l.startswith("FAILED ")]
    tail = [l for l in done.stdout.splitlines()
            if " passed" in l or " failed" in l or "error" in l.lower()]
    return done.returncode, (tail[-1] if tail else ""), failing


def judge(workdir, expected):
    """Ask the TARGET test first. It is small, so this is also the fast
    path; and its verdict is unambiguous, which `-x` over the whole
    battery is not."""
    rc, summary, failing = _pytest(workdir, ["-k", expected])
    if rc != 0 and failing:
        return "YAKALANDI", summary, failing
    rc, summary, failing = _pytest(workdir, [])
    if rc == 0:
        return "KACIRILDI", summary, []
    return "YANLIS-HEDEF", summary, failing


def run_battery(workdir):
    return _pytest(workdir, [])


def main():
    before = originals()
    workroot = Path(tempfile.mkdtemp(prefix="b1mut-copy-"))
    workdir = workroot / "repo"
    print("kopya kuruluyor...", flush=True)
    # ONLY what the battery needs. The repository also carries several
    # vendored virtualenvs whose nested paths exceed Windows MAX_PATH,
    # and copying them fails outright -- but they are also irrelevant:
    # the mutation targets and the tests are all that matters here.
    workdir.mkdir(parents=True)
    for tree in ("tools", "tests", "pipeline", "eval", "scripts", "services"):
        source = SOURCE_REPO / tree
        if source.is_dir():
            shutil.copytree(source, workdir / tree,
                            ignore=shutil.ignore_patterns(
                                "__pycache__", ".pytest_cache", "*.pyc"))
    for loose in ("pytest.ini", "requirements.txt", ".gitignore"):
        if (SOURCE_REPO / loose).is_file():
            shutil.copy2(SOURCE_REPO / loose, workdir / loose)
    # The battery includes contract tests that ask GIT about the
    # repository (tracked scripts, ignored state directory), so the
    # copy has to BE a repository; without this the baseline is red for
    # a reason that has nothing to do with any mutation.
    for git_argv in (("init", "-q"), ("add", "-A"),
                     ("-c", "user.email=k@example.invalid",
                      "-c", "user.name=Kurgu", "commit", "-qm", "kurgu")):
        subprocess.run(["git", "-C", str(workdir), *git_argv],
                       capture_output=True, text=True)

    targets = {m: workdir / "tools" / "agent_loop" / (m + ".py")
               for m in MODULES}

    rc, summary, _ = run_battery(workdir)
    print("BASELINE (kopyada): rc=" + str(rc) + " " + summary, flush=True)
    if rc != 0:
        print("BASELINE RED -- sonuclar anlamsiz olurdu.")
        shutil.rmtree(workroot, ignore_errors=True)
        return verdict_exit_code(rc, [], True)

    results = []
    for label, module, old, new, expected in MUTATIONS:
        path = targets[module]
        original = path.read_text(encoding="utf-8")
        if old not in original:
            results.append({"mutasyon": label, "hukum": "UYGULANAMADI",
                            "beklenen": expected, "kirilan": []})
            print("  " + label.ljust(30) + "UYGULANAMADI", flush=True)
            continue
        path.write_text(original.replace(old, new, 1), encoding="utf-8")
        imported = subprocess.run(
            [sys.executable, "-c",
             "from tools.agent_loop import (state, locking, worktree, "
             "preflight, execution, cli, schemas)"],
            cwd=str(workdir), capture_output=True, text=True)
        if imported.returncode != 0:
            path.write_text(original, encoding="utf-8")
            results.append({"mutasyon": label, "hukum": "GECERSIZ",
                            "beklenen": expected, "kirilan": []})
            print("  " + label.ljust(30) + "GECERSIZ (import kirildi)",
                  flush=True)
            continue
        verdict, summary, failing = judge(workdir, expected)
        path.write_text(original, encoding="utf-8")
        results.append({"mutasyon": label, "hukum": verdict,
                        "beklenen": expected, "kirilan": failing,
                        "ozet": summary})
        print("  " + label.ljust(30) + verdict.ljust(13)
              + (failing[0] if failing else summary), flush=True)

    after = originals()
    intact = before == after
    print("=" * 68)
    print("ANA AGAC DOKUNULMADI: " + str(intact))
    for module in MODULES:
        print("  " + module.ljust(10) + before[module][:16]
              + (" == " if before[module] == after[module] else " != ")
              + after[module][:16])
    killed = sum(1 for r in results if r["hukum"] == "YAKALANDI")
    print(str(killed) + "/" + str(len(results)) + " mutasyon HEDEF testiyle "
          "yakalandi")
    for r in results:
        if r["hukum"] != "YAKALANDI":
            print("  !! " + r["mutasyon"] + " -> " + r["hukum"]
                  + " (beklenen: " + r["beklenen"] + ", kirilan: "
                  + str(r["kirilan"][:3]) + ")")
    # a RUN OUTPUT, not source: writing it beside the tool left an
    # untracked artefact in the repository after every run
    report = Path(tempfile.gettempdir()) / "agent-loop-mutation-report.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print("makinece okunabilir rapor: " + str(report))
    shutil.rmtree(workroot, ignore_errors=True)
    return verdict_exit_code(0, results, intact)


if __name__ == "__main__":
    sys.exit(main())
