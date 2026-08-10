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
MODULES = ("state", "locking", "worktree", "preflight")
NL = chr(10)
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
]


def digest(path):
    return hashlib.sha256(
        path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def originals():
    return {m: digest(SOURCE_REPO / "tools" / "agent_loop" / (m + ".py"))
            for m in MODULES}


def _pytest(workdir, extra):
    argv = [sys.executable, "-m", "pytest", "tests/test_agent_loop_b1.py",
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
    for loose in ("pytest.ini", "requirements.txt"):
        if (SOURCE_REPO / loose).is_file():
            shutil.copy2(SOURCE_REPO / loose, workdir / loose)

    targets = {m: workdir / "tools" / "agent_loop" / (m + ".py")
               for m in MODULES}

    rc, summary, _ = run_battery(workdir)
    print("BASELINE (kopyada): rc=" + str(rc) + " " + summary, flush=True)
    if rc != 0:
        print("BASELINE RED -- sonuclar anlamsiz olurdu.")
        shutil.rmtree(workroot, ignore_errors=True)
        return

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
             "from tools.agent_loop import state, locking, worktree, preflight"],
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


if __name__ == "__main__":
    main()
