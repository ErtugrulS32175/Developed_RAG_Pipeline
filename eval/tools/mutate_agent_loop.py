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
MODULES = ("state", "locking", "preflight", "execution",
           "cli", "schemas", "changes", "acceptance", "application",
           # B3: the evaluator adapter and the runner that joins every
           # phase. A module absent from this tuple cannot be mutated at
           # all, so its guards would be pinned by nothing here.
           "audit", "runner", "runner_events",
           # B4-R8: the frozen vocabulary itself. The failure-code table
           # lives here, and a mutation that files one road's mechanism
           # under the other road's code could not be expressed while
           # this module was absent -- the harness raised `KeyError`
           # instead of judging it.
           "contract")
NL = chr(10)

BS = chr(92)

# THE BATTERY, as one list the harness runs and the pin reads.
#
# EVERY mutation's expected-target has to be reachable inside one of these
# files. A target that cannot be EXECUTED is not a missing guard: `-k`
# matches nothing, the run falls through to the full battery and reports
# KACIRILDI, which reads exactly like the guard being absent.
#
# MEASURED THREE TIMES, which is why this is a constant now instead of a
# literal inside `_pytest` with a hand-maintained twin in the pin. The
# R2A/B2B batteries drifted apart from the namespace list twice
# (`b2bc2-*` named the application battery, `zorunlu-workspace-id` named
# the state-binding battery) and neither file was being run -- and the
# second cost a full 27-minute battery run per misjudgement before
# anything said so. The pin derives its namespace from THIS tuple, so the
# two cannot disagree again.
BATTERY = (
    "tests/test_agent_loop_b1.py",
    "tests/test_agent_loop_b2_execution.py",
    # R2B targets live in the CONTRACT battery too -- the canonicalization
    # rules are contract tests.
    "tests/test_agent_loop_contract.py",
    "tests/test_agent_loop_b2_changes.py",
    # B2B-B2A2: the migrated change-set mechanism.
    "tests/test_agent_loop_b2_changes_flat.py",
    # B2B-B2B: the main-checkout guard's own battery.
    "tests/test_agent_loop_b2_main_guard.py",
    # B2B-B2C: where the required execution identity is judged.
    "tests/test_agent_loop_state_binding.py",
    # B2B-C1 and B2B-C2: the acceptance and application batteries.
    "tests/test_agent_loop_b2_acceptance.py",
    "tests/test_agent_loop_b2_application.py",
    # B3: the evaluator adapter and the runner.
    "tests/test_agent_loop_b3_audit.py",
    "tests/test_agent_loop_b3_runner.py",
)

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
    # B2B-B2A1 moved this guard out of `required` and into an `oneOf`
    # gate, because the schema had to carry either a worktree or a flat
    # workspace and "exactly one" is not something `required` can say.
    #
    # B2B-B2C RETARGETED and RENAMED it back. There is one execution
    # surface now, so the gate collapsed into the simplest thing that
    # can express it -- and the intent never moved: a binding with no
    # execution identity names no place to run.
    ("zorunlu-workspace-id", "state",
     '                 "manifest_digest", "workspace_id"],',
     '                 "manifest_digest"],',
     "a_binding_without_exactly_this_identity_is_refused"),
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
    # ----------------------------------------------------------------
    # B5-R3 -- WHICH schema rule refused, as two closed words. Measured:
    # a real run recorded `implementer_schema_violation` and nothing
    # else, and the envelope that would have explained it existed only
    # in the child's stdout.
    # ----------------------------------------------------------------
    # 1. Every validator collapsed to the generic word: the journal
    #    would say a rule broke without saying which, which is the
    #    state this package exists to leave.
    ("b5r3-validator-sinifi", "contract",
     '    "required": SchemaIssue.REQUIRED,',
     '    "required": SchemaIssue.UNKNOWN,',
     "test_a_missing_required_field_is_named_from_the_schema"),
    # 2. The field pinned to the root: an operator sent to "the reply"
    #    when the schema knows the field.
    ("b5r3-alan-daima-kok", "execution",
     "    for part in path:" + NL + "        if isinstance(part, str):",
     "    for part in []:" + NL + "        if isinstance(part, str):",
     "test_a_broken_rule_names_the_declared_field_it_broke"),
    # 3. The missing-field computation skipped, so `required` stops
    #    naming the absent field.
    ("b5r3-eksik-alan-hesabi", "execution",
     "    if issue == contract.SchemaIssue.REQUIRED and not path:",
     "    if False:",
     "test_a_missing_required_field_is_named_from_the_schema"),
    # 4. The runner's allowlist narrowed: the adapter would classify and
    #    the journal would still say nothing.
    ("b5r3-runner-izin-listesi", "runner",
     '        for name, allowed in (("schema_issue", '
     "contract.ALL_SCHEMA_ISSUES)," + NL
     + '                              ("schema_field", '
     "contract.ALL_SCHEMA_FIELDS)):",
     '        for name, allowed in (("schema_issue", '
     "contract.ALL_SCHEMA_ISSUES),):",
     "test_a_schema_violation_reaches_the_journal_with_its_two_closed_words"),
    # 5. THE PRIVACY ONE: the membership check dropped, so a word the
    #    contract does not own -- including model text under that name --
    #    reaches the journal.
    ("b5r3-uyelik-kontrolu", "runner",
     "            if type(value) is str and value in allowed:",
     "            if type(value) is str:",
     "test_a_word_outside_the_contract_never_reaches_the_journal"),
    # ----------------------------------------------------------------
    # B5-R1 -- the SCHEMA's three relations. Measured: one alternation
    # anchored only at the start made the ancestor `tests` match
    # everything beneath it, and a real manifest naming an ordinary test
    # file was refused. The fix has two failure directions, so it gets a
    # mutant for each.
    # ----------------------------------------------------------------
    # 1. Ancestors compared by PREFIX again -- the original defect, and
    #    the one no security test can see, because it only ever refuses
    #    too much.
    ("b5r1-ata-tam-eslesme", "schemas",
     '        parts.append("^(" + "|".join(re.escape(entry) '
     'for entry in ancestors)' + NL + '                     + ")/?$")',
     '        parts.append("^(" + "|".join(re.escape(entry) '
     'for entry in ancestors)' + NL + '                     + ").*$")',
     "test_an_explicitly_named_safe_test_file_is_an_allowed_path"),
    # 2. The frozen family dropped from the pattern: a test file
    #    invented tomorrow would become an allowed path today.
    ("b5r1-sema-test-ailesi", "schemas",
     "    if CONTROL_PLANE_GLOBS:", "    if False:",
     "test_the_control_plane_its_ancestors_and_its_family_stay_refused"),
    # 3. The broad ancestors dropped entirely, which is how
    #    `allowed_paths: ["tests/"]` would come back.
    ("b5r1-genis-ata-izni", "schemas",
     "    if ancestors:", "    if False:",
     "test_the_control_plane_its_ancestors_and_its_family_stay_refused"),
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
    ("dizin-zinciri", "state",
     "    ensure_directory(target.parent)",
     "    target.parent.mkdir(parents=True, exist_ok=True)",
     "brand_new_directory_chain_is_made_durable"),
    # RETARGETED IN B4-R2, not retired: the security intent is "the argv
    # carries the inline canonical schema, never a path", and B4-R2 moved
    # which BINDING carries it (authoritative -> transport). Deleting the
    # label would have dropped the intent along with the old spelling.
    ("r2b-yol-degeri", "cli",
     '        "--json-schema", schemas.IMPLEMENTER_TRANSPORT_BINDING.'
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
    # RETARGETED IN B4-R2 for the same reason: the reported
    # `schema_sha256` must be the real digest of the schema that judged
    # the reply, which is now the AUTHORITY's rather than the argv's.
    ("r2b-yanlis-sha", "execution",
     "        json.loads(authority.canonical_json)), authority.sha256",
     "        json.loads(authority.canonical_json)), "
     "authority.sha256[::-1]",
     "test_the_argv_schema_is_inline_canonical_and_hashed"),
    # THE GUARD B4-R2 CREATED. The transport schema is strictly weaker
    # than the authority -- it has to be, because the API refuses the
    # authority -- so a validator built from it would silently stop
    # enforcing every constraint that cannot travel. This mutant makes
    # the weaker schema the acceptance gate, which is the exact
    # regression the split exists to prevent.
    ("b4r2-tasima-semasi-otorite", "execution",
     "    authority = schemas.IMPLEMENTER_SCHEMA_BINDING",
     "    authority = schemas.IMPLEMENTER_TRANSPORT_BINDING",
     "test_a_reply_is_judged_by_the_authority_not_by_the_argv_schema"),
    # THE FAIL-OPEN THIS PROJECT KEEPS MEETING, in its newest place. The
    # envelope's success flag must be EXACTLY `False`; truthiness accepts
    # `0`, `""` and a missing key alike, and none of those is the CLI
    # saying the run succeeded. A model that failed can still have
    # serialised a well-formed payload on the way out.
    ("b4r3-zarf-hata-bayragi", "execution",
     '    if payload.get("is_error") is not False:',
     '    if payload.get("is_error"):',
     "test_an_error_envelope_is_refused_even_with_a_valid_payload"),
    # B4-R6. The error-envelope classifier refines a failure that has
    # ALREADY been decided by the exit code, so each guard below protects
    # a different way of claiming to know more than the evidence allows.
    #
    # 1. The same truthiness fail-open, on the error side this time:
    #    `is_error: "true"` is a string, and only `True` is the CLI
    #    saying the envelope describes a failure.
    ("b4r6-zarf-hata-kapisi", "execution",
     '    if payload.get("is_error") is not True:',
     '    if not payload.get("is_error"):',
     "test_anything_unproven_stays_the_generic_process_failure"),
    # 2. An unproven subtype must stay GENERIC. A default here would
    #    give every unknown envelope a confident code nobody measured --
    #    the exact failure mode the evidence gate exists to prevent.
    ("b4r6-bilinmeyen-subtype", "execution",
     "    name = ENVELOPE_ERROR_SUBTYPES.get(subtype)",
     '    name = ENVELOPE_ERROR_SUBTYPES.get(subtype, '
     '"ProviderExecutionFailed")',
     "test_anything_unproven_stays_the_generic_process_failure"),
    # 3. The mapping must be the TABLE's, not one class for everything:
    #    four codes that all mean the same thing are one code.
    ("b4r6-sinif-esleme", "execution",
     '    return globals()[name]("model sureci bildirilen bir sinirda durdu",',
     '    return globals()["ProviderExecutionFailed"]('
     '"model sureci bildirilen bir sinirda durdu",',
     "test_each_evidenced_error_subtype_becomes_its_own_class"),
    # 4. The vendor's own spelling is a LOOKUP KEY and never a recorded
    #    value: putting it in the message would carry vendor text into
    #    every report the exception reaches.
    ("b4r6-ham-subtype-sizintisi", "execution",
     '    return globals()[name]("model sureci bildirilen bir sinirda durdu",'
     + NL + "                           exit_code=exit_code, **measurements)",
     '    return globals()[name](f"model sureci {subtype} sinirinda durdu",'
     + NL + "                           exit_code=exit_code, **measurements)",
     "test_the_classification_never_persists_the_envelope"),
    # ----------------------------------------------------------------
    # B4-R8 -- the EVALUATOR's non-zero exit, named from stderr that is
    # still in a bounded buffer. Same four intents as B4-R6, on the road
    # that had no failure code at all.
    # ----------------------------------------------------------------
    # 1. The match is EXACT and whole-line. Substring search is the
    #    failure this gate was written against: four of the six near
    #    misses in the battery contain the marker as a substring.
    ("b4r8-tam-eslesme-kapisi", "audit",
     "        name = STDERR_FAILURE_MARKERS.get(line)",
     "        name = next((v for k, v in STDERR_FAILURE_MARKERS.items()"
     + NL + "                     if k in line), None)",
     "test_a_sentence_that_is_not_exactly_the_marker_is_not_classified"),
    # 2. An unproven stderr must stay GENERIC. A default here hands
    #    every unnamed refusal a confident code nobody measured.
    ("b4r8-bilinmeyen-stderr", "audit",
     "            return globals()[name](" + NL
     + '                "denetci sureci bildirilen bir nedenle durdu",' + NL
     + "                exit_code=exit_code, **measurements)" + NL
     + "    return None",
     "            return globals()[name](" + NL
     + '                "denetci sureci bildirilen bir nedenle durdu",' + NL
     + "                exit_code=exit_code, **measurements)" + NL
     + '    return RepositoryRefused("denetci sureci bildirilen bir nedenle '
     'durdu",' + NL + "                             exit_code=exit_code, "
     "**measurements)",
     "test_a_nonzero_evaluator_exit_with_unknown_stderr_stays_generic"),
    # 3. The two roads share class NAMES and must never share codes: a
    #    name-only lookup files an evaluator failure as an implementer
    #    one, which is a confidently wrong code an operator acts on.
    ("b4r8-yol-karismasi", "contract",
     '    ("audit", "ProcessFailed"): FailureCode.EVALUATOR_PROCESS_FAILED,',
     '    ("audit", "ProcessFailed"): FailureCode.IMPLEMENTER_PROCESS_FAILED,',
     "test_the_two_roads_never_share_a_code"),
    # 4. The matched line is a LOOKUP KEY and never a recorded value:
    #    putting it in the message carries the vendor's own sentence
    #    into every report the exception reaches.
    ("b4r8-ham-stderr-sizintisi", "audit",
     '                "denetci sureci bildirilen bir nedenle durdu",' + NL
     + "                exit_code=exit_code, **measurements)",
     '                f"denetci sureci durdu: {line}",' + NL
     + "                exit_code=exit_code, **measurements)",
     "test_no_stderr_byte_reaches_the_exception_that_names_it"),
    # ----------------------------------------------------------------
    # B4-R11 -- the evaluator's two schemas. One constrains generation
    # and travels; one decides acceptance and never leaves. Each mutant
    # below is a way of collapsing them back into one, which is what a
    # measured provider 400 proved had been happening all along.
    # ----------------------------------------------------------------
    # 1. The acceptance authority on the argv: the defect itself, and
    #    also the road by which the acceptance rules reach a vendor.
    ("b4r11-argv-otorite-semasi", "audit",
     "    schema_file.write_bytes(transport_binding.canonical_bytes)",
     "    schema_file.write_bytes(binding.canonical_bytes)",
     "test_the_file_on_the_argv_carries_the_transport_and_not_the_authority"),
    # 2. Judging the reply with the WEAKER document. Every conditional
    #    rule the transport drops becomes unenforced: `approved` with
    #    findings, `changes_requested` with none.
    # RETARGETED in B4-R14. The false-green battery used to carry this
    # claim, and after the strict transport it cannot: swapping the
    # binding ALSO refuses those bodies, because the elided reply lacks
    # fields the strict copy requires -- so the test passed for a reason
    # that had nothing to do with which document decides. The
    # discriminator is a reply the AUTHORITY accepts and the transport
    # would not, which is what the positive control now asserts.
    ("b4r11-zayif-otorite", "audit",
     "    binding = schemas.SchemaBinding(call.schema)",
     "    binding = schemas.SchemaBinding(call.transport)",
     "test_a_healthy_reply_passes_the_transport_and_the_authority"),
    # 3. A STATIC locked transport describes a document that accepts ids
    #    the runner never minted -- the exact promise `locked_audit_schema`
    #    exists to keep.
    # RETARGETED in B4-R17: the intent is unchanged and the line moved
    # when the locked road began deriving through the evaluator
    # projection.
    ("b4r11-kilitli-statik-tasima", "audit",
     "        return (authoritative," + NL
     + "                schemas.evaluator_transport_schema(authoritative))",
     "        return (authoritative," + NL
     + "                schemas.evaluator_transport_schema(" + NL
     + "                    schemas.LOCKED_AUDIT_RESULT_SCHEMA))",
     "test_a_locked_transport_carries_this_call_s_issued_ids"),
    # 4. A derivation that edits its source weakens the ACCEPTANCE
    #    authority for the rest of the process: the authority is a
    #    module-level dictionary, so one in-place edit is permanent.
    # RETARGETED in B4-R14, not retired: the intent is unchanged and the
    # line carrying it moved when the derivation grew its second pass.
    ("b4r11-turetim-kaynagi-bozuyor", "schemas",
     "    reduced = _transport_node(schema, supported="
     "CODEX_TRANSPORT_KEYWORDS," + NL
     + "                              dropped=CODEX_TRANSPORT_DROPPED)",
     '    schema.pop("allOf", None)' + NL
     + "    reduced = _transport_node(schema, supported="
     "CODEX_TRANSPORT_KEYWORDS," + NL
     + "                              dropped=CODEX_TRANSPORT_DROPPED)",
     "test_the_codex_derivation_is_pure_and_deterministic"),
    # ----------------------------------------------------------------
    # B4-R14 -- the STRICT subset. Two measured provider refusals (root
    # `allOf`, then a property with no `type`) said the transport copy
    # has to satisfy the published contract as a whole, not one rule at
    # a time. Each mutant below removes one of those rules -- and each
    # is deliberately shaped to keep the module IMPORTABLE, because a
    # mutation that explodes at import breaks collection and the
    # harness cannot tell which guard caught it.
    # ----------------------------------------------------------------
    # 1. The inference that gives `const`/`enum` nodes an explicit type.
    #    Substituting a constant instead of inferring it is the same
    #    defect the provider named at ('properties', 'audit_kind').
    ("b4r14-tip-cikarimi", "schemas",
     "    else:" + NL + "        base = _scalar_type(strict)",
     "    else:" + NL + '        base = strict.get("type", "object")',
     "test_the_codex_transport_satisfies_the_strict_subset"),
    # 2. "Every property is required" is the subset's rule; keeping the
    #    authority's own list here is exactly the pre-B4-R14 document.
    ("b4r14-hepsi-zorunlu", "schemas",
     '        strict["required"] = list(strict["properties"])',
     '        strict["required"] = list(strict.get("required", ()))',
     "test_optional_authority_fields_become_required_and_nullable"),
    # 3. The normaliser is where an invalid reply could quietly become a
    #    valid one: widening it to every null would delete a REQUIRED
    #    null and an UNKNOWN null, and the authority would never see
    #    either of the refusals it is there to make.
    ("b4r14-eleme-genisligi", "schemas",
     "            if key in properties and key not in zorunlu "
     "and value is None:",
     "            if value is None:",
     "test_elision_removes_exactly_the_optional_nulls"),
    # ----------------------------------------------------------------
    # B4-R17 -- `next_action` is a CONSEQUENCE of `status`, derived by
    # the adapter because the strict subset cannot express the
    # conditional that ties them. Measured: the first evaluator reply
    # that ever reached the model failed on exactly that pairing.
    # ----------------------------------------------------------------
    # 1. The table IS the authority's own branches. One wrong entry and
    #    the derivation confidently produces a reply the authority
    #    refuses -- the original defect, now with our name on it.
    ("b4r17-turetme-tablosu", "schemas",
     '    "approved": "stop",', '    "approved": "await_repair",',
     "test_the_derived_action_table_is_exactly_what_the_authority_states"),
    # 2. A reply that wrote the field itself is REFUSED, never
    #    overwritten: a silent overwrite hides a model that stopped
    #    following the instruction, including one that got it wrong.
    ("b4r17-sessiz-ustune-yazma", "schemas",
     "        if name in payload:" + NL
     + '            raise ProjectionError("yanit turetilen bir alani '
     'kendisi yazdi")',
     "        if False:" + NL
     + '            raise ProjectionError("yanit turetilen bir alani '
     'kendisi yazdi")',
     "test_a_reply_the_projection_cannot_complete_is_refused"),
    # 3. The derivation must actually RUN, and before the authority
    #    judges: without it every compliant reply is missing a required
    #    field and the road is dead again.
    ("b4r17-projeksiyon-atlandi", "audit",
     "        payload = schemas.project_derived_fields(payload)",
     "        payload = dict(payload)",
     "test_the_b4r15_shape_now_completes_through_the_adapter"),
    # 4. The field must leave the TRANSPORT. Leaving it there asks the
    #    model for a value it cannot be constrained to get right, which
    #    is the whole reason this package exists.
    ("b4r17-tasimada-kaldi", "schemas",
     '        document.get("properties", {}).pop(name, None)',
     '        document.get("properties", {}).get(name, None)',
     "test_the_derived_field_is_absent_from_both_evaluator_transports"),
    # 5. The matrix is the only lever left for rules the transport
    #    cannot carry; dropping it leaves them unstated everywhere.
    ("b4r17-protokol-matrisi", "audit",
     '    return "\\n".join([prompt, *PROTOCOL_MATRIX])',
     "    return prompt",
     "test_the_protocol_matrix_travels_with_every_evaluator_call"),
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
    # ----------------------------------------------------------------
    # B2B-A -- the verified change set. Each of these was reproduced as
    # a self-attack first; two SURVIVED that round and the mechanism
    # was corrected before they became manifest entries.
    # ----------------------------------------------------------------
    # B2B-B2B: `b2ba-izlenmeyen-korlugu` is GONE. It mutated the switch
    # that made git list untracked files, and the main-checkout guard no
    # longer asks git for an inventory at all -- a filesystem walk has
    # no "untracked" concept to switch off, so there is no line whose
    # removal restores the blindness. The intent is pinned as behaviour
    # instead: `test_an_ignored_path_is_not_invisible` in the main-guard
    # battery catches both a new and a changed gitignored file.
    ("b2ba-onek-kacisi", "changes",
     "        if trimmed and (folded == trimmed" + NL
     + '                        or folded.startswith(trimmed + "/")):',
     "        if trimmed and folded.startswith(trimmed):",
     "test_a_prefix_sibling_directory_is_not_covered"),
    ("b2ba-yasakli-onceligi", "changes",
     "    if _covered(path, forbidden) or not _covered(path, allowed):",
     "    if not _covered(path, allowed):",
     "test_forbidden_beats_allowed"),
    ("b2ba-kontrol-glob", "changes",
     "    return any(fnmatch.fnmatchcase(folded, _fold(pattern))" + NL
     + "               for pattern in contract.CONTROL_PLANE_GLOBS)",
     "    return False",
     "test_a_test_file_invented_tomorrow_is_caught_by_the_glob"),
    # B2B-B2A2: both sides of the comparison are canonical spellings
    # now. The direction the mutation removes is the same one.
    ("b2ba-alt-kume", "changes",
     "    if set(canonical) != {_fold(change.path) for change in actual}:",
     "    if not set(canonical) <= {_fold(change.path) for change in actual}:",
     "test_a_change_the_model_omits_is_refused"),
    # B2B-B2B: `b2ba-git-rc` is GONE with the git call it judged. "A
    # failed evidence command is not a clean answer" now lives in the
    # walker seam, and `b2bar1-snapshot-okuma` below mutates exactly
    # that line -- two labels cannot share one target.
    ("b2ba-parmak-izi-icerik", "changes",
     '        b"' + BS + '0".join((change.path.encode("utf-8"), '
     'change.kind.encode("ascii"),' + NL
     + '                    change.mode.encode("ascii"),' + NL
     + '                    change.sha256.encode("ascii"))) + b"' + BS + 'n"',
     '        change.path.encode("utf-8") + b"' + BS + 'n"',
     "test_the_fingerprint_covers_content_not_only_paths"),
    ("b2ba-gorev-sonkontrolu", "changes",
     "        if preflight.manifest_changed(task_file, snapshot) or " + BS + NL
     + "                preflight.snapshot_manifest(task_file).digest != "
     "task_before:",
     "        if False:",
     "test_an_ignored_manifest_inside_the_repository_is_still_re_verified"),
    # B2B-B2B RETARGETED. Same gate, new instrument: the operator's
    # checkout is compared as a filesystem snapshot now instead of a git
    # inventory. The test it answers to is unchanged.
    ("b2ba-ana-agac-sonkontrolu", "changes",
     "        if _main_snapshot(repo_path, main_key, main_policy) != "
     "main_before:",
     "        if False:",
     "test_an_edit_to_the_operator_checkout_is_caught"),
    ("b2ba-finally-yok", "changes",
     "    finally:" + NL
     + "        # EVERY exit: a call that failed may have edited files first,"
     + NL
     + "        # and a safety violation outranks the failure that hid it. The"
     + NL
     + "        # original error is chained, never erased." + NL
     + "        try:" + NL
     + "            actual = verify_after()" + NL
     + "        except ChangeSetError as violation:" + NL
     + "            raise violation from failure",
     "    actual = verify_after()",
     "test_a_forbidden_edit_outranks_the_call_failure_and_chains_it"),
    # B2B-B2A2 RETARGETED. The type gate moved from "the file git named"
    # to "the entry the walker described", and it is still ONE gate: a
    # symlink and a junction both arrive as a link carrying a keyed
    # fingerprint, so a second branch would be invisible to remove.
    ("b2ba-dosya-turu", "changes",
     '        if (entry.kind not in ("file", "dir") or entry.reparse_tag' + NL
     + "                or entry.link_target_mac):",
     "        if False:",
     "test_an_object_this_evidence_model_cannot_represent_is_refused"),
    # B3 RETARGETED BOTH, and the reason is worth keeping: the repair
    # seam that arrived with B3 carries its OWN copy of these two guards,
    # word for word, and it sits EARLIER in the file. `replace(old, new,
    # 1)` therefore stopped mutating the function these targets exercise
    # and started mutating the repair one -- so both entries reported a
    # kill that never happened, and the survivor scan caught them only
    # because it asks the target test directly. A pattern that is not
    # unique to the function it means to judge is not a pattern.
    ("b2ba-kosu-kimligi", "changes",
     '    """The declaration is compared to the evidence, exactly."""' + NL
     + "    reply = outcome.reply" + NL
     + '    if reply.get("run_id") != run_id:',
     '    """The declaration is compared to the evidence, exactly."""' + NL
     + "    reply = outcome.reply" + NL
     + "    if False:",
     "test_a_reply_naming_another_run_is_refused"),
    ("b2ba-yineleme", "changes",
     "    # the set never saw." + NL
     + "    canonical = [_fold(item) for item in declared]" + NL
     + "    if len(set(canonical)) != len(canonical):",
     "    # the set never saw." + NL
     + "    canonical = [_fold(item) for item in declared]" + NL
     + "    if False:",
     "test_a_duplicate_declaration_is_refused"),
    # The repair seam's OWN guard, and the one that is genuinely its own:
    # a repair round's declaration is about THIS call, so it is compared
    # against the delta rather than against the cumulative candidate.
    ("b3-onarim-delta-bildirimi", "changes",
     "    if set(canonical) != {_fold(change.path) for change in delta}:",
     "    if False:",
     "test_a_repair_declaration_is_compared_against_its_own_delta"),
    # ----------------------------------------------------------------
    # B2B-A R1 -- the six audit findings, each as its own mutation
    # ----------------------------------------------------------------
    # B2B-B2B: `b2bar1-indeks-digesti` is GONE, and with it the last of
    # the three index mutations. Both of the gates it and
    # `b2bar1-indeks-bayragi` described read git's per-entry flags, and
    # nothing in this module reads them any more -- the blindness they
    # existed to catch is not reachable, because the filesystem has no
    # opinion about which of its files git is watching. The P0 is pinned
    # as behaviour in `test_a_blinded_index_cannot_hide_the_edit`, which
    # sets the flag BEFORE the call -- the shape the old guard could
    # never see -- and requires the refusal to come from the filesystem
    # guard by its exact sentence.
    ("b2bar1-manifest-icerde", "changes",
     "    if relative is None:" + NL
     + '        raise EvidenceUnavailable("gorev dosyasi depo agacinin '
     'disinda")',
     "    if relative is None:" + NL
     + '        relative = "kurgu-disarida.json"',
     "test_a_manifest_outside_the_repository_is_refused"),
    ("b2bar1-manifest-taban", "changes",
     '    if snapshot.task["baseline_sha"] != baseline_sha:',
     "    if False:",
     "test_a_manifest_naming_another_baseline_is_refused"),
    ("b2bar1-kanonik-kapsam", "changes",
     '    parts = [part for part in str(text).replace("' + BS + BS + '", "/")'
     '.split("/")' + NL
     + '             if part not in ("", ".")]' + NL
     + '    joined = "/".join(parts)' + NL
     + '    return joined.casefold() if os.name == "nt" else joined',
     '    return str(text).replace("' + BS + BS + '", "/").rstrip("/")',
     "test_a_forbidden_entry_cannot_be_escaped_by_spelling"),
    # B2B-B2A2 RETARGETED. Same finding, same intent, new evidence: a
    # deletion carries the mode the file HAD, because recording every
    # deletion the same way erases the one field that tells two
    # otherwise identical deletions apart in the fingerprint.
    ("b2bar1-silme-modu", "changes",
     "                changes.append(_Change(path=path, kind=DELETED," + NL
     + "                                       mode=left.mode, sha256=\"\"))",
     "                changes.append(_Change(path=path, kind=DELETED," + NL
     + "                                       mode=\"000000\", sha256=\"\"))",
     "test_the_classifier_keeps_modes_and_refuses_empty_directory_changes"),
    # B2B-B2B RETARGETED. The finding is the same one: a filesystem
    # object nobody can read is an unanswered question, and the
    # operating system's own message -- which names absolute paths -- is
    # not text this module may repeat. It used to be caught while
    # hashing a file git had listed; it is caught in the walker seam now.
    ("b2bar1-snapshot-okuma", "changes",
     "    except OSError:" + NL
     + '        refusal = EvidenceUnavailable("dosya sistemi kaniti '
     'alinamadi")' + NL
     + "    raise refusal",
     "    except OSError:" + NL
     + "        raise" + NL
     + "    raise refusal",
     "test_a_failed_scan_after_the_model_refuses_the_result"),
    # B2B-B2A2 RETARGETED. "Already dirty" is now "the two trees did not
    # start equal" -- the same refusal, before a model process exists,
    # for the same reason: attribution is impossible otherwise.
    ("b2ba-kirli-agac", "changes",
     "    reference_before, implementer_before = _read_pair(workspace, key)"
     + NL + "    if reference_before != implementer_before:",
     "    reference_before, implementer_before = _read_pair(workspace, key)"
     + NL + "    if False:",
     "test_a_workspace_that_starts_out_of_step_is_refused_before_the_model"),
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
     "test_a_deceptive_identity_never_reaches_the_workspace_binding"),
    # ----------------------------------------------------------------
    # B2B-C1 -- frozen acceptance commands in a disposable mirror. Four
    # mutations, one per mechanism the package exists for: the claim is
    # re-derived, the argv comes from the registry, an incomplete read is
    # not an answer, and a container's verdict is consumed.
    # ----------------------------------------------------------------
    ("b2bc1-taze-dogrulama", "acceptance",
     "    if (verified.changed_files, verified.added, verified.modified,"
     + NL + "            verified.deleted, verified.fingerprint) != (" + NL
     + "            candidate.changed_files, candidate.added, "
     "candidate.modified," + NL
     + "            candidate.deleted, candidate.fingerprint):",
     "    if False:",
     "test_a_stale_verified_change_set_is_refused_against_fresh_evidence"),
    ("b2bc1-kayit-disi-argv", "acceptance",
     "    try:" + NL
     + "        argv = cli.resolve_registry_command(command_id," + NL
     + "                                            "
     "contract.COMMAND_REGISTRY," + NL
     + "                                            paths=paths)" + NL
     + "    except cli.UnsafeInvocation:" + NL
     + '        raise AcceptanceRefused("kabul komutu cozumlenemedi") '
     "from None",
     '    argv = list(contract.COMMAND_REGISTRY[command_id]["argv"]) '
     "+ list(paths)",
     "test_only_the_registry_decides_what_a_path_argument_may_be"),
    # B2B-C1-R1: the same gate, one keyword along. The outcome chain
    # became `elif` when the lifecycle stopped returning from inside the
    # measured block, so the pattern follows it rather than going stale.
    ("b2bc1-okuma-hukmu", "acceptance",
     "        elif not joined or any(stream.outcome != process.READ_COMPLETED"
     + NL + "                               for stream in streams):",
     "        elif False:",
     "test_output_overflow_timeout_and_a_failed_reader_are_all_refusals"),
    # B2B-C1-R1 RETARGETED. The intent never moved -- a cleanup result
    # that is not consumed is a process tree nobody is watching -- but
    # the authority did: the window between `launch_contained` returning
    # and the poll loop had no envelope at all, so the mutation now
    # removes the OUTER lifecycle's verdict instead of the inner drain's.
    ("b2bc1-bosaltma-hukmu", "acceptance",
     "    reclaimed = True if settled else _reclaim(child, container," + NL
     + "                                              started_streams, grace)",
     "    reclaimed = True",
     "test_a_failure_after_launch_still_proves_the_cleanup"),
    # B2B-C2: one mutation per mechanism the application package exists
    # for -- the receipt names the candidate it tested, an ADDED target
    # the operator already has is never overwritten, a failed operation
    # is rolled back, and the checkout's difference is proven to BE the
    # candidate before any success is reported.
    ("b2bc2-rapor-parmak-izi", "application",
     "    if report.candidate_fingerprint != candidate.fingerprint:",
     "    if False:",
     "test_a_candidate_edited_after_acceptance_never_reaches_the_checkout"),
    # The condition ALONE, not the raise with it: a comment sits between
    # the two in the source, and a pattern that spans it goes stale the
    # moment anybody edits the comment -- which reads as NOT-APPLIED and
    # turns a missing guard into a quiet pass.
    ("b2bc2-ekleme-carpismasi", "application",
     "            if current is not None:", "            if False:",
     "test_an_added_target_that_already_exists_is_never_overwritten"),
    ("b2bc2-geri-alma-atlama", "application",
     "        if payload is not None and slots_tree is not None:",
     "        if False and slots_tree is not None:",
     "test_a_failure_at_any_operation_rolls_back_exactly"),
    ("b2bc2-son-fark-kontrolu", "application",
     "    if tuple((item.path, item.kind, item.sha256)" + NL
     + "             for item in observed) != expected:",
     "    if False:",
     "test_a_late_change_anywhere_is_not_a_success"),
    # B2B-C2-R1: the authority itself. Removing this call leaves the
    # report's own field checks standing -- which is exactly the state
    # the package shipped in and which a hand-built report walked
    # straight through, so the mutant has to be caught by the
    # no-receipt test rather than by any of them.
    ("b2bc2-makbuz-otoritesi", "application",
     "    _assert_receipt(acceptance_report, candidate, state_path, binding)",
     "    pass",
     "test_a_report_built_by_hand_without_a_gate_is_refused"),
    # ----------------------------------------------------------------
    # B3 -- the evaluator adapter and the runner. One mutation per
    # mechanism each package exists for.
    #
    # THE ONES THAT ARE NOT HERE ARE A MEASUREMENT, not an oversight.
    # Three further audit-adapter guards -- the last-message watcher, the
    # drain verdict and the prompt-delivery check -- were tried as
    # mutations and SURVIVED, because the tests that name them accept two
    # legal outcomes each: a file ceiling that trips late still refuses
    # through the post-exit size check, and a surviving grandchild is
    # allowed to be either reported or reaped. A mutation those tests
    # cannot kill is not evidence, and weakening this manifest to include
    # it would be worse than recording the gap.
    # ----------------------------------------------------------------
    # The evaluator runs in the candidate, never in the copy the work is
    # measured against: a model with a path to the reference tree can
    # read the answer it is being compared to.
    ("b3-denetci-kok-secimi", "audit",
     "            baseline_sha=call.baseline_sha).implementer_root",
     "            baseline_sha=call.baseline_sha).reference_root",
     "test_the_evaluator_runs_in_the_candidate_implementer_root"),
    # A locked call with no issued ids builds an enum of nothing, which
    # refuses every reply including an honest approval -- a gate that
    # cannot pass is one nobody can tell from a broken one.
    ("b3-kilitli-kimlik-zorunlu", "audit",
     "        if issued_run_id is None or not issued_finding_ids " + BS + NL
     + "                or not issued_mechanism_ids:",
     "        if False:",
     "test_a_locked_audit_without_issued_ids_is_refused_before_launch"),
    ("b3-stdout-tasma-hukmu", "audit",
     "        overflowed = [stream for stream in streams if stream.overflowed]",
     "        overflowed = []",
     "test_an_oversized_stdout_is_still_refused"),
    # The ALLOWLIST, not the shape. Any 32 hex characters satisfy the
    # opaque pattern, including ids the runner never minted -- so the
    # weakened form here is exactly the version that proved "opaque" was
    # a promise about a format rather than a binding to a run.
    ("b3-kilitli-kimlik-baglamasi", "schemas",
     '    finding["finding_id"] = {"enum": sorted(issued_finding_ids)}',
     '    finding["finding_id"] = dict(_OPAQUE_ID)',
     "test_a_locked_audit_is_bound_to_the_ids_the_runner_issued"),
    # A reviewer that wrote is a broken protocol, and the proof is
    # before/after evidence over both roots rather than the sandbox flag
    # on the command line.
    ("b3-denetci-yazma-kaniti", "runner",
     "        if moved:", "        if False:",
     "test_an_evaluator_that_edits_an_allowed_path_blocks_the_application"),
    # THE RULE THE WHOLE LOOP EXISTS FOR.
    ("b3-ikinci-yama-kurali", "runner",
     "        if repeated:", "        if False:",
     "test_the_same_mechanism_twice_stops_before_the_round_budget_does"),
    # The round cap is a SECOND limit, and removing it does not merely
    # allow another repair: `_walk` would carry a rejected final audit
    # straight into the application.
    ("b3-onarim-butcesi", "runner",
     '        if self.rounds["repair"] >= self.max_repair_rounds:',
     "        if False:",
     "test_a_final_audit_asking_for_changes_never_starts_a_second_repair"),
    ("b3-kapsam-disi-bulgu", "runner",
     "            if changes.canonical_path(cited) not in changed:",
     "            if False:",
     "test_a_finding_about_a_file_this_run_did_not_change_blocks"),
    ("b3-kabul-hukmu", "runner",
     "        if not report.passed:", "        if False:",
     "test_a_failing_acceptance_gate_never_reaches_the_audit_or_the_checkout"),
    # Both of these are caught by the WORKSPACE, not by the stop reason:
    # the adapters refuse a spent budget and an out-of-range timeout too,
    # with the same closed codes, so a test that only read the reason
    # would be green against either mutant. What changes is how far the
    # run got before it stopped.
    ("b3-butce-on-kontrolu", "runner",
     "        if remaining <= 0:", "        if False:",
     "test_a_zero_budget_starts_no_model_process"),
    ("b3-saat-on-kontrolu", "runner",
     "        if self._remaining_seconds() <= 0:", "        if False:",
     "test_a_spent_deadline_starts_no_model_process"),
    # The backup is written AFTER a successful move, so it always holds
    # the last state that was really good. The `_open_state` call writes
    # the same line, which is why this pattern carries the line that
    # follows it in `_advance`.
    ("b3-durum-yedegi", "runner",
     "        runner_events.save_state_backup(self.state_dir, payload)" + NL
     + "        self.state = target",
     "        self.state = target",
     "test_resume_recovers_the_same_run_from_the_backup"),
    # `approved` is a claim that the candidate is IN the checkout. Moving
    # the transition in front of the move makes it a claim about an
    # intention.
    ("b3-uygulama-once-onay", "runner",
     "        with self._guard():" + NL
     + "            applied = application.apply_accepted_candidate(" + NL
     + "                **self.identity, verified_changes=verified," + NL
     + "                acceptance_report=report)" + NL
     + "        self.applied_files = applied.applied_files" + NL
     + "        self._advance(contract.State.APPROVED," + NL
     + "                      stop_reason=contract.StopReason.COMPLETED)",
     "        self._advance(contract.State.APPROVED," + NL
     + "                      stop_reason=contract.StopReason.COMPLETED)" + NL
     + "        with self._guard():" + NL
     + "            applied = application.apply_accepted_candidate(" + NL
     + "                **self.identity, verified_changes=verified," + NL
     + "                acceptance_report=report)" + NL
     + "        self.applied_files = applied.applied_files",
     "test_the_state_is_never_approved_without_an_application"),
    # The run's own record is built from a per-kind ALLOWLIST. Copying
    # the reply's fields instead puts model-authored prose into the file
    # that outlives the run.
    # A mechanism is identified by its id COMPARED ACROSS ROUNDS. Minting
    # a fresh allowlist per call leaves the evaluator unable to name the
    # mechanism it named last time, so the second-patch rule becomes a
    # rule nothing can trip -- and the run reports the round budget
    # instead, which tells the operator to buy another round.
    ("b3-kilitli-kimlik-omru", "runner",
     "        if self.issued_ids is None:", "        if True:",
     "test_a_locked_finding_crosses_as_a_class_and_a_count_and_nothing_else"),
    ("b3-bulgu-izin-listesi", "runner_events",
     "    allowed = RECORD_FIELDS.get(audit_kind)",
     "    allowed = tuple(finding) if isinstance(finding, dict) else ()",
     "test_no_model_authored_prose_is_written_into_the_state_directory"),
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
    argv = [sys.executable, "-m", "pytest", *BATTERY,
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
             "from tools.agent_loop import (state, locking, "
             "preflight, execution, cli, schemas, changes, acceptance, "
             "application, audit, runner, runner_events)"],
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
