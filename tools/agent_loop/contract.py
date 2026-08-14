"""FROZEN CONTRACT for the Claude <-> Codex agent loop. PHASE A.

CONTRACT ONLY -- no behaviour lives in this module. It fixes the states,
the transitions, the stop reasons, the CLI argument rules and the
privacy boundary, so the red tests can name what they require and the
runner (Phase B) has one target it cannot quietly miss.

WHY IT EXISTS. The last workstream cost roughly twenty audit rounds
because the same mechanism was patched four and five times in a row,
each patch answering the previous counterexample and creating the next.
The loop exists to make that shape IMPOSSIBLE rather than merely
discouraged: one mechanism per package, one implementation round, one
repair round, and a hard stop the moment a mechanism breaks twice.

THIS IS NOT AUTONOMOUS DEVELOPMENT. It is a budgeted implement-audit
protocol whose every outward-facing action -- commit, push, spend,
scope change -- stops and asks a human.


ROLES, AND WHY THEY MAY NOT MERGE
---------------------------------

    IMPLEMENTER (claude)  edits only inside `allowed_paths`; may not
                          change acceptance criteria; applies exactly
                          ONE repair group from the evaluator's
                          findings.
    EVALUATOR   (codex)   READ-ONLY. Produces findings backed by an
                          independent probe or a file:line citation.
                          May not design, may not widen scope, may not
                          edit. It reports violations of the acceptance
                          contract that is ALREADY frozen.

Roles swap only on an explicit human instruction. An evaluator that
edits is not a stricter evaluator, it is an implementer with no
reviewer -- which is the exact failure this separation buys off.

STRICTLY SEQUENTIAL. The two never run against the same working tree at
the same time. Concurrent editing is not a v1 feature and is not a
configuration option; if it is ever wanted it needs separate git
worktrees and its own package.


THE STATE MACHINE
-----------------

    PREFLIGHT ──fail──> BLOCKED (no model was called)
        │
        ▼
    IMPLEMENTING ──> ACCEPTANCE ──> AUDITING
                                       │
                        approved ──────┴────── changes_requested
                            │                        │
                            ▼                        ▼
                        APPROVED                   REPAIRING
                                                     │
                                                     ▼
                                              ACCEPTANCE_2
                                                     │
                                                     ▼
                                              FINAL_AUDITING
                                                     │
                              approved ──────────────┴────── changes_requested
                                  │                                │
                                  ▼                                ▼
                              APPROVED                          BLOCKED
                                                    (a THIRD patch is never attempted)

Every terminal state is one of APPROVED, BLOCKED, FAILED. There is no
"finished with warnings": an outcome that cannot be classified is
BLOCKED, because an unclassified success is how a broken run gets
reported as a good one.


THE SECOND-PATCH RULE, WHICH IS THE POINT OF THE WHOLE LOOP
-----------------------------------------------------------

If the SAME mechanism is reported broken in two separate evaluator
rounds, no third heuristic patch is attempted. The run stops with
`repeated_mechanism_failure` and hands the human: the opaque mechanism
id, how many rounds it survived, the current diff, the test counts, and
the statement that this needs a DESIGN change rather than another
patch.

A mechanism is identified by `finding.mechanism_id` -- assigned by the
evaluator, compared across rounds. Two findings with different wording
about the same mechanism id are the same mechanism.


WHAT THE STATE DIRECTORY MAY HOLD
---------------------------------

`.agent-loop/` (git-ignored, verified by preflight) holds an opaque run
id, the state machine, counters, exit codes, test counts, durations,
budget figures, normalised repo-relative paths, and structured
findings.

It may NEVER hold: document text; questions, answers, keys, passages or
holdout content; environment variable VALUES; tokens, DSNs, passwords
or API keys; unbounded raw model stdout/stderr; or user document
filenames. The loop writes about the work, never the material.


MODEL OUTPUT IS UNTRUSTED INPUT
-------------------------------

Both models' output is parsed as hostile: validated against a JSON
Schema with `additionalProperties: false`, size-capped, stripped of
terminal control characters, and every path field normalised and proven
to sit inside the repo. A response that does not validate is a FAILURE,
never something to repair by guessing at intent -- "it probably meant
this" is how a schema stops being a boundary.
"""
from __future__ import annotations

PROTOCOL_VERSION = "1.0"

# The directory is a constant, not a setting: a configurable state root
# is one typo away from writing run state into the document tree.
STATE_DIR_NAME = ".agent-loop"


class State:
    """Every state the loop can be in. Anything else is a bug."""

    PREFLIGHT = "preflight"
    IMPLEMENTING = "implementing"
    ACCEPTANCE = "acceptance"
    AUDITING = "auditing"
    REPAIRING = "repairing"
    ACCEPTANCE_2 = "acceptance_2"
    FINAL_AUDITING = "final_auditing"
    APPROVED = "approved"
    BLOCKED = "blocked"
    FAILED = "failed"


TERMINAL_STATES = (State.APPROVED, State.BLOCKED, State.FAILED)

# The ONLY legal transitions. A runner that moves anywhere else is
# wrong even if the destination looks reasonable.
ALLOWED_TRANSITIONS = {
    State.PREFLIGHT: (State.IMPLEMENTING, State.BLOCKED, State.FAILED),
    State.IMPLEMENTING: (State.ACCEPTANCE, State.BLOCKED, State.FAILED),
    State.ACCEPTANCE: (State.AUDITING, State.BLOCKED, State.FAILED),
    State.AUDITING: (State.APPROVED, State.REPAIRING, State.BLOCKED,
                     State.FAILED),
    State.REPAIRING: (State.ACCEPTANCE_2, State.BLOCKED, State.FAILED),
    State.ACCEPTANCE_2: (State.FINAL_AUDITING, State.BLOCKED, State.FAILED),
    # NO path back to REPAIRING: that edge IS the third patch.
    State.FINAL_AUDITING: (State.APPROVED, State.BLOCKED, State.FAILED),
    State.APPROVED: (),
    State.BLOCKED: (),
    State.FAILED: (),
}


class Role:
    IMPLEMENTER = "implementer"
    EVALUATOR = "evaluator"


class Status:
    """What a model says about its own turn."""

    IMPLEMENTED = "implemented"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    BLOCKED = "blocked"
    FAILED = "failed"


ALL_STATUSES = (Status.IMPLEMENTED, Status.APPROVED, Status.CHANGES_REQUESTED,
                Status.BLOCKED, Status.FAILED)

# Which statuses each role may return. An implementer that returns
# `approved` is grading its own work; an evaluator that returns
# `implemented` has edited something.
ROLE_STATUSES = {
    Role.IMPLEMENTER: (Status.IMPLEMENTED, Status.BLOCKED, Status.FAILED),
    Role.EVALUATOR: (Status.APPROVED, Status.CHANGES_REQUESTED,
                     Status.BLOCKED, Status.FAILED),
}


class StopReason:
    """Why the loop stopped. Every terminal state carries exactly one."""

    COMPLETED = "completed"
    PREFLIGHT_FAILED = "preflight_failed"
    MODEL_PROCESS_FAILED = "model_process_failed"
    DIRTY_WORKTREE = "dirty_worktree"
    STAGED_CHANGES = "staged_changes"
    BASELINE_MISMATCH = "baseline_mismatch"
    PATH_NOT_ALLOWED = "path_not_allowed"
    EVALUATOR_MODIFIED_WORKSPACE = "evaluator_modified_workspace"
    SCHEMA_VIOLATION = "schema_violation"
    REPEATED_MECHANISM_FAILURE = "repeated_mechanism_failure"
    REPAIR_ROUNDS_EXHAUSTED = "repair_rounds_exhausted"
    OUT_OF_SCOPE_FINDING = "out_of_scope_finding"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WALL_CLOCK_EXCEEDED = "wall_clock_exceeded"
    TIMEOUT = "timeout"
    USER_APPROVAL_REQUIRED = "user_approval_required"
    LEAK_SCAN_HARD_FINDING = "leak_scan_hard_finding"
    ACCEPTANCE_FAILED = "acceptance_failed"
    LOCK_HELD = "lock_held"
    INTERRUPTED = "interrupted"
    CONTROL_PLANE_MODIFIED = "control_plane_modified"


ALL_STOP_REASONS = tuple(
    value for name, value in vars(StopReason).items()
    if not name.startswith("_") and isinstance(value, str))


# ---------------------------------------------------------------------
# THE CONTROL PLANE -- what a RUNNING loop may never touch
# ---------------------------------------------------------------------

# A task could set `allowed_paths: ["tools/agent_loop/"]` and the
# implementer would then be free to rewrite the command registry, the
# forbidden-flag list, the schemas, the state machine and the tests that
# judge it -- and then "pass" against rules it had just written. Naming
# `contract_change` as a human gate does not prevent that; only refusing
# the edit does.
#
# These prefixes are immutable for a RUN. Humans develop this code
# normally, through the ordinary review process; what is forbidden is
# the loop editing the loop while the loop is running.
CONTROL_PLANE_PATHS = (
    "tools/agent_loop/",
    "tests/test_agent_loop_contract.py",
    # the VERIFICATION tools the registry runs. A task that can edit the
    # leak scanner can silence it; one that can edit the gate script can
    # make the gate pass. Being downstream of the loop does not make
    # them less part of what judges it.
    "scripts/p0_gate.sh",
    "eval/tools/leak_scan.py",
)

# The loop's safety battery grows by phase. Listing today's test files one by
# one made the next phase's tests editable by the loop until somebody
# remembered to extend this tuple. Patterns are therefore part of the frozen
# control-plane contract too; implementations must match candidate paths
# against these globs in addition to the exact prefixes above.
CONTROL_PLANE_GLOBS = (
    "tests/test_agent_loop*.py",
)

# ANCESTORS COUNT. `allowed_paths: ["tools/"]` was accepted because only
# the exact protected prefix was refused -- and `tools/` contains
# `tools/agent_loop/`. So does `tests/`, `eval/` and `scripts/`. The
# blocked set is every protected prefix AND every directory above one:
# permission over a parent is permission over the child.
def _with_ancestors(prefixes):
    blocked = set()
    for prefix in prefixes:
        trimmed = prefix.rstrip("/")
        blocked.update({trimmed, trimmed + "/"})
        parts = trimmed.split("/")
        for depth in range(1, len(parts)):
            ancestor = "/".join(parts[:depth])
            blocked.update({ancestor, ancestor + "/"})
    blocked.update({".", "./", ""})
    return frozenset(blocked)


CONTROL_PLANE_BLOCKED_PATHS = _with_ancestors(CONTROL_PLANE_PATHS)

# The task manifest protects ITSELF: its path is chosen by the caller,
# so it cannot appear in a static list. The runner hashes the file it
# was given, before and after, and a change is terminal.
TASK_MANIFEST_IS_IMMUTABLE = True

# PHASE B DESIGN, frozen so the runner is built against it rather than
# around it: the implementer runs in a FLAT WORKSPACE, never in the
# operator's working tree.
#
# The baseline commit is materialised from RAW GIT OBJECTS into two
# independent plain trees that contain no `.git` at all. The model works
# in the IMPLEMENTER tree; the REFERENCE tree is the unchanging thing its
# work is measured against, and the model has no path to it. Hashing
# after the fact detects damage that has already happened and been left
# behind; a separate workspace means the main tree is never the thing at
# risk, and only a verified, authorised change may later be applied back.
#
# THE DISPOSABLE GIT WORKTREE THIS REPLACED WAS NOT AN AUTHORITY. It
# carried a `.git` link straight into the model's reach, and git lost the
# evidence job twice to state reachable through it: the per-repository
# index, where `skip-worktree` hid a modified control-plane file, and
# shared configuration, where a clean filter hid a change AND ran a
# model-supplied command during verification.
IMPLEMENTER_RUNS_IN_FLAT_WORKSPACE = True

# Hashed before and after every implementer call. Any difference is
# terminal -- not a finding to repair, because the thing that would
# judge the repair is the thing that changed.
CONTROL_PLANE_VIOLATION_IS_TERMINAL = True


# ---------------------------------------------------------------------
# CLOSED VOCABULARIES -- where free text used to be
# ---------------------------------------------------------------------

class SummaryCode:
    """A locked-set audit's whole verdict, as a code.

    The finding objects were made textless and the ENVELOPE was not: a
    2000-character `summary` sat right next to them, required, and a
    passage fits in it comfortably. A closed vocabulary has no room."""

    CRITERIA_MET = "criteria_met"
    CRITERIA_UNMET = "criteria_unmet"
    REGRESSION_DETECTED = "regression_detected"
    NO_REGRESSION = "no_regression"
    INSUFFICIENT_SIGNAL = "insufficient_signal"


ALL_SUMMARY_CODES = (SummaryCode.CRITERIA_MET, SummaryCode.CRITERIA_UNMET,
                     SummaryCode.REGRESSION_DETECTED,
                     SummaryCode.NO_REGRESSION,
                     SummaryCode.INSUFFICIENT_SIGNAL)


class EventCode:
    """Everything an event may say. `detail` was a free string, which is
    a document-sized hole in a file that gets written on every step."""

    PREFLIGHT_OK = "preflight_ok"
    PREFLIGHT_FAILED = "preflight_failed"
    STATE_TRANSITION = "state_transition"
    MODEL_CALL_STARTED = "model_call_started"
    MODEL_CALL_FINISHED = "model_call_finished"
    OUTPUT_TRUNCATED = "output_truncated"
    SCHEMA_VIOLATION = "schema_violation"
    ACCEPTANCE_STARTED = "acceptance_started"
    ACCEPTANCE_FINISHED = "acceptance_finished"
    BUDGET_CHECK = "budget_check"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_RELEASED = "lock_released"
    CONTROL_PLANE_MODIFIED = "control_plane_modified"
    EVALUATOR_MODIFIED_WORKSPACE = "evaluator_modified_workspace"
    GATE_REQUIRED = "gate_required"
    INTERRUPTED = "interrupted"
    RUN_FINISHED = "run_finished"


ALL_EVENT_CODES = tuple(
    value for name, value in vars(EventCode).items()
    if not name.startswith("_") and isinstance(value, str))


class FailureCode:
    """WHICH mechanism failed, as a closed code (B4-R4).

    THE GAP THIS CLOSES, measured on two real runs. One stopped with
    `schema_violation` and another with `model_process_failed`, and
    neither code says what actually broke: `schema_violation` is raised
    both by the adapter refusing a reply and by the change-set gate
    refusing a DECLARATION that did not match the filesystem, while
    `model_process_failed` covers a non-zero exit, a container that could
    not be built and a process tree that outlived its call. Three
    different repairs hide behind each of those names.

    A stop reason answers "may the run continue"; a failure code answers
    "which mechanism broke". They are different questions and the journal
    now records both."""

    IMPLEMENTER_PROCESS_FAILED = "implementer_process_failed"
    IMPLEMENTER_OUTPUT_LIMIT = "implementer_output_limit"
    IMPLEMENTER_TIMEOUT = "implementer_timeout"
    IMPLEMENTER_SCHEMA_VIOLATION = "implementer_schema_violation"
    IMPLEMENTER_PROMPT_NOT_DELIVERED = "implementer_prompt_not_delivered"
    IMPLEMENTER_CONTAINMENT_FAILED = "implementer_containment_failed"
    IMPLEMENTER_PROCESS_TREE_SURVIVED = "implementer_process_tree_survived"
    CHANGE_DECLARATION_MISMATCH = "change_declaration_mismatch"
    CHANGE_EVIDENCE_UNAVAILABLE = "change_evidence_unavailable"
    CHANGE_UNSAFE = "change_unsafe"


ALL_FAILURE_CODES = tuple(
    value for name, value in vars(FailureCode).items()
    if not name.startswith("_") and isinstance(value, str))

# The table the runner reads, keyed by (MODULE LEAF, CLASS NAME).
#
# WHY BOTH PARTS. The runner may not import `execution` -- that absence
# is an invariant its own battery pins by walking the runner's AST, and
# it is what stops the loop from ever editing without the change-set
# gates. So the failure cannot be matched by `isinstance`, and it cannot
# be matched by class name alone either: `audit` defines `ProcessFailed`,
# `SchemaViolation`, `ContainmentFailed`, `OutputLimitExceeded`,
# `Timeout`, `ProcessTreeSurvived` and `WorkspaceNotBound` under exactly
# the names `execution` uses. A name-only table would file an EVALUATOR
# process failure as `implementer_process_failed`, and a confidently
# wrong closed code is worse than no code at all.
#
# WHAT IS DELIBERATELY ABSENT: the evaluator family. This package is
# scoped to the implementer road, so an `audit` failure still yields no
# code rather than a guessed one -- the gap is stated, not papered over.
FAILURE_CODES = {
    ("execution", "ProcessFailed"): FailureCode.IMPLEMENTER_PROCESS_FAILED,
    ("execution", "OutputLimitExceeded"): FailureCode.IMPLEMENTER_OUTPUT_LIMIT,
    ("execution", "Timeout"): FailureCode.IMPLEMENTER_TIMEOUT,
    ("execution", "SchemaViolation"):
        FailureCode.IMPLEMENTER_SCHEMA_VIOLATION,
    ("execution", "PromptNotDelivered"):
        FailureCode.IMPLEMENTER_PROMPT_NOT_DELIVERED,
    ("execution", "ContainmentFailed"):
        FailureCode.IMPLEMENTER_CONTAINMENT_FAILED,
    ("execution", "ProcessTreeSurvived"):
        FailureCode.IMPLEMENTER_PROCESS_TREE_SURVIVED,
    ("changes", "DeclarationMismatch"):
        FailureCode.CHANGE_DECLARATION_MISMATCH,
    ("changes", "EvidenceUnavailable"):
        FailureCode.CHANGE_EVIDENCE_UNAVAILABLE,
    ("changes", "UnsafeChange"): FailureCode.CHANGE_UNSAFE,
}

ALL_ROLES = (Role.IMPLEMENTER, Role.EVALUATOR)

# Ids that come back from a LOCKED audit are not model-chosen slugs --
# a slug is free text with a short leash, and "gizli-belge-adi" is a
# perfectly valid one. The runner mints these before the call and
# validates every returned id against the allowlist it handed out; an
# id it did not issue is a schema violation, not a new case.
OPAQUE_ID_PATTERN = r"^[0-9a-f]{32}$"
OPAQUE_ID_BITS = 128

# The pattern alone only proves SHAPE: any 32 hex characters satisfy it,
# including ones the runner never issued. The binding is an ALLOWLIST --
# the runner mints the ids, builds a per-call schema pinning `run_id` to
# a const and the finding ids to an enum of exactly what it issued, and
# an id outside that set is a schema violation rather than a new case.
LOCKED_IDS_ARE_ALLOWLISTED = True

# A runner invariant rather than a schema one, because it relates two
# fields: money already spent can never exceed the ceiling. Checked
# after every model call and asserted before every new one.
BUDGET_INVARIANT = "spent_usd <= max_usd"


# ---------------------------------------------------------------------
# CLI INTEGRATION -- measured from the installed binaries, not recalled.
#   claude 2.1.220     `claude --help`
#   codex  (exec)      `codex exec --help`
# A flag that is not in these tuples was not verified to exist and must
# not be passed.
# ---------------------------------------------------------------------

CLAUDE_REQUIRED_FLAGS = (
    "--print",              # non-interactive
    "--output-format",      # json
    "--json-schema",        # the response shape is enforced by the CLI too
    "--max-budget-usd",     # hard cost ceiling, --print only
    "--allowedTools",       # explicit tool allowlist
    "--permission-mode",    # never bypassPermissions (see FORBIDDEN)
)

CODEX_REQUIRED_FLAGS = (
    "exec",                 # the non-interactive subcommand
    "--sandbox",            # must be read-only
    "--cd",                 # the repo root, stated explicitly
    "--output-schema",      # JSON Schema file for the final response
    "--output-last-message",  # final message to a file, not scraped stdout
    "-c",                   # carries the approval policy, see below
)

CODEX_SANDBOX_READ_ONLY = "read-only"

# An unattended loop must never sit waiting for a human keypress. The
# top-level `codex` command takes `-a/--ask-for-approval`, but `codex
# exec` REJECTS it -- verified against the installed binary:
#     codex exec -a never  ->  error: unexpected argument '-a' found
# The route that `exec` does accept is the config override, so that is
# what the contract mandates. Written as the exact token pair the argv
# builder must emit, so a test can assert it rather than trust it.
CODEX_APPROVAL_OVERRIDE = ("-c", "approval_policy=never")

# Refused structurally, in BOTH directions: a task file may not request
# them and a built argv may not contain them. These are the flags that
# turn a sandboxed reviewer into an unsupervised writer.
FORBIDDEN_FLAGS = (
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
    "--dangerously-bypass-approvals-and-sandbox",
    "--dangerously-bypass-hook-trust",
)

FORBIDDEN_FLAG_VALUES = (
    ("--permission-mode", "bypassPermissions"),
    ("--sandbox", "workspace-write"),
    ("--sandbox", "danger-full-access"),
    ("-s", "workspace-write"),
    ("-s", "danger-full-access"),
)


# ---------------------------------------------------------------------
# THE COMMAND REGISTRY -- a task may NAME a command, never spell one
# ---------------------------------------------------------------------

# An argv LIST removes shell injection; it does not make a dangerous
# command safe. `["git", "push"]` is a perfectly well-formed argv list.
# So a task file carries a command_id and nothing else, and the argv it
# resolves to is frozen HERE, in code a reviewer reads, rather than in a
# JSON file a model or a hurried edit can rewrite.
#
# THE SHELL RULE, STATED PRECISELY, because the loose version reads as a
# contradiction: `p0_gate` runs `bash scripts/p0_gate.sh`. What is
# forbidden is an ARBITRARY or INLINE shell -- `shell=True`, `bash -c`,
# `powershell -Command`, or any string a model could influence. What is
# allowed is invoking a FIXED, TRACKED, REVIEWED script file named in
# this registry. The interpreter is not the hazard; an unreviewed
# command string is.
COMMAND_REGISTRY = {
    "pytest_full": {
        "argv": ("python", "-m", "pytest", "-q"),
        "accepts_paths": False,
    },
    "pytest_selected": {
        "argv": ("python", "-m", "pytest", "-o", "addopts=", "-q"),
        # repo-relative test paths, schema-validated; no other argument
        "accepts_paths": True,
    },
    "p0_gate": {
        "argv": ("bash", "scripts/p0_gate.sh"),
        "accepts_paths": False,
    },
    "leak_scan": {
        "argv": ("python", "-m", "eval.tools.leak_scan"),
        "accepts_paths": False,
    },
}

# Defence in depth over the registry itself: no entry may invoke a
# version-control, shell, package-manager or network tool. Tested, so a
# future "just add one small command" cannot slip past review.
REGISTRY_FORBIDDEN_PROGRAMS = (
    "git", "gh", "hg", "svn",
    "sh", "bash -c", "cmd", "cmd.exe", "powershell", "pwsh",
    "pip", "pip3", "npm", "npx", "yarn", "poetry", "conda",
    "curl", "wget", "ssh", "scp", "rsync", "docker", "kubectl",
    "rm", "del", "format", "mkfs",
)


# ---------------------------------------------------------------------
# WHAT THE IMPLEMENTER MAY DO
# ---------------------------------------------------------------------

# NOT configuration. `allowed_tools=["Bash"]` was accepted, and a
# Claude with Bash can `git add`, `git commit`, `git push`, install a
# dependency or reach the network -- every one of them a human gate the
# runner would never even see, because the gate lives in the runner and
# the action happened inside the model's own tool call.
#
# The implementer reads and edits files. It does not run things: the
# runner runs the registry commands, and that separation is the only
# reason the gates mean anything.
IMPLEMENTER_ALLOWED_TOOLS = ("Read", "Glob", "Grep", "Edit", "Write")

IMPLEMENTER_FORBIDDEN_TOOLS = (
    "Bash", "BashOutput", "KillShell", "Agent", "Task",
    "WebFetch", "WebSearch", "NotebookEdit", "SlashCommand",
)


# ---------------------------------------------------------------------
# HUMAN GATES
# ---------------------------------------------------------------------

# The loop stops and asks rather than doing any of these. They are not
# "risky steps to be careful with" -- they are outside what an
# unattended loop is allowed to decide.
USER_APPROVAL_REQUIRED = (
    "git_add",
    "git_commit",
    "git_push",
    "branch_delete",
    "history_rewrite",
    "contract_change",
    "golden_set_change",
    "locked_holdout_change",
    "gpu_or_pod_rental",
    "paid_service_start",
    "dependency_install",
    "network_write",
    "production_deploy",
    "migration_apply",
    "data_or_index_change",
    "scope_widening",
    "destructive_operation",
)


# ---------------------------------------------------------------------
# PRIVACY BOUNDARY -- evaluator to implementer
# ---------------------------------------------------------------------

# TWO AUDIT KINDS, because one schema could not keep this promise.
#
# A single evaluator schema carried free-text `claim` and
# `required_action`. A runner cannot look at free text and decide
# whether it is a code observation or a sentence lifted out of a locked
# document -- and a redaction heuristic that tries would be exactly the
# kind of guess this project keeps getting burned by. So the boundary
# moves into the TYPE:
#
#   CODE_AUDIT    findings about tracked source. May carry file, line
#                 and a short description, because the implementer can
#                 already read that file.
#   LOCKED_AUDIT  findings from the locked holdout. Carries an opaque
#                 id, an error CLASS and a COUNT. No free text at all,
#                 so there is no field a passage could travel in.
class AuditKind:
    CODE = "code_audit"
    LOCKED = "locked_audit"


ALL_AUDIT_KINDS = (AuditKind.CODE, AuditKind.LOCKED)


class LockedFindingClass:
    """The only vocabulary a locked-set finding may use."""

    WRONG_ROW = "wrong_row"
    UNGROUNDED = "ungrounded"
    ABSTAINED_WHEN_ANSWERABLE = "abstained_when_answerable"
    ANSWERED_WHEN_UNANSWERABLE = "answered_when_unanswerable"
    FORMAT_VIOLATION = "format_violation"
    REGRESSION_VS_BASELINE = "regression_vs_baseline"


ALL_LOCKED_FINDING_CLASSES = (
    LockedFindingClass.WRONG_ROW,
    LockedFindingClass.UNGROUNDED,
    LockedFindingClass.ABSTAINED_WHEN_ANSWERABLE,
    LockedFindingClass.ANSWERED_WHEN_UNANSWERABLE,
    LockedFindingClass.FORMAT_VIOLATION,
    LockedFindingClass.REGRESSION_VS_BASELINE,
)

# What a CODE finding may carry to the implementer.
FINDING_TRANSFERABLE_FIELDS = (
    "finding_id",
    "mechanism_id",
    "severity",
    "file",          # tracked source only
    "line",
    "claim",
    "reproduction_result",
    "required_action",
)

# What a LOCKED finding may carry. Note the absence of every text field.
LOCKED_FINDING_TRANSFERABLE_FIELDS = (
    "finding_id",
    "mechanism_id",
    "severity",
    "error_class",
    "case_count",
)

# Never crossed, and never written to the state directory either.
NEVER_TRANSFERABLE = (
    "question_text",
    "answer_text",
    "answer_key",
    "passage",
    "source_page",
    "document_name",
    "holdout_case_content",
    "characterisation_note",
)


# ---------------------------------------------------------------------
# DEFAULTS -- deliberately small
# ---------------------------------------------------------------------

DEFAULTS = {
    "max_implementation_rounds": 1,
    "max_repair_rounds": 1,
    "max_evaluator_rounds": 2,
    "max_wall_clock_minutes": 120,
    "max_output_bytes": 256 * 1024,
    "model_call_timeout_seconds": 1800,
    "max_budget_usd": 5.0,
}

# Identifiers coming back from a model are constrained to a shape that
# cannot carry a path, a shell fragment or a document string.
IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9_-]{2,63}$"
