"""The bounded agent loop. PACKAGE B3.

THE ONE PLACE THE PHASES ARE JOINED, and deliberately the only thing in
this file: preflight, the flat workspace, one implementer call, the
acceptance gate, the read-only audit, at most one repair, and -- only
after all of them -- the application of the candidate into the
operator's checkout. Every one of those mechanisms lives in its own
module and is proven there; nothing here re-implements any of them,
because a second implementation of a gate is a second opinion about
what the gate means.

WHAT THIS RUNNER MAY NOT DO, stated as code rather than as a promise:

  * It cannot reach `execution.run_implementer`. That module is not
    imported here at all. Every model edit in the loop passes through
    `changes.run_verified_implementation` or
    `changes.run_verified_repair`, which is where the workspace binding,
    the manifest binding, the main-checkout guard and the path
    authorisation live. A runner that could call the adapter directly
    would be a runner that could edit with none of them.
  * It cannot build an `AcceptanceReport`. `acceptance.AcceptanceReport`
    is a public dataclass with a public constructor, so holding one
    proves only that somebody could type its field names -- B2B-C2-R1
    measured an exact-typed forgery reaching the operator's checkout
    with zero acceptance commands run. The only authority is the receipt
    `run_acceptance` persists, and `apply_accepted_candidate` is what
    reads it.
  * It cannot find a binary. There is no discovery in this package;
    `binaries` is a mandatory keyword argument and a caller who forgets
    it gets a `TypeError`, not an invoice.
  * It cannot stage, commit, push, install, deploy, migrate or widen its
    own scope. Those are human gates: the loop stops, names the closed
    gate code in `pending_approval`, and asks.

ONE LOCK, ONE DEADLINE, ONE BUDGET LEDGER. The lock is the outermost
boundary, taken before a workspace, a mirror, a model process, an
acceptance command or an application exists. The deadline is absolute
and set once: no phase gets a fresh clock, and every call is given
`min(its own limit, what is left)`. A call that cannot be given the
contract's minimum does not start.

A DECLARED LIMIT, because an undeclared one is a lie by omission: the
CLIs report no cost, so `spent_usd` cannot be measured and this runner
does not invent a figure for it. What IS enforced is the ceiling on
every call -- `--max-budget-usd` carries the remaining budget on each
implementer invocation, and a budget that is already exhausted refuses
the call before a process exists. The whole-run total is bounded by the
per-call ceiling times the round caps (one implementation, one repair),
not by an observation this runner cannot make.

THE STATE MACHINE IS THE FROZEN ONE. Every move goes through
`state.advance`, so the contract's transition table decides, not this
file's control flow; `visited` is the real sequence and an approval that
skipped a phase cannot spell one.
"""
from __future__ import annotations

import contextlib
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path

from tools.agent_loop import (acceptance, application, audit, changes,
                              contract, flat_workspace, fs_evidence, locking,
                              runner_events, schemas)
from tools.agent_loop import preflight as preflight_gates
from tools.agent_loop import state as state_module

# NOT configuration, and not the task's to raise. The frozen contract
# caps these and the task schema caps them again; the effective limit is
# the smaller of the two, so a manifest can only ever ask for LESS.
MAX_IMPLEMENTATION_ROUNDS = contract.DEFAULTS["max_implementation_rounds"]
MAX_REPAIR_ROUNDS = contract.DEFAULTS["max_repair_rounds"]
MAX_EVALUATOR_ROUNDS = contract.DEFAULTS["max_evaluator_rounds"]

_MODEL_TIMEOUT = schemas.TASK_SCHEMA["properties"][
    "model_call_timeout_seconds"]
MIN_MODEL_SECONDS = _MODEL_TIMEOUT["minimum"]
MAX_MODEL_SECONDS = _MODEL_TIMEOUT["maximum"]

# The environment the loop hands its children. Two closed values and
# nothing else: WHICH run this is, and WHICH round of that run the call
# belongs to. Both are already inside the reply the model must produce,
# so neither is new information reaching it -- what they buy is a child
# that can answer as itself rather than as a fixed recording, which is
# what the test fakes need and what a real CLI wrapper may use for its
# own logging. No path, no secret and no operator value is exported.
ENV_RUN_ID = "AGENT_LOOP_RUN_ID"
ENV_ROUND = "AGENT_LOOP_ROUND"
# LOCKED audits only: the ids the runner minted for THIS call, which the
# per-call bound schema pins with `const` and `enum`. Exporting them is
# what makes an honest locked reply possible at all -- an id the runner
# did not issue is a schema violation, so the evaluator has to be told
# which ones it did issue.
ENV_LOCKED_RUN_ID = "AGENT_LOOP_LOCKED_RUN_ID"
ENV_FINDING_IDS = "AGENT_LOOP_FINDING_IDS"
ENV_MECHANISM_IDS = "AGENT_LOOP_MECHANISM_IDS"

# How many opaque ids a locked call hands out. The evaluator picks from
# these; it cannot mint its own, because a model-chosen 32-hex slug
# satisfies the pattern while naming nothing the runner issued.
LOCKED_ID_COUNT = 8

LockHeld = locking.LockHeld


class RunnerError(RuntimeError):
    """A refusal this module makes itself.

    Carries a fixed sentence chosen here and a closed contract reason --
    never a path, never captured output, never an OS message."""

    def __init__(self, message, *, reason=contract.StopReason.PREFLIGHT_FAILED):
        super().__init__(message)
        self.reason = reason


# Which terminal state a stop reason lands in. A TABLE rather than a
# judgement at each site: "blocked" and "failed" mean different things to
# an operator -- one is a decision waiting for a human, the other is
# something that broke -- and deciding it separately in nine places is
# how the same reason ends up reported two ways.
_FAILED_REASONS = frozenset({
    contract.StopReason.MODEL_PROCESS_FAILED,
    contract.StopReason.SCHEMA_VIOLATION,
    contract.StopReason.EVALUATOR_MODIFIED_WORKSPACE,
    contract.StopReason.PATH_NOT_ALLOWED,
    contract.StopReason.CONTROL_PLANE_MODIFIED,
    contract.StopReason.TIMEOUT,
})


def _terminal_for(reason):
    return (contract.State.FAILED if reason in _FAILED_REASONS
            else contract.State.BLOCKED)


@dataclass(frozen=True, slots=True)
class RunResult:
    """What happened, as identities, counts and closed codes.

    WHAT IS NOT HERE, and the absence is the point: no raw model output,
    no patch, no source byte, no absolute path, no working directory, no
    environment and no free-text error. Every string in this object is
    either a contract constant or an identifier the runner minted."""

    state: str
    stop_reason: str
    # A LIST, not a tuple, and the type is the contract: `visited` is
    # compared against a literal sequence of states, and a tuple never
    # equals a list however right its contents are. The dataclass is
    # frozen against REBINDING, which is what stops a caller replacing
    # the record of where the run went.
    visited: list = field(default_factory=list)
    run_id: object = None
    workspace_id: object = None
    baseline_sha: object = None
    implementation_rounds: int = 0
    repair_rounds: int = 0
    evaluator_rounds: int = 0
    pending_approval: tuple = ()
    surviving_children: int = 0
    recovered_from_backup: bool = False
    recovered_applications: tuple = ()
    acceptance_passed: bool = False
    applied_files: tuple = ()


class _Stop(Exception):
    """An orderly end to the run, carrying the closed reason and the
    terminal state it belongs in. Internal: it never leaves the module."""

    def __init__(self, reason, *, state, surviving_children=0,
                 pending_approval=(), event=None):
        super().__init__(reason)
        self.reason = reason
        self.state = state
        self.surviving_children = surviving_children
        self.pending_approval = tuple(pending_approval)
        # The lower layer's own closed EVENT code, carried so the journal
        # records WHAT happened and not merely that the run stopped:
        # `output_truncated` and `schema_violation` are the same stop
        # reason at this layer and two very different defects below it.
        self.event = event


class _Interrupted(Exception):
    """The run was cut short on purpose and must stay RESUMABLE: no
    terminal state is written and the workspace is left where it is."""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def state_directory(repo) -> Path:
    """Derived from the repository, never supplied.

    A caller-chosen state root is one typo away from writing run state
    into the document tree, which is why the contract makes the name a
    constant."""
    return Path(repo) / contract.STATE_DIR_NAME


def _mint_run_id() -> str:
    """An identifier the frozen grammar accepts and nothing can read
    anything out of."""
    return f"kosu-{secrets.token_hex(12)}"


@contextlib.contextmanager
def single_instance_lock(repo):
    """Hold this repository's loop for the block, or raise `LockHeld`.

    THE OUTERMOST BOUNDARY. It wraps the existing `locking` authority
    rather than re-deciding what ownership means: ownership is an
    exclusive byte-range lock on an open handle, so a second holder --
    another process, or another handle in this one -- is refused by the
    operating system with no window to race in. Nothing here deletes the
    lock file, for the reason that module documents at length.

    The path is derived from the repository, so the second instance
    cannot take a different lock by naming one."""
    with locking.single_instance_lock(state_directory(repo)) as held:
        yield held


def _exact_binaries(binaries):
    """Both roles, as existing paths the caller named.

    MANDATORY, and checked here as well as at the adapters: this package
    contains no discovery function, so the promise that no real model is
    ever called rests on the caller having to say which files to run."""
    if not isinstance(binaries, dict):
        raise RunnerError("ikili dosya haritasi bir esleme degil")
    missing = [role for role in ("implementer", "evaluator")
               if not binaries.get(role)]
    if missing:
        raise RunnerError(f"eksik ikili dosya: {sorted(missing)}")
    return {"implementer": binaries["implementer"],
            "evaluator": binaries["evaluator"]}


@contextlib.contextmanager
def _call_environment(values):
    """Export the call's closed identities, then put the environment back
    exactly as it was -- including the variables that were ABSENT, which
    a plain overwrite leaves behind for every later call."""
    previous = {name: os.environ.get(name) for name in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for name, before in previous.items():
            if before is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = before


def _recover_applications(repo):
    """Finish every crashed application of this repository, in the
    ROLLBACK direction, before anything else looks at the checkout.

    A half-applied candidate is a dirty tree with a reason, and preflight
    would refuse it without being able to say why. The recovery seam is
    the existing one: a caller supplies an id, never a path, and a
    holder whose journal does not name this repository is somebody
    else's."""
    recovered = []
    for application_id in application.find_pending_applications(repo):
        report = application.recover_application(
            repo, application_id=application_id)
        recovered.append(report.application_id)
    return tuple(recovered)


def preflight(task_path, *, repo, binaries) -> RunResult:
    """Every gate that must pass before a model could be called.

    Costs nothing when it fails: no workspace is built, no state document
    is written and no model process is started. The binaries are still
    mandatory -- an unusable binary is a failure worth learning here
    rather than at the first paid call."""
    binaries = _exact_binaries(binaries)
    repo_path = Path(repo)
    outcome = preflight_gates.run_preflight(
        task_path, repo=repo_path, binaries=binaries,
        state_dir=state_directory(repo_path))
    if not outcome.ok:
        return RunResult(state=contract.State.BLOCKED,
                         stop_reason=outcome.stop_reason,
                         visited=[contract.State.PREFLIGHT],
                         baseline_sha=outcome.baseline_sha)
    return RunResult(state=contract.State.PREFLIGHT,
                     stop_reason=contract.StopReason.COMPLETED,
                     visited=[contract.State.PREFLIGHT],
                     baseline_sha=outcome.baseline_sha)


def run(task_path, *, repo, binaries, audit_kind=contract.AuditKind.CODE,
        _test_edit=None, _test_request_gate=None,
        _test_interrupt_after=None) -> RunResult:
    """One whole run, from preflight to an applied candidate or a stop.

    `binaries` is keyword-only and mandatory. `audit_kind` chooses which
    frozen evaluator protocol this run uses -- the CODE audit, whose
    findings may cite tracked source, or the LOCKED audit, whose findings
    are an error class and a count and have no text field at all. It is a
    closed vocabulary value and neither choice widens anything: the
    locked schema is the stricter of the two, and its ids are minted
    here.

    The three `_test_` hooks are private seams the contract battery uses
    to reach behaviour that would otherwise need a real model to
    misbehave. They are not authority: they cannot name a path outside
    the candidate, cannot approve a gate and cannot write a terminal
    state."""
    binaries = _exact_binaries(binaries)
    if audit_kind not in contract.ALL_AUDIT_KINDS:
        raise RunnerError("denetim turu sozlesmede yok")
    repo_path = Path(repo)
    with single_instance_lock(repo_path):
        return _Run(task_path, repo=repo_path, binaries=binaries,
                    audit_kind=audit_kind, test_edit=_test_edit,
                    test_request_gate=_test_request_gate,
                    test_interrupt_after=_test_interrupt_after).execute()


def resume(*, repo, binaries) -> RunResult:
    """Read back an interrupted or damaged run, and repair what can be
    repaired WITHOUT deciding anything on the operator's behalf.

    Two repairs, both backwards-looking. A `state.json` that does not
    parse is replaced from the backup this runner writes after every
    successful transition -- never guessed at, never reset, because a
    corrupt state is evidence and throwing it away destroys the evidence
    while leaving the cause. A crashed application is rolled back through
    the existing recovery seam.

    It does not continue the loop. Continuing needs the task manifest,
    and a resume that re-derived one from a state directory would be a
    resume that chose which task to run.

    `binaries` is mandatory here too even though nothing is launched: the
    no-discovery rule is uniform across this seam, and an entry point
    that quietly did not need them is one somebody would later teach to
    launch something."""
    binaries = _exact_binaries(binaries)
    repo_path = Path(repo)
    with single_instance_lock(repo_path):
        state_dir = state_directory(repo_path)
        recovered_applications = _recover_applications(repo_path)
        recovered = False
        try:
            payload = state_module.read_state(state_dir)
        except state_module.CorruptState:
            payload = runner_events.restore_state_backup(state_dir)
            recovered = True
        binding = state_module.read_binding(state_dir)
        if binding["run_id"] != payload["run_id"]:
            raise state_module.IncompatibleState(
                "durum ve baglama farkli kosulari gosteriyor")
        current = payload["state"]
        stop_reason = payload.get("stop_reason") or (
            contract.StopReason.INTERRUPTED)
        rounds = payload.get("rounds", {})
        return RunResult(
            state=current, stop_reason=stop_reason, visited=[current],
            run_id=payload["run_id"], workspace_id=binding["workspace_id"],
            baseline_sha=binding["baseline_sha"],
            implementation_rounds=rounds.get("implementation", 0),
            repair_rounds=rounds.get("repair", 0),
            evaluator_rounds=rounds.get("evaluator", 0),
            recovered_from_backup=recovered,
            recovered_applications=recovered_applications)


class _Run:
    """One run's whole life. Everything it needs is derived from the
    manifest and the identities it minted; nothing is a caller's path."""

    def __init__(self, task_path, *, repo, binaries, audit_kind, test_edit,
                 test_request_gate, test_interrupt_after):
        self.task_path = task_path
        self.repo = repo
        self.binaries = binaries
        self.audit_kind = audit_kind
        self.test_edit = test_edit
        self.test_request_gate = test_request_gate
        self.test_interrupt_after = test_interrupt_after
        self.state_dir = state_directory(repo)

        self.run_id = None
        self.workspace_id = None
        self.baseline = None
        self.task = None
        self.digest = None
        self.deadline = None
        self.state = contract.State.PREFLIGHT
        self.visited = [contract.State.PREFLIGHT]
        self.rounds = {"implementation": 0, "repair": 0, "evaluator": 0}
        self.mechanisms_seen = []
        self.seen_mechanisms = set()
        self.candidate_files = ()
        self.issued_ids = None
        self.audit_rounds = []
        self.pending_approval = ()
        self.recovered_applications = ()
        self.acceptance_passed = False
        self.applied_files = ()
        self.max_implementation_rounds = MAX_IMPLEMENTATION_ROUNDS
        self.max_repair_rounds = MAX_REPAIR_ROUNDS

    # -----------------------------------------------------------------
    # the run
    # -----------------------------------------------------------------

    def execute(self) -> RunResult:
        self.recovered_applications = _recover_applications(self.repo)
        outcome = preflight_gates.run_preflight(
            self.task_path, repo=self.repo, binaries=self.binaries,
            state_dir=self.state_dir)
        if not outcome.ok:
            # A PREFLIGHT FAILURE COSTS NOTHING: no state document, no
            # workspace, no journal. Writing a record here would create
            # the state directory a later run then has to reason about.
            return self._result(contract.State.BLOCKED, outcome.stop_reason)
        self.task = outcome.manifest.task
        self.digest = outcome.manifest.digest
        self.baseline = outcome.baseline_sha
        self.run_id = _mint_run_id()
        self.max_implementation_rounds = min(
            MAX_IMPLEMENTATION_ROUNDS, self.task["max_implementation_rounds"])
        self.max_repair_rounds = min(MAX_REPAIR_ROUNDS,
                                     self.task["max_repair_rounds"])
        # ONE absolute deadline for the whole run, set before anything
        # starts. No phase gets a fresh clock.
        self.deadline = time.monotonic() + \
            self.task["max_wall_clock_minutes"] * 60

        self._open_state()
        try:
            self._gate_check()
            self._assert_budget()
            self._assert_clock()
            self._open_workspace()
            return self._walk()
        except _Stop as stop:
            if stop.event is not None:
                self._event(stop.event, state=self.state)
            if self.state not in contract.TERMINAL_STATES:
                self._advance(stop.state, stop_reason=stop.reason)
            return self._result(stop.state, stop.reason,
                                surviving_children=stop.surviving_children,
                                pending_approval=stop.pending_approval)
        except _Interrupted:
            # NO TERMINAL STATE and NO CLEANUP. The state document stays
            # exactly where the last completed phase left it, which is
            # what makes the run resumable at all, and the workspace it
            # names is still on disk.
            self._event(contract.EventCode.INTERRUPTED, state=self.state)
            return self._result(self.state, contract.StopReason.INTERRUPTED)

    def _walk(self) -> RunResult:
        """The only legal order. Each phase is entered by a transition
        the frozen table allows, and every one of them is recorded before
        the work inside it starts."""
        self._advance(contract.State.IMPLEMENTING)
        verified = self._implement()
        self._checkpoint(contract.State.IMPLEMENTING)

        self._advance(contract.State.ACCEPTANCE)
        report = self._accept(verified)
        self._checkpoint(contract.State.ACCEPTANCE)

        self._advance(contract.State.AUDITING)
        approved = self._audit(contract.State.AUDITING)
        self._checkpoint(contract.State.AUDITING)
        if approved:
            return self._approve(verified, report)

        self._advance(contract.State.REPAIRING)
        verified = self._repair(verified)
        self._checkpoint(contract.State.REPAIRING)

        self._advance(contract.State.ACCEPTANCE_2)
        report = self._accept(verified)
        self._checkpoint(contract.State.ACCEPTANCE_2)

        self._advance(contract.State.FINAL_AUDITING)
        # A `changes_requested` here never returns: `_audit` raises, with
        # `repeated_mechanism_failure` when the mechanism is one this run
        # has already seen and `repair_rounds_exhausted` when it is a new
        # one the budget cannot pay for. THERE IS NO SECOND REPAIR EDGE,
        # in this method or in the frozen transition table.
        self._audit(contract.State.FINAL_AUDITING)
        self._checkpoint(contract.State.FINAL_AUDITING)
        return self._approve(verified, report)

    # -----------------------------------------------------------------
    # identity, state and the journal
    # -----------------------------------------------------------------

    @property
    def identity(self):
        """The tuple every downstream seam takes. Identities only: no
        workspace path, no cwd, no allow list, no diff."""
        return {"repo": self.repo, "state_dir": self.state_dir,
                "task_path": self.task_path, "manifest_digest": self.digest,
                "run_id": self.run_id, "workspace_id": self.workspace_id,
                "baseline_sha": self.baseline}

    def _open_state(self):
        state_module.ensure_directory(self.state_dir)
        stamp = _now()
        payload = {"protocol_version": contract.PROTOCOL_VERSION,
                   "run_id": self.run_id, "state": contract.State.PREFLIGHT,
                   "started_at": stamp, "updated_at": stamp,
                   "baseline_sha": self.baseline,
                   "rounds": dict(self.rounds),
                   "budget": {"max_usd": self.task["max_budget_usd"],
                              "spent_usd": 0.0}}
        state_module.write_state(self.state_dir, payload)
        runner_events.save_state_backup(self.state_dir, payload)
        self._event(contract.EventCode.PREFLIGHT_OK,
                    state=contract.State.PREFLIGHT)

    def _open_workspace(self):
        """The two flat trees, then the binding that names them.

        The binding cannot be written first: it carries the execution
        identity, and there is no identity until the workspace exists."""
        workspace = flat_workspace.create(self.repo, state_dir=self.state_dir,
                                          run_id=self.run_id,
                                          baseline_sha=self.baseline)
        self.workspace_id = workspace.workspace_id
        state_module.write_binding(self.state_dir, {
            "protocol_version": contract.PROTOCOL_VERSION,
            "run_id": self.run_id,
            "repo_id": state_module.repo_identity(self.repo),
            "baseline_sha": self.baseline, "manifest_digest": self.digest,
            "workspace_id": self.workspace_id})

    def _advance(self, target, **fields):
        """One legal step, persisted, journalled and backed up.

        THE BACKUP IS WRITTEN AFTER THE MOVE, so it always holds the last
        state that was actually good. Keeping the PREVIOUS document
        instead would mean a recovery rewinds a step nobody asked to
        undo."""
        payload = state_module.advance(
            self.state_dir, target, updated_at=_now(),
            rounds=dict(self.rounds),
            mechanisms_seen=[list(group) for group in self.mechanisms_seen],
            **fields)
        runner_events.save_state_backup(self.state_dir, payload)
        self.state = target
        self.visited.append(target)
        self._event(contract.EventCode.STATE_TRANSITION, state=target)
        return payload

    def _event(self, code, *, state=None, **fields):
        payload = {"ts": _now(), "run_id": self.run_id, "event": code}
        if state is not None:
            payload["state"] = state
        payload.update({name: value for name, value in fields.items()
                        if value is not None})
        runner_events.append_event(self.state_dir, payload)

    def _checkpoint(self, phase):
        if self.test_interrupt_after == phase:
            raise _Interrupted(phase)

    def _result(self, state, stop_reason, *, surviving_children=0,
                pending_approval=()):
        return RunResult(
            state=state, stop_reason=stop_reason, visited=list(self.visited),
            run_id=self.run_id, workspace_id=self.workspace_id,
            baseline_sha=self.baseline,
            implementation_rounds=self.rounds["implementation"],
            repair_rounds=self.rounds["repair"],
            evaluator_rounds=self.rounds["evaluator"],
            pending_approval=tuple(pending_approval) or self.pending_approval,
            surviving_children=surviving_children,
            recovered_applications=self.recovered_applications,
            acceptance_passed=self.acceptance_passed,
            applied_files=self.applied_files)

    # -----------------------------------------------------------------
    # gates that cost nothing
    # -----------------------------------------------------------------

    def _gate_check(self):
        """A gated action stops and asks. It is never performed, and no
        step after this one runs -- so there is no workspace, no mirror,
        no model process and nothing outward-facing to undo."""
        gate = self.test_request_gate
        if gate is None:
            return
        if gate not in contract.USER_APPROVAL_REQUIRED:
            raise RunnerError("kapi sozlesmede yok")
        self.pending_approval = (gate,)
        self._event(contract.EventCode.GATE_REQUIRED, state=self.state)
        raise _Stop(contract.StopReason.USER_APPROVAL_REQUIRED,
                    state=contract.State.BLOCKED, pending_approval=(gate,))

    def _remaining_budget(self):
        """What is left, from the persisted ledger rather than a local
        counter: the invariant the contract names is about the recorded
        figures, and a runner that checked its own variables would be
        checking something no crash ever sees."""
        budget = state_module.read_state(self.state_dir)["budget"]
        return budget["max_usd"] - budget["spent_usd"]

    def _assert_budget(self):
        """Checked BEFORE a call. A budget enforced afterwards has
        already been spent."""
        remaining = self._remaining_budget()
        self._event(contract.EventCode.BUDGET_CHECK, state=self.state)
        if remaining <= 0:
            raise _Stop(contract.StopReason.BUDGET_EXHAUSTED,
                        state=contract.State.BLOCKED)
        return remaining

    def _remaining_seconds(self):
        return self.deadline - time.monotonic()

    def _assert_clock(self):
        """Asked beside the budget, before anything is built.

        A run whose wall clock is already spent must not create a
        workspace, materialise a baseline or start a process: the deadline
        is absolute, so "there is no time left" is as true here as it
        would be three phases later, and learning it here costs
        nothing."""
        if self._remaining_seconds() <= 0:
            raise _Stop(contract.StopReason.WALL_CLOCK_EXCEEDED,
                        state=contract.State.BLOCKED)

    def _model_seconds(self):
        """`min(call limit, what is left of the run)`.

        A call that cannot be given the contract's own minimum does not
        start: handing an adapter a value outside the frozen range would
        be refused there anyway, and reporting that as an input error
        would name the wrong defect."""
        limit = self.task.get("model_call_timeout_seconds",
                              contract.DEFAULTS["model_call_timeout_seconds"])
        seconds = int(min(limit, self._remaining_seconds()))
        if seconds < MIN_MODEL_SECONDS:
            raise _Stop(contract.StopReason.WALL_CLOCK_EXCEEDED,
                        state=contract.State.BLOCKED)
        return min(seconds, MAX_MODEL_SECONDS)

    def _acceptance_seconds(self):
        seconds = int(self._remaining_seconds())
        if seconds <= 0:
            raise _Stop(contract.StopReason.WALL_CLOCK_EXCEEDED,
                        state=contract.State.BLOCKED)
        return seconds

    def _model(self, role):
        configured = self.task.get(role)
        if isinstance(configured, dict):
            return configured.get("model")
        return None

    def _translate(self, failure):
        """A lower layer's typed refusal becomes this run's stop.

        The REASON is the lower layer's own -- every module in this
        package fixes one on its refusals -- so nothing is re-decided
        here and two gates cannot end up behind one message. Anything
        without a closed reason is not ours to classify and flies on
        unchanged."""
        reason = getattr(failure, "reason", None)
        if not isinstance(reason, str) or reason not in \
                contract.ALL_STOP_REASONS:
            return None
        survived = 0 if getattr(failure, "cleanup_complete", True) else 1
        event = getattr(failure, "event", None)
        if event not in contract.ALL_EVENT_CODES:
            event = None
        return _Stop(reason, state=_terminal_for(reason),
                     surviving_children=survived, event=event)

    @contextlib.contextmanager
    def _guard(self):
        try:
            yield
        except _Stop:
            raise
        except Exception as failure:                   # noqa: BLE001
            stop = self._translate(failure)
            if stop is None:
                raise
            raise stop from failure

    # -----------------------------------------------------------------
    # the implementer
    # -----------------------------------------------------------------

    def _model_environment(self, round_index, issued=None):
        values = {ENV_RUN_ID: self.run_id, ENV_ROUND: str(round_index)}
        if issued:
            values[ENV_LOCKED_RUN_ID] = issued["run_id"]
            values[ENV_FINDING_IDS] = ",".join(issued["finding_ids"])
            values[ENV_MECHANISM_IDS] = ",".join(issued["mechanism_ids"])
        return _call_environment(values)

    def _apply_test_edit(self):
        """The private edit hook, written INSIDE the candidate.

        It cannot reach anything else: the names are joined onto the
        implementer root and refused if they resolve outside it, so the
        hook can stage a forbidden-path violation for the change-set gate
        to catch without being able to commit one.

        IT RUNS AFTER THE MODEL CALL, not before, and that is a
        measurement rather than a preference. `run_verified_implementation`
        opens by requiring the reference and implementer trees to be
        EQUAL -- that equality is what makes "anything that differs
        afterwards was done by this call" true for a first round. A hook
        that pre-loaded a file would therefore be refused as a divergent
        workspace, and a test written for the path-authorisation gate
        would go red at the attribution gate instead. Red for the wrong
        reason still counts as red."""
        if not self.test_edit:
            return
        root = flat_workspace.assert_binding(
            self.repo, state_dir=self.state_dir, run_id=self.run_id,
            workspace_id=self.workspace_id,
            baseline_sha=self.baseline).implementer_root
        base = root.resolve()
        for relative, text in dict(self.test_edit).items():
            target = (base / relative).resolve()
            if target != base and base not in target.parents:
                raise RunnerError("deneme duzenlemesi aday agacin disinda")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")

    def _implement(self):
        # The cap is the SMALLER of the contract's and the manifest's, and
        # it is asked rather than assumed: `_walk` calls this once, and a
        # constant that only documents a limit is a limit nobody enforces.
        if self.rounds["implementation"] >= self.max_implementation_rounds:
            raise _Stop(contract.StopReason.REPAIR_ROUNDS_EXHAUSTED,
                        state=contract.State.BLOCKED)
        budget = self._assert_budget()
        seconds = self._model_seconds()
        self._event(contract.EventCode.MODEL_CALL_STARTED, state=self.state)
        with self._guard():
            with self._model_environment(self.rounds["implementation"]):
                verified = changes.run_verified_implementation(
                    self.binaries["implementer"], **self.identity,
                    prompt=self._implementer_prompt(), budget_usd=budget,
                    timeout_seconds=seconds,
                    max_output_bytes=self.task["max_output_bytes"],
                    model=self._model("implementer"))
        self._apply_test_edit()
        self.rounds["implementation"] += 1
        self._event(contract.EventCode.MODEL_CALL_FINISHED, state=self.state,
                    exit_code=verified.exit_code,
                    duration_ms=verified.duration_ms)
        self.candidate_files = verified.changed_files
        return verified

    def _repair(self, previous):
        """ONE repair, bound to the candidate the evaluator audited.

        The previous fingerprint AND the previous file set travel into
        the seam: a repair aimed at a workspace whose contents moved
        since the audit is a repair of something nobody reviewed."""
        budget = self._assert_budget()
        seconds = self._model_seconds()
        self._event(contract.EventCode.MODEL_CALL_STARTED, state=self.state)
        with self._guard():
            with self._model_environment(self.rounds["repair"] + 1):
                verified = changes.run_verified_repair(
                    self.binaries["implementer"], **self.identity,
                    previous_fingerprint=previous.fingerprint,
                    previous_changed_files=previous.changed_files,
                    prompt=self._repair_prompt(), budget_usd=budget,
                    timeout_seconds=seconds,
                    max_output_bytes=self.task["max_output_bytes"],
                    model=self._model("implementer"))
        self.rounds["repair"] += 1
        self._event(contract.EventCode.MODEL_CALL_FINISHED, state=self.state,
                    exit_code=verified.exit_code,
                    duration_ms=verified.duration_ms)
        self.candidate_files = verified.changed_files
        return verified

    # -----------------------------------------------------------------
    # acceptance
    # -----------------------------------------------------------------

    def _accept(self, verified):
        """The frozen commands, in a disposable mirror, EVERY round.

        A second candidate is a second question: the repair's acceptance
        run invalidates the first receipt the moment it begins and writes
        a new one for the tree that actually exists now."""
        self.acceptance_passed = False
        seconds = self._acceptance_seconds()
        self._event(contract.EventCode.ACCEPTANCE_STARTED, state=self.state)
        with self._guard():
            report = acceptance.run_acceptance(
                **self.identity, verified_changes=verified,
                timeout_seconds=seconds,
                max_output_bytes=self.task["max_output_bytes"])
        self._event(contract.EventCode.ACCEPTANCE_FINISHED, state=self.state,
                    duration_ms=report.total_duration_ms)
        if not report.passed:
            raise _Stop(contract.StopReason.ACCEPTANCE_FAILED,
                        state=contract.State.BLOCKED)
        self.acceptance_passed = True
        return report

    # -----------------------------------------------------------------
    # the audit
    # -----------------------------------------------------------------

    def _locked_ids(self):
        """The ids a LOCKED audit may use, minted ONCE FOR THE RUN.

        The static opaque pattern proves shape only -- any 32 hex
        characters satisfy it -- so the binding is an allowlist: the
        per-call schema pins the run id with `const` and each finding and
        mechanism id to an `enum` of exactly what was issued.

        MINTED PER RUN AND NOT PER CALL, which is the difference between
        the second-patch rule working and being unenforceable. A
        mechanism is identified by its id COMPARED ACROSS ROUNDS; if the
        final audit were handed a fresh allowlist, the evaluator could
        not name the mechanism it named the first time even when it is
        looking at exactly the same broken thing, and
        `repeated_mechanism_failure` would be a rule nothing could ever
        trip."""
        if self.issued_ids is None:
            self.issued_ids = {
                "run_id": secrets.token_hex(16),
                "finding_ids": [secrets.token_hex(16)
                                for _ in range(LOCKED_ID_COUNT)],
                "mechanism_ids": [secrets.token_hex(16)
                                  for _ in range(LOCKED_ID_COUNT)]}
        return self.issued_ids

    def _candidate_now(self):
        """The candidate as the filesystem has it RIGHT NOW, through the
        change-set module's own derivation. Never replayed from a
        record."""
        return changes.derive_candidate_changes(**self.identity)

    def _audit(self, phase):
        """One READ-ONLY evaluator call, proven read-only afterwards.

        Two roots are measured across the call and neither is optional:
        the candidate the evaluator judges, and the operator's checkout,
        which the evaluator has no business touching at all. A reviewer
        that wrote is a broken protocol -- there is no automatic
        rollback, because restoring the files would not restore the
        trust."""
        if self.rounds["evaluator"] >= MAX_EVALUATOR_ROUNDS:
            raise _Stop(contract.StopReason.REPAIR_ROUNDS_EXHAUSTED,
                        state=contract.State.BLOCKED)
        round_index = self.rounds["evaluator"]
        self._assert_budget()
        seconds = self._model_seconds()
        issued = (self._locked_ids()
                  if self.audit_kind == contract.AuditKind.LOCKED else None)

        # THE JOURNAL IS WRITTEN BEFORE THE FIRST READING, and that
        # ordering is load-bearing rather than tidy. The main-checkout
        # guard below compares the WHOLE operator checkout across the
        # call, and the state directory lives inside it -- so an event
        # appended between the two readings is a change this run made to
        # the tree it is about to accuse the evaluator of touching. The
        # guard stays total (nothing under `.agent-loop/` is excused)
        # precisely because nothing is written while it is open.
        self._event(contract.EventCode.MODEL_CALL_STARTED, state=self.state)
        policy = changes.freeze_main_policy(self.repo)
        main_key = secrets.token_bytes(fs_evidence.KEY_BYTES)
        with self._guard():
            main_before = changes.main_projection(self.repo, key=main_key,
                                                  policy=policy)
            before = self._candidate_now()

        failure = None
        try:
            with self._guard():
                with self._model_environment(round_index, issued):
                    outcome = audit.run_evaluator(
                        self.binaries["evaluator"], repo=self.repo,
                        state_dir=self.state_dir, run_id=self.run_id,
                        workspace_id=self.workspace_id,
                        baseline_sha=self.baseline,
                        audit_kind=self.audit_kind,
                        prompt=self._evaluator_prompt(before),
                        timeout_seconds=seconds,
                        max_output_bytes=self.task["max_output_bytes"],
                        issued_run_id=issued["run_id"] if issued else None,
                        issued_finding_ids=issued["finding_ids"] if issued
                        else (),
                        issued_mechanism_ids=issued["mechanism_ids"] if issued
                        else (),
                        model=self._model("evaluator"))
        except BaseException as raised:
            failure = raised
            raise
        finally:
            # EVERY exit, including a failed call: an evaluator that
            # crashed may have written first, and a workspace violation
            # outranks the failure that hid it. The original error is
            # chained, never erased.
            try:
                self._assert_read_only(before, main_before, main_key, policy)
            except _Stop as violation:
                raise violation from failure

        self.rounds["evaluator"] += 1
        self._event(contract.EventCode.MODEL_CALL_FINISHED, state=self.state,
                    exit_code=outcome.exit_code,
                    duration_ms=outcome.duration_ms)
        return self._verdict(outcome, phase, round_index)

    def _assert_read_only(self, before, main_before, main_key, policy):
        """Proven by evidence taken before and after, not by a flag.

        A refusal from the re-derivation counts as a modification: the
        evaluator's own write is exactly what turns an authorised
        candidate into one the change-set gate will not describe, and
        `contracts/gizli.md` appearing under the candidate root raises
        rather than showing up as a different fingerprint."""
        try:
            after = self._candidate_now()
            moved = after.fingerprint != before.fingerprint
            main_after = changes.main_projection(self.repo, key=main_key,
                                                 policy=policy)
            moved = moved or bool(changes.main_difference(main_before,
                                                          main_after))
        except changes.ChangeSetError:
            moved = True
        if moved:
            self._event(contract.EventCode.EVALUATOR_MODIFIED_WORKSPACE,
                        state=self.state)
            raise _Stop(contract.StopReason.EVALUATOR_MODIFIED_WORKSPACE,
                        state=contract.State.FAILED)

    def _verdict(self, outcome, phase, round_index):
        """What the reply MEANS, decided against the run's own counters.

        Returns True for an approval and False when a repair is owed.
        Every other outcome raises: this method is the only place a
        `changes_requested` can become a second round, and it refuses to
        be that place twice."""
        reply = outcome.reply
        status = reply["status"]
        findings = reply.get("findings", ())
        self._record_findings(phase, outcome, round_index)
        if status == contract.Status.APPROVED:
            return True
        if status in (contract.Status.BLOCKED, contract.Status.FAILED):
            # the frozen schema requires a closed `stop_reason` on both
            # of these, so there is nothing to guess at here
            raise _Stop(reply["stop_reason"], state=contract.State.BLOCKED)
        self._assert_in_scope(findings)
        mechanisms = [finding["mechanism_id"] for finding in findings]
        repeated = sorted({name for name in mechanisms
                           if name in self.seen_mechanisms})
        self.mechanisms_seen.append(sorted(set(mechanisms)))
        if repeated:
            # THE RULE THE WHOLE LOOP EXISTS FOR. A mechanism that broke
            # twice needs a design change, not a third heuristic patch.
            raise _Stop(contract.StopReason.REPEATED_MECHANISM_FAILURE,
                        state=contract.State.BLOCKED)
        self.seen_mechanisms.update(mechanisms)
        if self.rounds["repair"] >= self.max_repair_rounds:
            # A DIFFERENT mechanism each round still stops at the budget
            # of one repair: the round cap and the second-patch rule are
            # two separate limits and either alone is escapable.
            raise _Stop(contract.StopReason.REPAIR_ROUNDS_EXHAUSTED,
                        state=contract.State.BLOCKED)
        return False

    def _assert_in_scope(self, findings):
        """A finding may only require action on a file THIS run changed.

        Anything else is a repair aimed outside what the task authorised
        and outside what the acceptance gate measured -- which is scope
        widening, and scope widening is a human gate rather than
        something a repair round may do quietly."""
        changed = {changes.canonical_path(path)
                   for path in self.candidate_files}
        for finding in findings:
            cited = finding.get("file")
            if cited is None:
                continue
            if changes.canonical_path(cited) not in changed:
                raise _Stop(contract.StopReason.OUT_OF_SCOPE_FINDING,
                            state=contract.State.BLOCKED)

    def _record_findings(self, phase, outcome, round_index):
        """The findings artefact, through the journal's allowlist.

        A LOCKED finding arrives as an error class and a count and is
        written as an error class and a count; a CODE finding keeps its
        file and line and loses its prose, because model-authored text
        does not belong in a file that outlives the run."""
        reply = outcome.reply
        record = {"round": round_index, "state": phase,
                  "audit_kind": outcome.audit_kind,
                  "status": reply["status"],
                  "findings": [runner_events.record_finding(finding,
                                                            outcome.audit_kind)
                               for finding in reply.get("findings", ())]}
        if "summary_code" in reply:
            record["summary_code"] = reply["summary_code"]
        self.audit_rounds.append(record)
        runner_events.write_findings(self.state_dir, {
            "protocol_version": contract.PROTOCOL_VERSION,
            "run_id": self.run_id, "rounds": list(self.audit_rounds)})

    # -----------------------------------------------------------------
    # application
    # -----------------------------------------------------------------

    def _approve(self, verified, report):
        """The candidate reaches the operator's checkout, or the run does
        not end approved.

        The report handed over is the one `run_acceptance` RETURNED, and
        the application layer does not believe it on its own: it re-reads
        the receipt that call persisted and compares the two. So a forged
        report -- an exact-typed instance somebody constructed -- has
        nothing behind it, and the state below is only reachable after
        the move actually returned."""
        with self._guard():
            applied = application.apply_accepted_candidate(
                **self.identity, verified_changes=verified,
                acceptance_report=report)
        self.applied_files = applied.applied_files
        self._advance(contract.State.APPROVED,
                      stop_reason=contract.StopReason.COMPLETED)
        self._release_workspace()
        return self._result(contract.State.APPROVED,
                            contract.StopReason.COMPLETED)

    def _release_workspace(self):
        """On a successful end ONLY. The candidate is now in the
        checkout, so the copy is redundant -- while a run that stopped
        still has the only copy of work a human is about to look at, and
        an interrupted one needs it to be resumable at all."""
        if self.workspace_id is None:
            return
        flat_workspace.remove(self.repo, state_dir=self.state_dir,
                              workspace_id=self.workspace_id)

    # -----------------------------------------------------------------
    # prompts -- built from the manifest, never from a model's output
    # -----------------------------------------------------------------

    def _common_prompt(self):
        forbidden = self.task.get("forbidden_paths", ())
        return [
            f"KOSU: {self.run_id}",
            f"HEDEF: {self.task['objective']}",
            "IZINLI YOLLAR: " + ", ".join(self.task["allowed_paths"]),
            "YASAK YOLLAR: " + (", ".join(forbidden) or "-"),
            "KABUL OLCUTLERI:",
            *[f"  - {item}" for item in self.task["acceptance_criteria"]],
        ]

    def _implementer_prompt(self):
        return "\n".join([
            "ROL: implementer",
            *self._common_prompt(),
            "KURALLAR:",
            "  - yalnizca izinli yollarda duzenleme yap",
            "  - kabul olcutlerini degistirme",
            "  - komut calistirma; dogrulamayi kosucu yapar",
            "  - yaniti sozlesme semasina gore ver",
        ])

    def _repair_prompt(self):
        """The findings, as the boundary allows them to travel.

        A CODE finding may carry its file, its line and its sentences,
        because the implementer can already open that file. A LOCKED
        finding carries an error CLASS and a COUNT and nothing else --
        there is no field a passage could travel in, and this method has
        no access to one either: it reads the artefact the journal wrote,
        which was already reduced to the transferable fields."""
        lines = ["ROL: implementer (onarim)", *self._common_prompt(),
                 "BULGULAR:"]
        latest = self.audit_rounds[-1] if self.audit_rounds else {}
        kind = latest.get("audit_kind")
        for finding in latest.get("findings", ()):
            if kind == contract.AuditKind.LOCKED:
                lines.append(
                    f"  - mekanizma {finding['mechanism_id']}: "
                    f"{finding['error_class']} x{finding['case_count']}")
            else:
                where = finding.get("file", "-")
                lines.append(f"  - mekanizma {finding['mechanism_id']} "
                             f"({finding['severity']}) {where}")
        lines += ["KURALLAR:",
                  "  - yalnizca bir onarim turu var; kapsam genisletme",
                  "  - yalnizca izinli yollarda duzenleme yap"]
        return "\n".join(lines)

    def _evaluator_prompt(self, candidate):
        return "\n".join([
            "ROL: evaluator (salt okunur)",
            *self._common_prompt(),
            f"ADAY PARMAK IZI: {candidate.fingerprint}",
            "DEGISEN DOSYALAR:",
            *[f"  - {path}" for path in candidate.changed_files],
            "KURALLAR:",
            "  - hicbir dosyayi degistirme",
            "  - kapsam disinda bulgu bildirme",
            "  - her bulguya bir mekanizma kimligi ver",
        ])
