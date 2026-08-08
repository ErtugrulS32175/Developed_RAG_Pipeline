"""FROZEN CONTRACT for candidate publication and ingest attempts.

CONTRACT ONLY -- no behaviour lives in this module. It fixes the
vocabulary (states, seams, error kinds, the attempt's fields) so the red
tests can name what they require and the implementation has one target it
cannot quietly miss. Nothing here reads, writes or decides.

WHY IT EXISTS. Three defects share one root: the system had no identity
for a RUN. It had one for a document (id), for a name (filename), and --
after round 18 -- for a content version (candidate_id). It had none for
"this attempt to index that content", so every guard built from those
three could be satisfied by the wrong actor:

  * The CLI path bound to nothing and re-knocked the candidate gate
    AFTER parsing. Its stale hash equalled the SERVED hash, which is a
    legitimate arm of the gate, so a brand-new valid candidate id was
    minted at the wrong moment and a stale snapshot was promoted over a
    newer authorised upload.
  * Two runs of the SAME candidate were indistinguishable. The loser
    re-read the active generation after parsing, re-bound itself to what
    the winner had just promoted, and stamped its own failure onto a
    healthy served generation.
  * The upload published to the database and to the disk at two
    different moments. A process started in between read a candidate
    whose bytes were not on disk yet, refused correctly, and marked the
    document error -- while the upload returned 200 pending. Two
    truthful answers, one contradictory record.


TWO STATUSES, TWO SUBJECTS
--------------------------

The third defect was a column carrying two meanings. They are separated
here and may never be merged again:

    documents.status          describes the SERVED version. It changes
                              only at promotion (or at a failure of the
                              serving generation itself). An upload does
                              NOT move it: uploading does not un-index
                              what is already being answered from.

    documents.candidate_state describes the CANDIDATE's lifecycle:
                              STAGED -> PUBLISHED.

    the upload response       reports the CANDIDATE's status ("pending"
                              = published but not indexed yet). It is
                              not the document's status and must never
                              be read as one.


CANDIDATE STATE MACHINE
-----------------------

    (no candidate) --stage_candidate--> STAGED
                                          |
                                   os.replace(bytes)
                                          |
                        finalize_candidate_publication
                                          |
                                       PUBLISHED --begin_attempt--> lease

    publish_candidate = the TOP-LEVEL operation that performs all three
    under one publish lock: stage, replace the bytes, finalize. The
    three names are distinct on purpose -- an earlier draft used one
    name for the staging step, for the finalisation and for the whole
    operation, which is three contracts wearing one word.

A candidate is PUBLISHED only when the row and the bytes agree. Until
then it does not exist for any consumer. A crashed upload leaves STAGED,
which is never processable and is superseded by the next publication.

CRASH WINDOWS, both idempotent (rule 11): a crash between stage and
replace leaves STAGED with old bytes -- re-running publish_candidate
with the same bytes reaches PUBLISHED and cancels nothing; a crash
between replace and finalize leaves STAGED with NEW bytes -- re-running
finalizes rather than re-staging, because the recorded hash already
matches the disk.


WHERE EACH RESPONSIBILITY LIVES
-------------------------------

An earlier draft asked ONE callable for two opposite behaviours: the
documented entry point had to refuse running without an attempt AND to
publish-then-begin one. Those are two jobs, so they are two names:

    pipeline.index.publication.publish_candidate(...)
        the shared publication SERVICE. API and CLI both go through it;
        it owns the lock, the three steps and the crash-window recovery.

    pipeline.index.ingest.cli_main(argv)
        the CLI: parse args -> publish_candidate -> begin_attempt ->
        ingest_attempt(snapshot, THAT attempt). Takes an explicit
        --replace; nothing else grants replacement. It RETURNS AN EXIT
        CODE and does not raise for contract refusals -- a CLI that
        crashed and a CLI that refused look identical to a caller, and
        an earlier test suite accepted either. The whole chain is
        mandatory: there is no success without the core call.

            EXIT_OK                        0   indexed
            EXIT_CANDIDATE_CONFLICT        2   different content, no
                                               --replace
            EXIT_CANDIDATE_NOT_PUBLISHED   3   candidate still STAGED
            EXIT_ATTEMPT_UNAVAILABLE       4   another attempt holds the
                                               lease
            EXIT_ATTEMPT_LOST              5   fenced, or the lease was
                                               taken over mid-run

    pipeline.index.ingest.ingest_attempt(snapshot, attempt)
        the CORE. An attempt is MANDATORY -- there is no unbound path
        into the index at all. It RETURNS THIS RUN'S OWN VERDICT,
        ``(outcome, note)`` with outcome in {DONE, PARTIAL}, because the
        caller cannot read that verdict anywhere else: rule 5 puts
        `error` and `partial` on the ATTEMPT and leaves the document's
        status describing the SERVED version. An API that stamped
        `processing` and then read the document back to find out how the
        run went therefore saw a real PARTIAL run as "never finished" --
        it answered 500 and relabelled a healthy index. Failures are
        raised, not returned; only the two terminal successes come back.

    pipeline.index.db
        stage_candidate / finalize_candidate_publication /
        begin_attempt / heartbeat_attempt / record_attempt_outcome /
        promote_generation. Statements only; no publication policy.


THE ATTEMPT
-----------

`IngestAttempt` is minted once by begin_attempt and never modified:

    attempt_id       identity of THIS RUN -- not of the content. It is
                     also the FENCING TOKEN: every write this run makes
                     compare-and-swaps on it.
    document_id      the row being indexed
    candidate_id     the content version being indexed
    candidate_sha    that version's bytes
    observed_active  the active generation AT ATTEMPT START

begin_attempt performs, in ONE transaction: verify the candidate is
PUBLISHED, read the active generation, take the lease. It must NOT wait
on the long publish lock -- it observes a STAGED candidate atomically and
raises `CandidateNotPublished`, which the API answers as HTTP 409. A
process request must never block behind an upload's disk write.

THE LEASE lives on the document row:

    attempt_id          the current holder -- the fencing token
    attempt_owner       which worker holds it (host/pid/uuid), for
                        operators reading the row, never for authority
    attempt_expires_at  DATABASE clock, never the worker's

`heartbeat_attempt` extends `attempt_expires_at` while a run is alive,
and the new value is STRICTLY GREATER than the old one -- an
implementation that returns True without moving the clock forward is a
no-op wearing a heartbeat's name, and `>=` would accept it. An EXPIRED
but NOT YET TAKEN lease may be revived by its own holder: expiry makes a
lease TAKEABLE, it does not by itself transfer ownership. Once someone
takes over, the displaced worker's heartbeat raises `AttemptLeaseLost`;
a heartbeat racing a takeover has exactly one winner.

A `begin_attempt` against a LIVE lease raises `AttemptAlreadyRunning`.
Against an EXPIRED lease it takes over: the row's attempt_id becomes the
newcomer's, and every later write by the displaced worker fails its CAS
and raises `AttemptLeaseLost`.

THE CLOSURE PROTOCOL, when a DIFFERENT candidate arrives:

  * The fence lands at STAGE, not at publish. The moment a different
    candidate is staged, the live attempt is indexing bytes nobody will
    serve; making it run until the publication finishes would let it
    promote in that window.
  * The displaced attempt becomes TERMINAL `SUPERSEDED` in that same
    statement -- written by the system, never by the displaced worker.
  * `attempt_id`, `attempt_owner` and `attempt_expires_at` are cleared
    together with it: a fenced lease that lingers would block the new
    candidate behind a run that can no longer do anything.
  * A new attempt may begin only once the new candidate is PUBLISHED --
    the fence does not hand the lease to the newcomer, it empties it.


RULES, FROZEN
-------------

 1. `observed_active` is read at attempt start and NEVER re-read. A run
    that re-reads after parsing re-binds itself to another run's result;
    that is defect two, exactly.
 2. Publishing a new, DIFFERENT candidate atomically FENCES any live
    attempt. A fenced attempt raises `AttemptFenced` and writes nothing.
 3. An idempotent upload -- the SAME bytes again -- neither cancels a
    running attempt, nor rotates the candidate id, nor moves the
    document's status.
 4. Promotion compare-and-swaps active generation, candidate id AND
    attempt id together; the swap, the lease release (attempt_id, owner
    and expires_at all cleared), the attempt's terminal `DONE` and the
    document's `done` status are ONE transaction. A partial application
    is not a weaker success, it is a corrupt record -- so the claim is
    proven by FAULT INJECTION: a failure raised in the middle must leave
    the document row, the chunk rows, the lease fields and the attempt
    record byte-identical to the pre-promotion snapshot, observed from a
    SEPARATE connection. Reading a successful end state proves nothing
    about atomicity; an implementation that commits between the steps
    passes that reading.
 5. WHILE IT STILL HOLDS ITS LEASE, an attempt records its own verdict
    (`error`, `partial`) against its OWN attempt -- that attempt's id and
    its start-time observed_active -- and never on the document. This is
    the ordinary ending of a run that fails BEFORE promotion: a
    single-holder lease means there is no second run to lose a race to,
    so the scenario is "the lease holder failed", not "someone else
    promoted first". The document's status is untouched by it.
 6. A worker whose lease was TAKEN OVER may write nothing at all --
    outcome, status and promotion each raise `AttemptLeaseLost`. Its
    attempt is not left dangling: the takeover itself closes the old
    attempt as `SUPERSEDED`, so the record is written by the SYSTEM at
    the moment of takeover, never by the displaced worker afterwards.
    Rules 5 and 6 divide on one question -- does this worker still hold
    the lease? An earlier draft blurred that line and asked for both.
 7. Ingest never knocks the candidate gate. Candidates are recorded by
    publication only, and the CLI is a first-class publisher through the
    SAME internal protocol as the API:
        publish_candidate -> begin_attempt -> ingest(attempt)
    Core ingest REFUSES to run without an attempt; there is no unbound
    production path. Different content requires an explicit `--replace`
    flag, and there is no implicit replacement through an environment
    variable.
 8. Legacy and NULL rows stay fail-closed: what cannot be verified is
    not processable.
 9. Two concurrent `begin_attempt` calls for one candidate: exactly one
    holds the lease; the other raises `AttemptAlreadyRunning`.
10. `stage_candidate` -- the ONE gate a candidate may enter through --
    refuses an offer that equals the SERVED bytes while a DIFFERENT
    candidate is recorded, unless replacement is explicitly authorised;
    accepting it is how a stale run reverts a newer authorised upload.
    The refusal is `CandidateConflict` and it rolls back, leaving every
    column as it was. The rule is tested THROUGH stage_candidate: an
    earlier suite tested it through the legacy `upsert_document`, so a
    completely broken stage_candidate would have stayed green.
13. The publication service owns the destination: `publish_candidate`
    derives the disk target from the CANONICAL filename `stage_candidate`
    RETURNS -- which is why that seam returns three values, not two --
    and accepts no caller-supplied path under any spelling. The API and
    the CLI both go through it; neither writes the upload directory
    itself. The canonical name may DIFFER from the one offered (a
    re-cased upload lands on the existing row), and the bytes must
    follow the row, not the request.
14. A terminal `ERROR` or `PARTIAL` outcome RELEASES the lease in the
    same statement that records it: attempt_id, owner and expires_at
    cleared together with the verdict. Otherwise a finished-but-failed
    attempt would block the next one until its lease expired -- a run
    that is over must not hold anything.
11. Both publication crash windows recover idempotently (see above).
12. `attempt_expires_at` is compared against the DATABASE clock. A
    worker's own clock has no authority over a lease.


SEAMS the tests hold the implementation to:

    db.stage_candidate(conn, filename, file_type, content_sha256,
                       allow_replace)
        -> (document_id, candidate_id, canonical_filename)
    db.finalize_candidate_publication(conn, document_id, candidate_id)
                                             -> bool
    db.begin_attempt(conn, document_id, owner) -> IngestAttempt
    db.heartbeat_attempt(conn, attempt)      -> bool
    db.record_attempt_outcome(conn, attempt, status, note) -> bool
    publication.publish_candidate(conn, filename, file_type, body,
                                  allow_replace)
        -> (document_id, candidate_id, canonical_filename)
    ingest.ingest_attempt(snapshot, attempt) -> (outcome, note)
    ingest.cli_main(argv)                    -> int

WHERE EACH CLAIM IS CHECKED. Behaviour that lives in SQL -- the gate
arms, the lease, the promotion CAS, the crash windows -- is checked
against a REAL server in tests/test_pg_attempt_integration.py. The local
tests check what is local: the module split, the CLI's wiring and flags,
the error kinds Python reports, and the API's publish-gap answer. A local
test may NEVER assert database behaviour through a model of it; a model
would pass an empty implementation.
"""
from dataclasses import dataclass


class CandidateState:
    """The two states a candidate row may be in. Anything else is a bug."""

    STAGED = "staged"
    PUBLISHED = "published"


ALL_CANDIDATE_STATES = (CandidateState.STAGED, CandidateState.PUBLISHED)


class AttemptOutcome:
    """How an attempt ended, and WHO is allowed to write each ending.

    Only ERROR and PARTIAL are the worker's to record: they are its own
    verdict on its own run. DONE belongs to the PROMOTION, which writes
    it in the same transaction as the swap and the lease release (rule
    4) -- a worker able to write DONE on its own could mark an attempt
    successful without anything being promoted. SUPERSEDED belongs to
    the SYSTEM, written at takeover or fencing (rule 6) -- a worker able
    to write it could close its own attempt as though it had been
    displaced, or close another's.

    ``WORKER_WRITABLE`` is that boundary, and it is enforced BEFORE any
    statement runs: a probe found record_attempt_outcome accepting both
    SUPERSEDED and an entirely unknown value."""

    DONE = "done"
    ERROR = "error"
    PARTIAL = "partial"
    SUPERSEDED = "superseded"


ALL_ATTEMPT_OUTCOMES = (AttemptOutcome.DONE, AttemptOutcome.ERROR,
                        AttemptOutcome.PARTIAL, AttemptOutcome.SUPERSEDED)

WORKER_WRITABLE_OUTCOMES = (AttemptOutcome.ERROR, AttemptOutcome.PARTIAL)


class AttemptError(Exception):
    """Base for every refusal that protects a run's or a candidate's
    identity."""


class CandidateConflict(AttemptError, ValueError):
    """The candidate gate refused an offer. Also a ValueError so the
    existing callers that translate ValueError into HTTP 409 keep
    working while the type becomes precise."""


class CandidateNotPublished(AttemptError):
    """begin_attempt on a candidate whose bytes are not published yet.
    The API answers 409 and touches no document or attempt state."""


class CandidateSuperseded(AttemptError):
    """Finalisation refused: a newer candidate was staged while this
    publication was writing its bytes. Publishing anyway would mark the
    NEWER candidate published on the OLDER one's bytes, so the
    publication fails instead of returning a success its caller would
    believe."""


class AttemptAlreadyRunning(AttemptError):
    """begin_attempt against a LIVE lease held by another attempt."""


class AttemptFenced(AttemptError):
    """A newer, different candidate superseded this attempt. Raised at
    the first write the fenced attempt tries; it writes nothing."""


class AttemptLeaseLost(AttemptError):
    """This attempt's lease was taken over after expiry. The displaced
    worker may neither stamp nor promote."""


class AttemptRecordInconsistent(AttemptError):
    """The document row says the lease is this attempt's, but the attempt
    record does not agree -- it is missing, or already terminal. Not a
    lost lease and not a fence: nobody took anything from this run, the
    two records simply disagree, and a promotion that cannot close its
    own attempt has no business reporting success."""


class AttemptOutcomeNotWritable(AttemptError, ValueError):
    """A worker tried to record an outcome that is not its to write --
    DONE (the promotion's), SUPERSEDED (the system's), or a value the
    contract does not know at all. Refused before any statement runs."""


class ExitCode:
    """The CLI's frozen answers. A refusal is a RETURN VALUE, not a
    traceback: a caller cannot tell a crash from a policy decision, and
    a test suite that accepts "it raised something" accepts both."""

    OK = 0
    CANDIDATE_CONFLICT = 2
    CANDIDATE_NOT_PUBLISHED = 3
    ATTEMPT_UNAVAILABLE = 4
    ATTEMPT_LOST = 5


@dataclass(frozen=True)
class IngestAttempt:
    """The immutable identity of one indexing run. Fields only."""

    attempt_id: str
    document_id: str
    candidate_id: str
    candidate_sha: str
    observed_active: int
