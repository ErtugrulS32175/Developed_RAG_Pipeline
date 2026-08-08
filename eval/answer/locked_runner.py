"""The contract-gated LOCKED STRUCTURAL SCREENING runner. Fail-closed.

NAMED FOR WHAT IT IS. This module replays mutation screening over locked
clusters under a verified contract chain; it is NOT the final blind gate.
The final gate additionally requires, per the frozen contract: human labels
on mutants and controls, the paired-control population, observed-control
false-review measurement, overall/per-set scorecards, and a human approval
step on construct validity. Those consume adjudication data that does not
exist yet and belong to a separate module (`locked_scorecard`, unbuilt);
pretending this runner is that gate was the auditor's sharpest finding
about it, and the rename is the fix that cannot regress.

Its obligations, all normative in the contract/addendum and all tested:

  * It verifies the contract chain against OWNER-STATED hashes BEFORE it
    opens a single case file. An incomplete chain, a wrong effective
    version, a wrong base hash, a missing/extra/tampered addendum -- each
    stops the run with nothing loaded. ``contract_complete=False`` loading
    nothing is the addendum's mandatory first regression test.
  * It measures LOCKED clusters only, through the same mutation machinery
    as the development harness -- the membership check inverts at the gate,
    so neither population can be fed to the other's path.
  * Its report carries COUNTS, OPAQUE IDS and BOUNDS. No question, answer,
    key or passage text can appear in it; the writer proves that against
    the loaded cases before a byte reaches disk, because the locked split
    has been burned once by prose and will not be burned by a report.
  * Gate arithmetic counts CLUSTERS. The one-sided 95% upper bound on the
    miss rate is exact binomial (0/30 -> 9.50%); candidate-basis classes
    (the relation family) get counts but NO structural-recall bound --
    screening is not proof, and the human construct-validity audit stays
    the arbiter, exactly as in the development report.

What this module deliberately cannot see: whether the files it is given
belong to the FRESH holdout or to the burned split. That is a human-process
fact governed by the contract owner; the runner's contribution is refusing
to run outside a verified chain and refusing to leak what it read.

OPERATIONAL NOTE for the real run: the shared eval directory is NOT a
valid --question-dir here -- it carries development and holdout sets side
by side, and this runner requires its question directory to be EXACTLY
the measured population (an unanswered set in the directory refuses the
run). The owner prepares a DEDICATED directory holding precisely the
locked sets of the run -- that directory, together with the declared
fingerprints and case count, IS the owner-pinned set manifest.
"""
import argparse
import math
import re
from pathlib import Path

from eval.answer.adversarial_feasibility import (
    CLASS_ORDER,
    TARGET_LOCKED_CLUSTERS,
    checked_output_path,
    is_locked,
    load_cases,
    repository_head,
    write_report,
)
from eval.answer.adversarial_mutate import (
    CANDIDATE_ROLE,
    CONFIRMED_ROLE,
    EXPECTED_DIAGNOSTIC,
    FROZEN_POLICY_MATRIX,
    build_mutants,
    population_exclusions,
    population_role,
    replay,
    verify_contract_chain,
)

PROTOCOL_VERSION = "adversarial_locked_screening_v1"
CONFIDENCE = 0.95

# The v2.1 addendum's integrity gate, items 2-4, pinned VERBATIM: the
# effective and base protocol names and the base contract hash are fixed by
# the frozen text, so an owner statement that contradicts them is refused
# even before the disk is consulted. The hash of a gitignored document is
# not the document; carrying it here is what makes the pin checkable.
REQUIRED_EFFECTIVE_VERSION = "adversarial_holdout_v2.1"
REQUIRED_BASE_VERSION = "adversarial_holdout_v2"
REQUIRED_BASE_SHA256 = (
    "d079183e19e6b524b3f6f50e08b95d9eec5baf28544c9c5c1da0b3ee96b60d4a")


def locked_cases(cases) -> tuple:
    """Filter by RECOMPUTED identity; refuse records that lie about it."""
    kept = []
    for case in cases:
        locked = is_locked(case.stable_id)  # rejects malformed ids outright
        if locked != case.locked:
            raise ValueError("case split flag disagrees with its stable id")
        if locked:
            kept.append(case)
    return tuple(kept)


def one_sided_upper_bound(misses: int, n: int,
                          confidence: float = CONFIDENCE) -> float:
    """Exact one-sided binomial upper bound on the miss rate.

    The smallest p with P(X <= misses | n, p) <= 1 - confidence, found by
    bisection on the exact CDF -- no scipy, no approximation, and the
    contract's own anchor number reproduces: 0 misses in 30 clusters gives
    0.0950. Bisection depth 60 pins the answer far below reporting
    precision and is deterministic."""
    if type(n) is not int or type(misses) is not int:
        raise ValueError("bounds take integer counts")
    if n <= 0 or misses < 0 or misses > n:
        raise ValueError("bounds need 0 <= misses <= n and n > 0")
    if misses == n:
        return 1.0
    alpha = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(60):
        mid = (low + high) / 2
        cdf = sum(
            math.comb(n, k) * mid ** k * (1 - mid) ** (n - k)
            for k in range(misses + 1)
        )
        if cdf > alpha:
            low = mid
        else:
            high = mid
    return round(high, 6)


def _gate_summary(classes: dict) -> dict:
    """Per-class, per-policy cluster counts with bounds where they are owed.

    A bound appears ONLY for confirmed-structural classes: publishing an
    upper bound over a candidate population would launder "pending human
    confirmation" into arithmetic.

    WHICH counter counts is the contract's call, not this function's: a
    class with a preregistered expected diagnostic is caught only when THAT
    code fired -- "held for some reason" is not detection, and an earlier
    version that counted any flag would have reported 30 wrong-code catches
    as zero misses (upper bound 9.5%) where the true diagnostic recall was
    0/30. Classes whose contract entry is None keep the safety-catch
    basis; both numbers are reported so the gap itself is visible."""
    summary = {}
    for name in CLASS_ORDER:
        body = classes[name]
        eligible = body["population_role"] != CANDIDATE_ROLE
        expected = EXPECTED_DIAGNOSTIC[name]
        per_policy = {}
        for policy, result in body["policies"].items():
            counts = result["counts"]
            n = counts.get("n", 0)
            any_flag = counts.get("caught", 0)
            if expected is None:
                caught = any_flag
            else:
                caught = counts.get("expected_diagnostic", 0)
            # The cluster floor is the CONTRACT's, not this function's: a
            # bound over 29 clusters is not a slightly weaker bound, it is
            # a number the frozen text says may not exist. An auditor probe
            # showed 29/29 with zero misses sailing through as
            # gate-eligible; the floor field and the withheld bound are
            # that hole closing.
            floor_met = n >= TARGET_LOCKED_CLUSTERS
            cell = {
                "clusters": n,
                "caught": caught,
                "caught_any_flag": any_flag,
                "recall_basis": ("safety_catch" if expected is None
                                 else "expected_diagnostic"),
                "missed": n - caught,
                "meets_cluster_floor": floor_met,
            }
            if eligible and floor_met:
                cell["miss_rate_upper_95"] = one_sided_upper_bound(
                    n - caught, n)
            per_policy[policy] = cell
        summary[name] = {
            "population_role": body["population_role"],
            "structural_recall_eligible": eligible,
            "policies": per_policy,
        }
    return summary


# --- what a locked report is ALLOWED to say ---------------------------------
# The auditor's demonstration: a blacklist of known fragments let a short
# key, a source page and a set name through, and case-folded variants past
# it. A blacklist enumerates what we thought of; the SCHEMA below enumerates
# what is permitted, and everything else is refused -- unknown keys, free
# strings, wrong types. Round 17 closed what the first schema still let by:
# ABBREVIATED hashes (12 hex chars answered for a sha256), ABSENT fields
# inside cells ({} was a valid cell), a bound whose VALUE nobody recomputed,
# policy cells missing from the frozen matrix, and two sections free to
# contradict each other. Everything the runner can derive is now REQUIRED
# to match the derivation exactly. The fragment check stays underneath as a
# second layer, folded on both sides so a re-cased copy cannot slip it.
_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD_HEX = re.compile(r"^[0-9a-f]{40}$")
_VERSION = re.compile(r"^[a-z0-9_.-]{1,64}$")
_ADDENDUM_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,80}\.md$")
_FLAG_CODE = re.compile(r"^[a-z_]{3,40}$")
_COUNT_KEYS = {"n", "caught", "expected_diagnostic", "published",
               "control_review", "control_published"}
_CELL_KEYS = {"clusters", "caught", "caught_any_flag", "recall_basis",
              "missed", "miss_rate_upper_95", "meets_cluster_floor"}
# every cell field except the bound is unconditional; the bound's presence
# is itself DERIVED (eligible AND at the floor) and checked as such
_CELL_REQUIRED = _CELL_KEYS - {"miss_rate_upper_95"}


def _refuse(path, why):
    raise ValueError(f"rapor semasi disinda ({path}): {why}")


def _require_int(value, path):
    """Counts are non-negative integers -- a negative count is not a small
    anomaly, it is arithmetic that cannot have happened."""
    if type(value) is not int or value < 0:
        _refuse(path, "negatif olmayan tam sayi bekleniyor")


def _check_policy_cell(cell, path, eligible, expected_basis):
    if set(cell) - _CELL_KEYS:
        _refuse(path, "bilinmeyen alan")
    if _CELL_REQUIRED - set(cell):
        _refuse(path, "zorunlu hucre alani eksik")
    for key, value in cell.items():
        if key == "recall_basis":
            if value not in ("safety_catch", "expected_diagnostic"):
                _refuse(path, "bilinmeyen recall tabani")
        elif key == "miss_rate_upper_95":
            # NaN != NaN: the comparison below refuses it along with
            # anything outside the unit interval
            if type(value) is not float or not (0.0 <= value <= 1.0):
                _refuse(path, "sinir [0,1] araliginda float olmali")
        elif key == "meets_cluster_floor":
            if type(value) is not bool:
                _refuse(path, "bool bekleniyor")
        else:
            _require_int(value, f"{path}.{key}")
    # arithmetic and every derivable field must MATCH the derivation: a
    # cell whose numbers cannot all be true at once, a floor flag that
    # contradicts its own cluster count, a bound present where the
    # contract forbids one (or absent where it owes one), or a bound whose
    # value is not the exact recomputed binomial -- each is a corrupted or
    # hand-edited report
    n = cell["clusters"]
    caught = cell["caught"]
    any_flag = cell["caught_any_flag"]
    missed = cell["missed"]
    if caught > n or any_flag > n or missed != n - caught:
        _refuse(path, "hucre aritmetigi tutarsiz")
    if caught > any_flag:
        _refuse(path, "tani sayaci genel bayragi asamaz")
    if cell["recall_basis"] != expected_basis:
        _refuse(path, "recall tabani sinif sozlesmesiyle celisiyor")
    if expected_basis == "safety_catch" and caught != any_flag:
        _refuse(path, "safety-catch tabaninda iki sayac ayni olmali")
    floor = n >= TARGET_LOCKED_CLUSTERS
    if cell["meets_cluster_floor"] is not floor:
        _refuse(path, "taban bayragi kume sayisiyla celisiyor")
    if (eligible and floor) != ("miss_rate_upper_95" in cell):
        _refuse(path, "sinir tam olmasi gereken yerde olmali, baska yerde "
                      "olmamali")
    if "miss_rate_upper_95" in cell:
        if cell["miss_rate_upper_95"] != one_sided_upper_bound(missed, n):
            _refuse(path, "sinir yeniden hesaplananla uyusmuyor")


def _assert_report_schema(report: dict) -> None:
    """Every field of the report against an allowlist; unknown means refused,
    and everything derivable is re-derived and compared."""
    from eval.answer.adversarial_mutate import DETECTORS, population_role

    top = {"protocol_version", "contract", "repository_head",
           "evaluation_layer", "population", "locked_cases",
           "development_cases_excluded", "source", "gate_summary", "classes"}
    if set(report) != top:
        _refuse("rapor", "beklenmeyen ust alan kumesi")
    if report["protocol_version"] != PROTOCOL_VERSION:
        _refuse("protocol_version", "bilinmeyen surum")
    if report["evaluation_layer"] != "generator_guard":
        _refuse("evaluation_layer", "bilinmeyen katman")
    if report["population"] != "locked_clusters_only":
        _refuse("population", "bilinmeyen populasyon")
    # a locked report ALWAYS names its code: the runner refuses to run
    # without a verified HEAD, so a headless report cannot be one of ours
    if not _GIT_HEAD_HEX.match(str(report["repository_head"])):
        _refuse("repository_head", "40 karakter commit hex'i bekleniyor")
    _require_int(report["locked_cases"], "locked_cases")
    _require_int(report["development_cases_excluded"], "dev_excluded")

    contract = report["contract"]
    if set(contract) != {"contract_version", "contract_sha256",
                         "effective_protocol_version", "contract_complete",
                         "addenda"}:
        _refuse("contract", "beklenmeyen alan kumesi")
    if contract["contract_complete"] is not True:
        _refuse("contract", "kilitli rapor tam zincir ister")
    for field in ("contract_version", "effective_protocol_version"):
        if not _VERSION.match(str(contract[field])):
            _refuse(field, "surum deseni disi")
    if not _SHA256_HEX.match(str(contract["contract_sha256"])):
        _refuse("contract_sha256", "tam 64 karakter sha256 hex'i bekleniyor")
    for name, entry in contract["addenda"].items():
        if not _ADDENDUM_NAME.match(name):
            _refuse("addenda", "ad deseni disi")
        if not isinstance(entry, dict) or set(entry) != {"sha256", "protocol"}:
            _refuse("addenda", "sha256+protocol bekleniyor")
        if not _SHA256_HEX.match(str(entry["sha256"])):
            _refuse("addenda", "tam 64 karakter sha256 hex'i bekleniyor")
        if entry["protocol"] is not None and not _VERSION.match(
                str(entry["protocol"])):
            _refuse("addenda", "protokol deseni disi")

    # the source statement is an EXACT set: removable counters and
    # smuggled extra "*_fingerprint" keys both passed the loose version
    source_keys = {"result_files", "question_files",
                   "raw_result_files_fingerprint",
                   "question_files_fingerprint",
                   "eligibility_input_fingerprint",
                   "split_manifest_fingerprint"}
    if set(report["source"]) != source_keys:
        _refuse("source", "alan kumesi birebir degil; eksik veya fazla alan")
    for key, value in report["source"].items():
        if key in ("result_files", "question_files"):
            _require_int(value, f"source.{key}")
        else:
            if not _SHA256_HEX.match(str(value)):
                _refuse(f"source.{key}", "tam 64 karakter sha256 hex'i "
                                         "bekleniyor")

    frozen = set(FROZEN_POLICY_MATRIX)
    for section in ("gate_summary", "classes"):
        if set(report[section]) != set(CLASS_ORDER):
            _refuse(section, "her sinif zorunlu; eksik veya fazla sinif var")

    for name, body in report["gate_summary"].items():
        path = f"gate_summary.{name}"
        if set(body) != {"population_role", "structural_recall_eligible",
                         "policies"}:
            _refuse(path, "govde alan kumesi birebir degil")
        expected_role = population_role(name)
        if body["population_role"] != expected_role:
            _refuse(path, "rol sinif sozlesmesiyle celisiyor")
        eligible = expected_role != CANDIDATE_ROLE
        if body["structural_recall_eligible"] is not eligible:
            _refuse(path, "uygunluk rolden turetilenle celisiyor")
        if set(body["policies"]) != frozen:
            _refuse(path, "politika matrisi donuk: eksik veya fazla hucre")
        expected_basis = ("safety_catch" if EXPECTED_DIAGNOSTIC[name] is None
                          else "expected_diagnostic")
        for policy, cell in body["policies"].items():
            if not _FLAG_CODE.match(policy):
                _refuse(path, "politika adi deseni disi")
            _check_policy_cell(cell, f"{path}.{policy}", eligible,
                               expected_basis)

    for name, body in report["classes"].items():
        path = f"classes.{name}"
        required = {"mutants", "basis", "population_role",
                    "structural_recall_eligible", "expected_diagnostic",
                    "policies"}
        allowed = required | {"excluded_by_reason"}
        if set(body) - allowed or required - set(body):
            _refuse(path, "govde alan kumesi birebir degil")
        _require_int(body["mutants"], f"{path}.mutants")
        if body["basis"] != DETECTORS[name]["basis"]:
            _refuse(path, "taban sinif sozlesmesiyle celisiyor")
        if body["population_role"] != population_role(name):
            _refuse(path, "rol sinif sozlesmesiyle celisiyor")
        if body["structural_recall_eligible"] is not (
                population_role(name) != CANDIDATE_ROLE):
            _refuse(path, "uygunluk rolden turetilenle celisiyor")
        if body["expected_diagnostic"] != EXPECTED_DIAGNOSTIC[name]:
            _refuse(path, "tani kodu sinif sozlesmesiyle celisiyor")
        if set(body["policies"]) != frozen:
            _refuse(path, "politika matrisi donuk: eksik veya fazla hucre")
        for policy, result in body["policies"].items():
            if not _FLAG_CODE.match(policy):
                _refuse(path, "politika adi deseni disi")
            if set(result) != {"counts", "flags"}:
                _refuse(f"{path}.{policy}", "alan kumesi birebir degil")
            counts = result["counts"]
            flags = result["flags"]
            for key, value in counts.items():
                if key not in _COUNT_KEYS:
                    _refuse(f"{path}.{policy}", "bilinmeyen sayac")
                _require_int(value, f"{path}.{policy}.{key}")
            for code, count in flags.items():
                if not _FLAG_CODE.match(code):
                    _refuse(f"{path}.{policy}", "kod deseni disi")
                _require_int(count, f"{path}.{policy}.flags")
            if counts:
                # every counter the replay derives is RE-DERIVED here: a
                # probe hand-edited mutant/published counts and a deleted
                # flag table and the loose checks accepted all of them
                if set(counts) != _COUNT_KEYS:
                    _refuse(f"{path}.{policy}", "sayac kumesi birebir degil")
                n = counts["n"]
                if n != body["mutants"]:
                    _refuse(f"{path}.{policy}",
                            "n mutant sayisiyla celisiyor")
                if counts["published"] != n - counts["caught"]:
                    _refuse(f"{path}.{policy}",
                            "published sayaci turetilemiyor")
                if counts["expected_diagnostic"] > counts["caught"]:
                    _refuse(f"{path}.{policy}",
                            "tani sayaci genel sayaci asamaz")
                if counts["control_review"] + counts["control_published"] != n:
                    _refuse(f"{path}.{policy}",
                            "kontrol sayaclari toplami n degil")
                if (counts["caught"] > 0) != bool(flags):
                    _refuse(f"{path}.{policy}",
                            "bayrak dagilimi sayaclarla celisiyor")
                if flags and (sum(flags.values()) < counts["caught"]
                              or any(v > n for v in flags.values())):
                    _refuse(f"{path}.{policy}",
                            "bayrak dagilimi aritmetigi tutarsiz")
            else:
                if body["mutants"] != 0 or flags:
                    _refuse(f"{path}.{policy}",
                            "bos sayac yalniz sifir mutantla tutarli")
        for reason, count in body.get("excluded_by_reason", {}).items():
            if not _FLAG_CODE.match(reason):
                _refuse(path, "haric tutma deseni disi")
            _require_int(count, f"{path}.excluded")

    # the two sections describe ONE measurement: every gate cell must be
    # re-derivable from its class cell, or the report contradicts itself
    for name in CLASS_ORDER:
        class_policies = report["classes"][name]["policies"]
        for policy, cell in report["gate_summary"][name]["policies"].items():
            counts = class_policies[policy]["counts"]
            expected_caught = (counts.get("caught", 0)
                               if EXPECTED_DIAGNOSTIC[name] is None
                               else counts.get("expected_diagnostic", 0))
            if (cell["clusters"] != counts.get("n", 0)
                    or cell["caught_any_flag"] != counts.get("caught", 0)
                    or cell["caught"] != expected_caught):
                _refuse(f"gate_summary.{name}.{policy}",
                        "bolumler arasi celiski: hucre sinif sayaclarindan "
                        "turetilemiyor")


def _fold(text: str) -> str:
    import unicodedata

    s = str(text).lower().replace("ı", "i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def _assert_content_free(report: dict, cases) -> None:
    """Layer two under the schema: no folded fragment of any loaded case in
    the serialized report. The schema refuses unknown FIELDS; this refuses
    known fields whose VALUES echo case text -- and both sides are folded,
    so a case-changed or de-accented copy is still caught.

    SET NAMES and PASSAGE SOURCES are content too, at ANY length: the
    sliding windows start at 12 characters, and audit probes rode first a
    short set name and then a TWO-character one into the report through
    value slots the schema patterns happened to admit. Names are matched
    as whole tokens from length 2 (underscore counts as a joiner, so
    identifier-embedded uses of a word are not the name); purely numeric
    names are outside this layer because counts are numbers."""
    import json

    blob = _fold(json.dumps(report, ensure_ascii=False))
    for case in cases:
        fragments = [case.question, case.answer, case.key]
        fragments.extend(p.text for p in case.context.passages)
        fragments.extend(c.quote for c in case.evidence)
        for fragment in fragments:
            for piece in _long_pieces(_fold(fragment)):
                if piece in blob:
                    raise ValueError(
                        "rapor kilitli icerik tasiyor; yazma reddedildi")
    tokens = {case.set_name for case in cases}
    tokens.update(passage.source for case in cases
                  for passage in case.context.passages
                  if getattr(passage, "source", None))
    for raw in tokens:
        token = _fold(raw)
        if len(token) < 2 or token.isdigit():
            continue
        if re.search(r"(?<![a-z0-9_])" + re.escape(token) + r"(?![a-z0-9_])",
                     blob):
            raise ValueError(
                "rapor kilitli icerik tasiyor; yazma reddedildi")


def _long_pieces(text: str):
    """Sliding fragments long enough to be content, short enough to catch
    partial copies. Window 16, step 8: any report substring of the case
    text 23 characters or longer necessarily contains one aligned window,
    so a report quoting even half a sentence fails. Shorter overlaps are
    below the length at which text is identifying."""
    text = " ".join(str(text).split())
    if len(text) < 12:
        return
    if len(text) < 16:
        yield text
        return
    for start in range(0, len(text) - 15, 8):
        yield text[start:start + 16]


def _require_clean_tree() -> None:
    """The running-code fingerprint is HEAD -- which only names the code
    when the tree is clean. A screening run from a dirty tree would report
    a commit hash for code nobody can reproduce.

    Anchored to THIS module's repository, never the process's working
    directory: a probe called the runner from a different, clean repo and
    the check blessed that repo while the dirty one actually ran."""
    import subprocess

    from eval.answer.adversarial_feasibility import REPO_ROOT

    status = subprocess.run(["git", "status", "--porcelain"],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError(
            "calisma agaci temiz degil; kilitli tarama commit'lenmis "
            "koddan kosulur")


REQUIRED_SOURCE_KEYS = ("raw_result_files_fingerprint",
                        "question_files_fingerprint")


def run_locked(run_dir: Path, question_dir: Path, expected_version: str,
               expected_contract_sha256: str, expected_addenda: dict,
               expected_source_fingerprints: dict | None = None,
               expected_locked_cases: int | None = None,
               expected_head: str | None = None) -> dict:
    """Gate first, load second -- the order is the security property.

    NOTHING here is optional at screening time, and nothing is skippable:
    an earlier signature carried a ``require_clean_tree`` switch for tests,
    which was a public bypass wearing a keyword's clothes -- any caller
    could flip it and run a "locked screening" from an unreproducible
    tree. The clean-tree check is unconditional now; tests that need a
    synthetic tree patch the checker, which no production entry point does.

    The owner's manifest states the contract chain, BOTH source
    fingerprints, the locked case count and the code snapshot; the run
    refuses on any missing or mismatched statement. Optional verification
    was the auditor's finding: with fingerprints omitted, a source file
    quietly dropped from 96 to 95 cases and the run accepted it -- an
    unverifiable number wearing a verified number's clothes."""
    if expected_version != REQUIRED_EFFECTIVE_VERSION:
        raise ValueError("beyan edilen etkin surum addendum pinine uymuyor")
    if expected_contract_sha256 != REQUIRED_BASE_SHA256:
        raise ValueError("beyan edilen taban hash addendum pinine uymuyor")
    stated = expected_source_fingerprints or {}
    for key in REQUIRED_SOURCE_KEYS:
        if not stated.get(key):
            raise ValueError(f"kaynak parmak izi beyani zorunlu: {key}")
    if expected_locked_cases is None or expected_locked_cases <= 0:
        raise ValueError("beklenen kilitli vaka sayisi beyani zorunlu")
    if not expected_head:
        raise ValueError("beklenen kod anlik goruntusu (HEAD) beyani zorunlu")
    gate = verify_contract_chain(
        expected_version, expected_contract_sha256, expected_addenda)
    if gate.contract_version != REQUIRED_BASE_VERSION:
        raise ValueError("taban protokol adi addendum pinine uymuyor")
    _require_clean_tree()
    # ONE snapshot of HEAD, taken here and reused for the report: a second
    # read let a probe move HEAD between verification and writing, and the
    # report carried a commit nobody verified
    head = repository_head()
    if head != expected_head:
        raise ValueError("calisan kod beyan edilen HEAD degil")
    cases, source_metadata = load_cases(run_dir, question_dir)
    for key, value in stated.items():
        if source_metadata.get(key) != value:
            raise ValueError(
                f"kaynak parmak izi beyanla uyusmuyor: {key}")
    # The question DIRECTORY must be exactly the measured population: a
    # question set with no result file was silently not-loaded, so its
    # absence never reached any fingerprint. load_cases guarantees every
    # loaded result has its question file from this directory, so a bare
    # count comparison IS set equality. The dev harness tolerates orphan
    # question sets (the shared eval directory carries unanswered holdout
    # sets by design); a locked screening tolerates nothing it did not
    # measure.
    on_disk = len(list(Path(question_dir).glob("*.json")))
    if on_disk != source_metadata.get("question_files"):
        raise ValueError(
            f"soru dizini olculen populasyonla birebir degil: dizinde "
            f"{on_disk} dosya, olculen {source_metadata.get('question_files')}"
            f" -- sonucsuz bir soru seti sessizce dusmus olabilir")
    population = locked_cases(cases)
    if not population:
        raise ValueError("kilitli vaka yok; olculecek populasyon bos")
    if len(population) != expected_locked_cases:
        raise ValueError(
            f"kilitli vaka sayisi beyanla uyusmuyor: {len(population)} != "
            f"{expected_locked_cases}; bir kaynak dosya sessizce degismis "
            f"olabilir")
    produced = build_mutants(population, gate)
    classes = replay(produced, frozen_matrix=True)
    for name, reasons in population_exclusions(population).items():
        classes[name]["excluded_by_reason"] = reasons
    report = {
        "protocol_version": PROTOCOL_VERSION,
        # the GATE's snapshot, never a fresh disk read: a contract file
        # swapped between verification and this line must not be able to
        # dress the report in a version the owner never approved
        "contract": {
            "contract_version": gate.contract_version,
            "contract_sha256": gate.contract_sha256,
            "effective_protocol_version": gate.effective_protocol_version,
            "contract_complete": True,
            "addenda": {
                name: {"sha256": sha, "protocol": protocol}
                for name, sha, protocol in gate.addenda
            },
        },
        "repository_head": head,
        "evaluation_layer": "generator_guard",
        "population": "locked_clusters_only",
        "locked_cases": len(population),
        "development_cases_excluded": len(cases) - len(population),
        "source": source_metadata,
        "gate_summary": _gate_summary(classes),
        "classes": classes,
    }
    _assert_report_schema(report)
    _assert_content_free(report, cases)
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Kilitli YAPISAL TARAMA: zincir dogrulanmadan hicbir sey "
                    "yuklenmez; nihai kor kapi degildir (insan etiketi ve "
                    "kontrol populasyonlari ayri modulde)")
    parser.add_argument("run_dir")
    parser.add_argument("--question-dir", required=True,
                        help="YALNIZ bu kosunun kilitli setlerini iceren "
                             "ADANMIS dizin (paylasilan eval dizini degil): "
                             "dizindeki her set olculmus olmali, yoksa "
                             "kosu reddeder")
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--expected-contract-sha", required=True)
    parser.add_argument("--expected-addendum", action="append", default=[],
                        metavar="AD=SHA256",
                        help="beklenen her addendum; sahibin beyani, diskten "
                             "okunan degil")
    parser.add_argument("--expected-result-fingerprint", required=True,
                        help="raw_result_files_fingerprint beyani (zorunlu)")
    parser.add_argument("--expected-question-fingerprint", required=True,
                        help="question_files_fingerprint beyani (zorunlu)")
    parser.add_argument("--expected-locked-cases", type=int, required=True,
                        help="kilitli vaka sayisi beyani (zorunlu); eksik "
                             "bir kaynak dosya sessiz 95'i yakalayan sayi")
    parser.add_argument("--expected-head", required=True,
                        help="kosulacak kodun commit'i (zorunlu)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    expected_addenda = {}
    for item in args.expected_addendum:
        name, _, sha = item.partition("=")
        if not name or not sha:
            raise SystemExit("addendum beyani AD=SHA256 bicimindedir")
        expected_addenda[name] = sha
    expected_sources = {
        "raw_result_files_fingerprint": args.expected_result_fingerprint,
        "question_files_fingerprint": args.expected_question_fingerprint,
    }

    report = run_locked(Path(args.run_dir), Path(args.question_dir),
                        args.expected_version, args.expected_contract_sha,
                        expected_addenda,
                        expected_source_fingerprints=expected_sources,
                        expected_locked_cases=args.expected_locked_cases,
                        expected_head=args.expected_head)
    path = checked_output_path(Path(args.output))
    write_report(report, path)

    print(f"kilitli yapisal tarama: {report['locked_cases']} vaka · "
          f"zincir {report['contract']['effective_protocol_version']}")
    for name, body in report["gate_summary"].items():
        role = "" if body["structural_recall_eligible"] else "  [ADAY: sinir yok]"
        cells = "  ".join(
            f"{policy}:{cell['caught']}/{cell['clusters']}"
            + (f" (ust {cell['miss_rate_upper_95']:.2%})"
               if "miss_rate_upper_95" in cell
               else ("" if cell["meets_cluster_floor"] else " [taban<30]"))
            for policy, cell in body["policies"].items()
        )
        print(f"  {name:<28}{cells}{role}")
    print(f"yazildi: {path}")
    print("NOT: bu yapisal taramadir; nihai kapi insan etiketli scorecard "
          "modulunu bekler.")


if __name__ == "__main__":
    main()
