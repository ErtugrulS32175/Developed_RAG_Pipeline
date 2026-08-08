#!/usr/bin/env bash
# The ONE official command for the P0 gate.
#
# The database half of the attempt contract may not be skipped: an
# opt-in environment variable makes "we did not check" look like "we
# passed". This script creates a DISPOSABLE cluster, runs the gate with
# RAGTEST_P0_GATE=1 (missing DSN becomes a failure, not a skip), and
# removes the cluster afterwards. It needs no credentials, no Docker and
# no network, and it never touches an existing server.
#
#   scripts/p0_gate.sh              # create, run, destroy
#   PGBIN=/usr/lib/postgresql/16/bin scripts/p0_gate.sh
#
# An existing throwaway server can be used instead by exporting
# RAGTEST_PG_TEST_DSN before calling; nothing is then created or removed.
#
# DELETION SAFETY. The working directory is created by `mktemp -d` and
# CANNOT be chosen by the caller: an earlier version honoured
# RAGTEST_PG_WORKDIR and ran `rm -rf` over it from an EXIT trap that
# fired even when initdb had failed -- one stale environment variable
# away from deleting an unrelated directory. Now the path comes from
# mktemp, a marker file records that this script owns it, and the
# removal happens only when the marker is present in a path that is
# still under the temp root.
#
# SCOPE SAFETY. PYTEST_ADDOPTS is cleared, the project's own `addopts` is
# overridden with `-o addopts=`, and the run must both COLLECT the pinned
# number of cases and RUN every one of them. Checking only
# collected == ran let a deleted test or a new `--ignore` shrink the gate
# to 37/37 and still call it a pass.
set -euo pipefail

# A HARD CONSTANT, deliberately not readable from the environment: an
# env-overridable pin is not a pin. Whoever deletes or adds a case edits
# this line in the same commit, where a reviewer sees it.
EXPECTED_CASES=81

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${RAGTEST_PG_PORT:-55433}"
MARKER_NAME=".ragtest-p0-gate-sahibi"
WORKDIR=""
PG=""
GATE_FILES=(tests/test_pg_attempt_integration.py
            tests/test_ingest_attempt_binding.py
            tests/test_publication_service.py)

find_pgbin() {
    if [ -n "${PGBIN:-}" ]; then echo "$PGBIN"; return; fi
    if command -v initdb >/dev/null 2>&1; then
        dirname "$(command -v initdb)"; return
    fi
    for candidate in /c/Program\ Files/PostgreSQL/*/bin \
                     /usr/lib/postgresql/*/bin /usr/local/pgsql/bin; do
        if [ -x "$candidate/initdb" ] || [ -x "$candidate/initdb.exe" ]; then
            echo "$candidate"; return
        fi
    done
    echo "initdb bulunamadi: PGBIN ver ya da RAGTEST_PG_TEST_DSN ile hazir" \
         "bir atilabilir sunucu goster" >&2
    exit 2
}

cleanup() {
    # Three independent conditions, ALL required before anything is
    # removed: we created the path, the marker we wrote is still there,
    # and the resolved path still lives under the temp root.
    [ -n "$WORKDIR" ] || return 0
    [ -f "$WORKDIR/$MARKER_NAME" ] || return 0
    local resolved temp_root
    resolved="$(cd "$WORKDIR" 2>/dev/null && pwd -P)" || return 0
    temp_root="$(cd "${TMPDIR:-/tmp}" 2>/dev/null && pwd -P)" || return 0
    case "$resolved" in
        "$temp_root"/*) ;;
        *) echo "[P0] guvenlik: $resolved gecici kok altinda degil," \
                "silinmedi" >&2; return 0 ;;
    esac
    if [ -n "$PG" ] && [ -d "$resolved/pgdata" ]; then
        "$PG/pg_ctl" -D "$resolved/pgdata" -m fast stop >/dev/null 2>&1 || true
    fi
    rm -rf "$resolved"
}
trap cleanup EXIT

# The workdir is created even when a ready server is supplied, because the
# CLI half of the gate PUBLISHES: it puts real bytes in the upload
# directory through the shared service. Left at its default that is
# ./data/uploads -- the operator's own document directory -- so the gate
# points UPLOAD_DIR at its own disposable tree and takes it away again.
WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/ragtest-p0-gate-XXXXXX")"
: > "$WORKDIR/$MARKER_NAME"
export UPLOAD_DIR="$WORKDIR/uploads"
mkdir -p "$UPLOAD_DIR"

if [ -z "${RAGTEST_PG_TEST_DSN:-}" ]; then
    PG="$(find_pgbin)"
    echo "[P0] atilabilir kume kuruluyor: $WORKDIR (port $PORT)"
    "$PG/initdb" -D "$WORKDIR/pgdata" -U postgres -A trust -E UTF8 \
        --locale=C >/dev/null
    "$PG/pg_ctl" -D "$WORKDIR/pgdata" -l "$WORKDIR/pg.log" \
        -o "-p $PORT -c listen_addresses=127.0.0.1" -w start >/dev/null
    "$PG/psql" -h 127.0.0.1 -p "$PORT" -U postgres -d postgres \
        -c "CREATE DATABASE p0gate" >/dev/null
    export RAGTEST_PG_TEST_DSN="postgresql://postgres@127.0.0.1:$PORT/p0gate"
else
    echo "[P0] hazir sunucu kullaniliyor (kurulmadi, silinmeyecek)"
fi

export RAGTEST_P0_GATE=1
unset PYTEST_ADDOPTS PYTEST_PLUGINS || true
cd "$REPO_ROOT"

collected="$(python -m pytest -o addopts= "${GATE_FILES[@]}" \
             --collect-only -q 2>/dev/null | grep -c '::' || true)"
echo "[P0] toplanan vaka: $collected (beklenen $EXPECTED_CASES)"

set +e
output="$(python -m pytest -o addopts= "${GATE_FILES[@]}" -q 2>&1)"
status=$?
set -e
echo "$output" | tail -25

summary="$(echo "$output" | tail -5 | grep -E '[0-9]+ (passed|failed)' || true)"
ran=$(echo "$summary" | grep -oE '[0-9]+ (passed|failed)' \
      | awk '{s+=$1} END {print s+0}')
echo "[P0] calisan vaka: $ran"
if echo "$summary" | grep -qE '[0-9]+ (deselected|skipped)'; then
    echo "[P0] KAPI GECERSIZ: deselected/skipped var -- kapi daraltilmis" >&2
    exit 3
fi
if [ "$collected" -ne "$EXPECTED_CASES" ]; then
    echo "[P0] KAPI GECERSIZ: toplanan $collected, pinlenen" \
         "$EXPECTED_CASES -- vaka silinmis ya da eklenmis olabilir;" \
         "degisiklik kasitliysa EXPECTED_CASES ayni commit'te guncellenir" >&2
    exit 3
fi
if [ "$ran" -ne "$collected" ]; then
    echo "[P0] KAPI GECERSIZ: toplanan $collected, calisan $ran" >&2
    exit 3
fi
exit "$status"
