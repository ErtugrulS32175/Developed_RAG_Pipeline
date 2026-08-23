"""Apply the versioned schema under a transaction-scoped advisory lock."""
from __future__ import annotations

import json

from pipeline.index import db


def main() -> int:
    conn = None
    try:
        conn = db.get_conn(service=True)
        db.init_schema(conn)
        if not db.schema_is_current(conn):
            print(json.dumps({"migration_version": 1, "status": "failed"}))
            return 1
        version, digest = db.expected_schema_state()
        print(json.dumps({
            "migration_version": 1,
            "status": "current",
            "schema_version": version,
            "schema_sha256": digest,
        }, sort_keys=True))
        return 0
    except Exception:
        # Connection exceptions commonly include host/user coordinates. The
        # operational result is closed and intentionally omits their prose.
        print(json.dumps({"migration_version": 1, "status": "failed"},
                         sort_keys=True))
        return 1
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
