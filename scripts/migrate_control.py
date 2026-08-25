"""Apply the content-free control-plane schema with closed output."""
import json

from pipeline.control import db


def main() -> int:
    connection = None
    try:
        connection = db.get_migration_conn()
        db.init_schema(connection)
        print(json.dumps({
            "migration_version": db.CONTROL_SCHEMA_VERSION,
            "status": "current",
        }, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({
            "migration_version": db.CONTROL_SCHEMA_VERSION,
            "status": "failed",
        }, sort_keys=True))
        return 1
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    raise SystemExit(main())
