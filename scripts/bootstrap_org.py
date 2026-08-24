"""Bootstrap one tenant and its content-blind OpenWebUI architecture admin."""
import argparse
import json
import uuid

from pipeline.index import db


def parser():
    value = argparse.ArgumentParser(
        description="Create an organization tenant and its first architect")
    value.add_argument("--tenant-id", required=True, type=uuid.UUID)
    value.add_argument("--tenant-name", required=True)
    value.add_argument("--openwebui-subject", required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.tenant_name.strip() or args.tenant_name != args.tenant_name.strip():
        raise SystemExit("tenant-name boslukla baslayamaz veya bitemez")
    if (not args.openwebui_subject.strip()
            or args.openwebui_subject != args.openwebui_subject.strip()
            or any(not 32 <= ord(char) < 127
                   for char in args.openwebui_subject)):
        raise SystemExit("openwebui-subject gecersiz")
    conn = db.get_conn(service=True)
    try:
        db.require_runtime_ready(conn)
        result = db.bootstrap_org_tenant(
            conn, tenant_id=args.tenant_id, name=args.tenant_name,
            issuer="open-webui", subject=args.openwebui_subject)
    finally:
        conn.close()
    print(json.dumps({key: str(value) for key, value in result.items()},
                     sort_keys=True))


if __name__ == "__main__":
    main()
