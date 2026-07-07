import os
import uuid
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

# No real credential in the default: set PG_DSN in .env. The placeholder
# password intentionally fails auth so a missing .env surfaces loudly
# instead of silently connecting with a committed secret.
PG_DSN = os.getenv("PG_DSN", "postgresql://rag:CHANGE_ME@localhost:5433/ragdb")

# fastembed's Qdrant/bm25 hashes tokens (no fixed vocabulary) into a range that
# exceeds pgvector's sparsevec dimension cap (1_000_000_000). Every sparse index
# is remapped into [1, SPARSE_DIM] before storage or query -- this constant must
# match the `sparsevec(N)` dimension declared in schema.sql exactly.
SPARSE_DIM = 999_999_937


def get_conn() -> psycopg.Connection:
    conn = psycopg.connect(PG_DSN)
    register_vector(conn)
    return conn


def init_schema(conn) -> None:
    sql = Path(__file__).parent.joinpath("schema.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def upsert_document(conn, filename: str, file_type: str, status: str = "processing") -> str:
    """Look up a document by filename, reusing its id if it already exists."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (id, filename, file_type, status) "
            "VALUES (%(id)s, %(filename)s, %(file_type)s, %(status)s) "
            "ON CONFLICT (filename) DO UPDATE SET status = %(status)s "
            "RETURNING id",
            {"id": str(uuid.uuid4()), "filename": filename, "file_type": file_type, "status": status},
        )
        document_id = cur.fetchone()[0]
    conn.commit()
    return str(document_id)


def get_document(conn, document_id: str) -> dict | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, filename, file_type, uploaded_at, status FROM documents WHERE id = %s", (document_id,))
        return cur.fetchone()


def set_document_status(conn, document_id: str, status: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE documents SET status = %s WHERE id = %s", (status, document_id))
    conn.commit()


def clear_chunks_for_document(conn, document_id: str) -> None:
    """Delete only this document's previous chunks, so re-ingesting one file
    doesn't wipe out every other document (see the old `clear_chunks`, which
    used to TRUNCATE the whole table on every ingest run)."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
    conn.commit()


def remap_sparse_index(raw_id: int) -> int:
    return (raw_id % SPARSE_DIM) + 1


def sparse_to_literal(indices, values) -> str:
    remapped = ((remap_sparse_index(i), v) for i, v in zip(indices, values))
    pairs = ",".join(f"{i}:{v}" for i, v in sorted(remapped))
    return f"{{{pairs}}}/{SPARSE_DIM}"


def upsert_chunks(conn, rows: list[dict]) -> None:
    prepared = [
        {**r, "headings": Json(r["headings"]), "table_data": Json(r["table_data"]) if r["table_data"] else None}
        for r in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (id, document_id, type, text, source_tag, page, headings, table_data, dense, sparse) "
            "VALUES (%(id)s, %(document_id)s, %(type)s, %(text)s, %(source_tag)s, %(page)s, %(headings)s, "
            "%(table_data)s, %(dense)s, %(sparse)s::sparsevec)",
            prepared,
        )
    conn.commit()


def hybrid_search(conn, dense_vec, sparse_indices, sparse_values, top_k=15, rrf_k=1) -> list[dict]:
    sparse_lit = sparse_to_literal(sparse_indices, sparse_values)
    cols = (
        "c.id, c.type, c.text, c.source_tag, c.page, c.headings, c.table_data, "
        "d.filename"
    )
    from_clause = "chunks c LEFT JOIN documents d ON c.document_id = d.id"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {cols} FROM {from_clause} ORDER BY c.dense <=> %s::vector LIMIT %s",
            (dense_vec, top_k),
        )
        dense_ranked = cur.fetchall()

        cur.execute(
            f"SELECT {cols} FROM {from_clause} ORDER BY c.sparse <#> %s::sparsevec LIMIT %s",
            (sparse_lit, top_k),
        )
        sparse_ranked = cur.fetchall()

    # Reciprocal Rank Fusion: combine the two rankings into one score per chunk.
    scores: dict = {}
    payloads: dict = {}
    for ranked_list in (dense_ranked, sparse_ranked):
        for rank, row in enumerate(ranked_list, start=1):
            rid = row["id"]
            scores[rid] = scores.get(rid, 0.0) + 1.0 / (rrf_k + rank)
            payloads[rid] = row

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [payloads[rid] for rid, _ in fused]
