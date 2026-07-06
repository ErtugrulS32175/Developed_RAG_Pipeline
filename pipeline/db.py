import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Json
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

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


def clear_chunks(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE chunks")
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
            "INSERT INTO chunks (id, type, text, source_tag, page, headings, table_data, dense, sparse) "
            "VALUES (%(id)s, %(type)s, %(text)s, %(source_tag)s, %(page)s, %(headings)s, %(table_data)s, "
            "%(dense)s, %(sparse)s::sparsevec)",
            prepared,
        )
    conn.commit()


def hybrid_search(conn, dense_vec, sparse_indices, sparse_values, top_k=15, rrf_k=1) -> list[dict]:
    sparse_lit = sparse_to_literal(sparse_indices, sparse_values)
    cols = "id, type, text, source_tag, page, headings, table_data"

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT {cols} FROM chunks ORDER BY dense <=> %s::vector LIMIT %s",
            (dense_vec, top_k),
        )
        dense_ranked = cur.fetchall()

        cur.execute(
            f"SELECT {cols} FROM chunks ORDER BY sparse <#> %s::sparsevec LIMIT %s",
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
