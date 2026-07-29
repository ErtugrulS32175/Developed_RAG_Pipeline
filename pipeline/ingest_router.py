import os
import re
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from pipeline import db
from pipeline.router import route_and_parse
from pipeline.table_export import (
    table_to_markdown,
    save_table_xlsx,
    save_table_csv,
    save_table_json,
    estimate_table_confidence,
    export_result_xlsx,
)
from pipeline.embeddings import embed_dense, embed_sparse
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output"))
# Tables at or below this confidence, or with any validation issue, are copied
# to output/tables/_review/ and marked needs_review so a human checks them.
REVIEW_THRESHOLD = float(os.getenv("TABLE_REVIEW_THRESHOLD", "0.9"))

hf_tok = AutoTokenizer.from_pretrained("BAAI/bge-m3")
tokenizer = HuggingFaceTokenizer(tokenizer=hf_tok, max_tokens=512)
# always_emit_headings: by default a heading with no body under it is dropped
# entirely -- not in any chunk's text, not in any chunk's metadata. A document's
# masthead is exactly that shape, so the facts printed there (a law's number and
# date, a report's period) were absent from the index and unanswerable.
chunker = HybridChunker(tokenizer=tokenizer, always_emit_headings=True)


def chunk_text(chunk):
    """The text to index for a chunk: its body with its heading path prepended.

    `chunk.text` alone drops headings -- the chunker keeps them as metadata, so
    nothing that lives in a heading is searchable, and a heading-only chunk has
    no text at all. Contextualising also gives every ordinary chunk the section
    it belongs to, which is what tells two similarly-worded passages apart.
    """
    return chunker.contextualize(chunk)

_PAGE_TAG_RE = re.compile(r"page(\d+)")


def page_from_tag(source_tag):
    """Page number carried by the router's source tag ("page26:native" -> 26).

    The router hands the converter ONE single-page PDF per page, so the parsed
    document's own provenance always reports page 1 no matter which page it is
    -- the tag is the only place the real page number survives. Tags without a
    page (a standalone image) are page 0.
    """
    m = _PAGE_TAG_RE.match(source_tag or "")
    return int(m.group(1)) if m else 0


# A chunk this short is usually a fragment of a label/value block that the
# chunker split away from its subject -- measured case: a 45-character piece
# holding a company's founding year, with the company's name in a different
# chunk. Nothing in it matches a question about that company, so neither dense
# nor sparse search can reach it, and the fact is effectively lost.
#
# The threshold is measured, not guessed. At 150 a legitimate ~146-character
# passage was merged into its neighbour, and the combined chunk's embedding was
# diluted enough to drop it out of the results -- trading one lost answer for
# another. Fragments cluster well below 100; real short passages sit above it.
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "100"))
# ...but never grow a chunk without limit: an over-long passage dilutes its own
# embedding, which is the problem chunking exists to avoid.
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "2500"))


def merge_small_chunks(chunks, min_chars=None, max_chars=None):
    """Fold undersized chunks into an adjacent one so no fragment is left
    without the context that makes it findable.

    Merges in either direction -- a short chunk joins the one before it, and a
    short chunk at the START of a part absorbs the one after it, which it could
    not do if merging only ever looked backwards.

    Two boundaries are never crossed: `source_tag` (so a merge cannot move text
    onto the wrong page) and `type` (so a table fragment is never folded into
    narrative prose). Table chunks built by `chunks_from_tables` are whole units
    and are not passed through here at all.
    """
    min_chars = MIN_CHUNK_CHARS if min_chars is None else min_chars
    max_chars = MAX_CHUNK_CHARS if max_chars is None else max_chars

    merged = []
    for chunk in chunks:
        prev = merged[-1] if merged else None
        joinable = (
            prev is not None
            and prev["source_tag"] == chunk["source_tag"]
            and prev["type"] == chunk["type"]
            and len(prev["text"]) + len(chunk["text"]) + 1 <= max_chars
        )
        if joinable and (len(chunk["text"]) < min_chars or len(prev["text"]) < min_chars):
            prev["text"] = f"{prev['text']}\n{chunk['text']}"
            if not prev.get("headings"):
                prev["headings"] = chunk.get("headings") or []
            continue
        merged.append(dict(chunk))
    return merged


def chunk_plain_text(text, source_tag, max_tokens=480):
    """Split plain OCR text into token-bounded chunks by paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, current, cur_len = [], [], 0
    for p in paragraphs:
        p_len = len(hf_tok.encode(p, add_special_tokens=False))
        if cur_len + p_len > max_tokens and current:
            chunks.append(" ".join(current))
            current, cur_len = [], 0
        current.append(p)
        cur_len += p_len
    if current:
        chunks.append(" ".join(current))
    page = page_from_tag(source_tag)
    return [{"type": "text", "text": c, "source_tag": source_tag, "page": page, "headings": []} for c in chunks]

# A chunk's id is derived from what it is, not generated fresh each run. That
# single choice is what makes an interrupted ingest resumable: re-running
# produces the same ids, already-stored rows are recognised and skipped, and the
# embedding calls -- the slow, costly part -- are not repeated.
_CHUNK_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

RETRY_ATTEMPTS = int(os.getenv("INGEST_RETRY_ATTEMPTS", "4"))
RETRY_BACKOFF = float(os.getenv("INGEST_RETRY_BACKOFF", "2.0"))


def _chunk_id(document_id, chunk, index):
    key = f"{document_id}|{chunk['source_tag']}|{index}|{chunk['text']}"
    return str(uuid.uuid5(_CHUNK_NS, key))


def _retry(fn, attempts=None, backoff=None):
    """Retry a call that goes over the network, backing off between tries.

    A whole ingest used to die on one hiccup from the embedding service -- a
    restart, a timeout, a moment of load -- after minutes of work. Most such
    failures pass within seconds, so the fix is to wait rather than to abandon
    the run. The last failure is re-raised so a genuine outage still stops us.
    """
    attempts = RETRY_ATTEMPTS if attempts is None else attempts
    backoff = RETRY_BACKOFF if backoff is None else backoff
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == attempts:
                raise
            wait = backoff ** (attempt - 1)
            print(f"  [RETRY] {type(e).__name__}: {str(e)[:60]} "
                  f"-- {wait:.0f}s sonra tekrar ({attempt}/{attempts - 1})")
            time.sleep(wait)


def _write_review_report(path, table_id, confidence, issues, table=None):
    """Human-readable note dropped next to a flagged table's copy in _review/."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Tablo: {table_id}", f"Guven: {confidence:.2f}"]
    if table and table.get("mode") == "consensus":
        lines.append(f"Modeller: {'+'.join(table.get('backends', []))}")
        lines.append(f"Uyum: {table.get('agreement')}"
                     f" ({len(table.get('disagreements', []))} hucrede ayrisma)")
    lines.append("")
    if issues:
        lines.append("Sorunlar:")
        lines.extend(f"  - {x}" for x in issues)
    else:
        lines.append("Guven esigin altinda (issue listelenmedi).")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_xlsx(table, headers, rows, path):
    """Pipeline results carry disagreement/review metadata, so they get the rich
    export (amber-highlighted cells + Rapor sheet). Anything else falls back to a
    plain sheet."""
    if "needs_review" in table:
        export_result_xlsx(table, str(path))
    else:
        save_table_xlsx(headers, rows, path)

def chunks_from_tables(tables, source_tag, doc_stem):
    """Turn structured table results into RAG chunks and write xlsx/csv/json exports."""
    page_no = page_from_tag(source_tag)

    chunks = []
    tables_dir = OUTPUT_DIR / "tables"
    review_dir = tables_dir / "_review"
    for i, table in enumerate(tables):
        headers, rows = table["headers"], table["rows"]
        table_id = f"{doc_stem}_{source_tag.replace(':', '_')}_{i}"
        # the table pipeline already scored this table (number verification,
        # model agreement, template stage) and decided whether it needs review;
        # fall back to the shape-only proxy for tables from elsewhere.
        confidence = table.get("confidence")
        if confidence is None:
            confidence = estimate_table_confidence(headers, rows)
        issues = table.get("issues", [])
        needs_review = table.get("needs_review")
        if needs_review is None:
            needs_review = confidence < REVIEW_THRESHOLD or bool(issues)

        _save_xlsx(table, headers, rows, tables_dir / f"{table_id}.xlsx")
        save_table_csv(headers, rows, tables_dir / f"{table_id}.csv")
        save_table_json(table_id, page_no, headers, rows, confidence, tables_dir / f"{table_id}.json")

        if needs_review:
            _save_xlsx(table, headers, rows, review_dir / f"{table_id}.xlsx")
            _write_review_report(review_dir / f"{table_id}.issues.txt", table_id,
                                 confidence, issues, table)
            print(f"  [REVIEW] {table_id}: guven {confidence:.2f}, {len(issues)} sorun -> {review_dir}")

        # Carry the trust signals into the chunk, not just the values: a later
        # consumer (retrieval, an analytics layer, a reviewer UI) needs to know
        # which cells the two models disagreed on, not only what was extracted.
        table_data = {
            "table_id": table_id,
            "page": page_no,
            "headers": headers,
            "rows": rows,
            "confidence": confidence,
            "needs_review": needs_review,
            "issues": issues,
        }
        for key in ("mode", "backends", "agreement", "disagreements",
                    "structural_confidence", "number_fidelity", "template"):
            if key in table:
                table_data[key] = table[key]

        chunks.append({
            "type": "table",
            # No citation header inside the text: query.build_context builds one
            # for EVERY chunk from its stored metadata, so embedding it here
            # would both duplicate it and pollute the indexed tokens.
            "text": table_to_markdown(headers, rows),
            "source_tag": source_tag,
            "page": page_no,
            "headings": [],
            "table_data": table_data,
        })
    return chunks

def main(path):
    filename = Path(path).name
    parts = route_and_parse(path)
    print(f"\n[INGEST] {len(parts)} parca parse edildi, chunk'laniyor...")

    all_chunks = []
    before_merge = 0
    for source_tag, (content_type, content) in parts:
        if content_type == "docling":
            part = []
            for chunk in chunker.chunk(content):
                ctype = "text"
                if chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if "table" in str(item.label).lower():
                            ctype = "table"
                part.append({
                    "type": ctype,
                    "text": chunk_text(chunk),
                    "source_tag": source_tag,
                    "page": page_from_tag(source_tag),
                    "headings": chunk.meta.headings or [],
                })
            before_merge += len(part)
            all_chunks.extend(merge_small_chunks(part))
        elif content_type == "text":
            part = chunk_plain_text(content, source_tag)
            before_merge += len(part)
            all_chunks.extend(merge_small_chunks(part))
        elif content_type == "tables":
            # a table is already a unit; merging would corrupt its markdown and
            # break the mapping to its table_data
            part = chunks_from_tables(content, source_tag, Path(path).stem)
            before_merge += len(part)
            all_chunks.extend(part)

    folded = before_merge - len(all_chunks)
    print(f"[INGEST] Toplam {len(all_chunks)} chunk"
          + (f" ({folded} kirinti komsusuna katildi)" if folded else ""))

    conn = db.get_conn()
    db.init_schema(conn)
    document_id = db.upsert_document(conn, filename, Path(path).suffix.lower().lstrip("."))
    print(f"[INGEST] Belge: {filename} ({document_id})")

    # Chunks already stored from an earlier, interrupted run. Their ids are
    # derived from the content, so anything unchanged is skipped rather than
    # embedded a second time -- the embedding call is the expensive part.
    already = db.existing_chunk_ids(conn, document_id)
    if already:
        # how many of these get reused is not known yet: an id changes whenever
        # the text does, so a re-chunked document matches none of them
        print(f"[INGEST] bu belgeden {len(already)} chunk zaten var, "
              f"degismeyenler tekrar gomulmeyecek")

    try:
        batch, written_ids, skipped = [], set(), 0
        for i, c in enumerate(all_chunks):
            if not c["text"].strip():
                continue
            chunk_id = _chunk_id(document_id, c, i)
            written_ids.add(chunk_id)
            if chunk_id in already:
                skipped += 1
                continue
            sparse_indices, sparse_values = _retry(lambda: embed_sparse(c["text"]))
            batch.append({
                "id": chunk_id,
                "document_id": document_id,
                "type": c["type"],
                "text": c["text"],
                "source_tag": c["source_tag"],
                "page": c["page"],
                "headings": c["headings"],
                "table_data": c.get("table_data"),
                "dense": _retry(lambda: embed_dense(c["text"])),
                "sparse": db.sparse_to_literal(sparse_indices, sparse_values),
            })
            if len(batch) >= 32:
                db.upsert_chunks(conn, batch)
                batch = []
                print(f"  {i+1}/{len(all_chunks)} yazildi...")
        if batch:
            db.upsert_chunks(conn, batch)
    except Exception:
        db.set_document_status(conn, document_id, "error")
        # what has been written stays: ids are content-derived, so re-running
        # this command picks up from here instead of starting over
        print(f"[INGEST] HATA -- yazilanlar korundu, ayni komutu tekrar "
              f"calistirarak kaldigi yerden devam edebilirsin")
        raise

    # only now that everything is stored: drop rows from an older version of
    # this file that the current run did not produce
    stale = db.delete_stale_chunks(conn, document_id, written_ids)
    db.set_document_status(conn, document_id, "done")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s", (document_id,))
        count = cur.fetchone()[0]
    extra = []
    if skipped:
        extra.append(f"{skipped} atlandi")
    if stale:
        extra.append(f"{stale} eskimis chunk silindi")
    suffix = f" ({', '.join(extra)})" if extra else ""
    print(f"\n[INGEST] Tamamlandi. Bu belge icin vektor sayisi: {count}{suffix}")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "./data/sample.pdf"
    main(target)
