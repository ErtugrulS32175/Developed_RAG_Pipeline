import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

from pipeline.index import db
from pipeline.index import publication
from pipeline.index.attempt_contract import (
    AttemptAlreadyRunning,
    AttemptFenced,
    AttemptLeaseLost,
    AttemptOutcome,
    CandidateConflict,
    CandidateNotPublished,
    CandidateSuperseded,
    ExitCode,
)
from pipeline.extraction.router import route_and_parse
from pipeline.extraction.table_export import (
    table_to_markdown,
    save_table_xlsx,
    save_table_csv,
    save_table_json,
    estimate_table_confidence,
    export_result_xlsx,
)
from pipeline.index.embeddings import (
    embed_dense,
    embed_sparse,
    embedding_fingerprint,
)
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


def _content_key(document_id, chunk, index):
    """WHAT the chunk is, independent of which generation carries it --
    the lookup key for embedding reuse and the manifest's vocabulary."""
    key = f"{document_id}|{chunk['source_tag']}|{index}|{chunk['text']}"
    return str(uuid.uuid5(_CHUNK_NS, key))


def _chunk_id(document_id, chunk, index, generation):
    """WHICH row: the content key qualified by its immutable generation.
    Two generations holding the same content are two rows -- rows are
    copied between generations, never moved, because moving them is how an
    audit probe made the SERVED version lose its own unchanged chunks."""
    key = (f"{document_id}|g{generation}|{chunk['source_tag']}|{index}"
           f"|{chunk['text']}")
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

def _snapshot_bytes(body, name):
    """A private copy of bytes ALREADY IN HAND, under the original name.

    The name is kept because page tags, table ids and export filenames are
    all derived from it."""
    snapshot_dir = Path(tempfile.mkdtemp(prefix="ingest-anlik-"))
    snapshot = snapshot_dir / name
    snapshot.write_bytes(body)
    return snapshot, snapshot_dir


def _snapshot_of(path):
    """A PRIVATE copy of the bytes, and their hash, from ONE read.

    Hashing the path and parsing the path were two separate reads, and an
    ABA probe swapped the file to a second content between them and back
    before the re-hash: chunks from content B promoted under content A's
    hash. No re-hashing schedule closes that -- only parsing a private
    snapshot of the hashed bytes does."""
    source_bytes = Path(path).read_bytes()
    content_sha256 = hashlib.sha256(source_bytes).hexdigest()
    snapshot, snapshot_dir = _snapshot_bytes(source_bytes, Path(path).name)
    return snapshot, snapshot_dir, content_sha256


def _chunks_from_parts(parts, snapshot):
    """Parsed parts -> the chunks this run will index."""
    all_chunks, before_merge = [], 0
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
            # a table is already a unit; merging would corrupt its markdown
            # and break the mapping to its table_data
            part = chunks_from_tables(content, source_tag, snapshot.stem)
            before_merge += len(part)
            all_chunks.extend(part)
    folded = before_merge - len(all_chunks)
    print(f"[INGEST] Toplam {len(all_chunks)} chunk"
          + (f" ({folded} kirinti komsusuna katildi)" if folded else ""))
    return all_chunks


def _close_attempt(conn, attempt, status, note=None):
    """Record this run's verdict on its OWN attempt, best effort.

    Best effort in ONE direction only, and narrowly: a FENCED attempt and
    a LOST LEASE are refusals this run expected -- it no longer owns
    anything, and that refusal must not replace the failure the caller
    actually needs to see. Everything else propagates, including
    ``AttemptRecordInconsistent``: catching the whole AttemptError family
    once meant a run whose attempt record could not be closed at all
    still returned "partial completed", which is the same kind of lie
    the attempt record exists to prevent."""
    try:
        db.record_attempt_outcome(conn, attempt, status, note)
    except (AttemptFenced, AttemptLeaseLost) as refusal:
        print(f"[INGEST] deneme sonucu yazilamadi ({type(refusal).__name__}): "
              f"bu kosu cevrildi ya da lease'i devredildi")


def abandon_attempt(attempt, note):
    """Close an attempt as ERROR over a SHORT connection of its own.

    Two callers, one need: a lease that outlives the run holding it
    blocks the document for everyone until it EXPIRES.

    The core uses it for every failure BEFORE the indexing connection
    exists.

    The whole early phase runs without a connection on purpose: the parse
    is minutes long and must not hold a pooled one. Failing there used to
    just raise, leaving the lease held until it EXPIRED -- nobody could
    retry the document for the whole lease window because of a run that
    had already given up. A short connection to record the verdict costs
    one round trip and frees the document immediately.

    A CALLER THAT TOOK THE LEASE ITSELF uses it for the same reason. The
    API takes the attempt before anything is parsed, so if the run comes
    back with no terminal verdict -- or raises before recording one --
    the lease is the endpoint's to clean up. That call is IDEMPOTENT BY
    CONSTRUCTION rather than by a flag: a run that already recorded its
    own verdict cleared the lease in the same statement (rule 14), so the
    authority check refuses this second closure as a lost lease and
    ``_close_attempt`` absorbs it. Only an attempt that really ended
    still holding its lease is closed here.

    The note carries the exception's TYPE and nothing else. A parser's
    message can contain a file path, a fragment of the document or a
    service endpoint, and the attempt record is not a place to copy any
    of those to.

    What this deliberately does NOT do is swallow. ``_close_attempt``
    absorbs a fence and a lost lease -- refusals a run that no longer
    owns anything expects -- so the ORIGINAL parser failure survives
    them. Anything else, ``AttemptRecordInconsistent`` above all, comes
    out: a run that could not close its own record has a second problem,
    and hiding it behind the first is how both get missed."""
    conn = db.get_conn()
    try:
        _close_attempt(conn, attempt, AttemptOutcome.ERROR, note)
    finally:
        conn.close()


def _refuse_if_superseded(conn, attempt):
    """A cheap early-out between the long parse and the first write.

    The parse is the minutes-long phase. Everything after it costs calls
    to the embedding service, so a run that was fenced WHILE it parsed
    should find out here rather than after paying for a generation's
    worth of vectors.

    It only ever REFUSES. It does not re-read ``observed_active`` and it
    adopts nothing from the row -- re-binding to what another run had
    just promoted is the defect this whole package exists to close, and
    a check that can only abort cannot re-bind. Authority itself still
    lives where it always did: inside each write's OWN transaction. This
    is the fast lane, not the gate."""
    row = db.get_document(conn, attempt.document_id)
    if row is None:
        raise AttemptLeaseLost(
            "belge kaydi yok; bu deneme artik hicbir sey yazamaz")
    if str(row.get("candidate_id")) != str(attempt.candidate_id):
        raise AttemptFenced(
            "aday parse sirasinda degisti; bu deneme cevrildi ve hicbir "
            "sey yazmadi")


def ingest_attempt(snapshot, attempt):
    """Index a snapshot UNDER AN ATTEMPT. The attempt is not optional.

    There is no unbound path into the index. Everything this run needs to
    know about what it is indexing comes from the attempt object and from
    nowhere else:

      * ``observed_active`` is the generation the attempt captured AT ITS
        START and is never re-read. A run that re-read it after parsing
        re-bound itself to whatever another run had just promoted, and
        then stamped its own failure onto that healthy generation.
      * ``candidate_id`` and ``candidate_sha`` are the attempt's too, so
        the promotion swaps against the identity this run was actually
        given rather than whatever the row says by the time it finishes.

    The verdict of a failed or partial run goes on the ATTEMPT, never on
    the document: a run that failed did not make the SERVED version any
    worse, and the document's status describes the served version.

    Which is exactly why this RETURNS the verdict -- ``(outcome, note)``,
    outcome in {DONE, PARTIAL}. There is nowhere else for the caller to
    read it: on a partial run the document row is deliberately left
    alone, so an endpoint that stamped `processing` and then read the row
    back to see how the run went concluded that nothing had finished. It
    answered 500 and marked a healthy index `error` -- destroying the
    honesty `partial` exists to carry. Failures RAISE; only the two
    terminal successes come back.
    """
    # THE EARLY PHASE, all of it, under one guard. The two hash checks
    # were guarded and the parse was not, so a parser or chunker that
    # raised -- a corrupt page, a model timeout, a bug -- left the lease
    # held until it expired. Any failure here closes the attempt.
    try:
        actual = hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()
        if actual != attempt.candidate_sha:
            # the snapshot is ours, so this guards our own machinery (a
            # corrupt temp volume, an interfering scanner) rather than an
            # attacker with the path -- the swappable original was never
            # handed to the parser at all
            raise RuntimeError(
                "anlik goruntu bagli adayin baytlari degil; indeks el "
                "degmedi")

        parts, parse_failures = route_and_parse(str(snapshot))
        if hashlib.sha256(Path(snapshot).read_bytes()).hexdigest() != actual:
            # checked on BOTH sides of the parse. The before-check binds
            # the run to its candidate; this one catches our own
            # machinery moving underneath us mid-parse. Neither replaces
            # the other.
            raise RuntimeError(
                "anlik goruntu ingest sirasinda degisti -- mevcut indeks "
                "korundu")
        print(f"\n[INGEST] {len(parts)} parca parse edildi, chunk'laniyor..."
              + (f" ({len(parse_failures)} parca HATALI, kayit kismi olacak)"
                 if parse_failures else ""))
        all_chunks = _chunks_from_parts(parts, Path(snapshot))
    except BaseException as failure:
        abandon_attempt(attempt, type(failure).__name__)
        raise

    document_id = attempt.document_id
    observed_active = attempt.observed_active      # captured at START

    # The connection's whole life sits inside one try/finally: an
    # exception in schema init, in the metadata reads or in the
    # finalisation used to leak it just as surely as one in the embed loop.
    conn = db.get_conn()
    try:
        db.init_schema(conn)
        print(f"[INGEST] Belge: {document_id} · deneme "
              f"{attempt.attempt_id}")

        # Fenced while we parsed? Then stop before spending anything.
        _refuse_if_superseded(conn, attempt)

        # A parse that produced no USABLE text and reported no failure is
        # a lie somewhere upstream -- refuse before touching the index.
        # Counting PARTS was not enough: a parser once returned a part
        # whose every chunk was whitespace, and the empty generation it
        # staged passed a size-0 manifest and swept the healthy index.
        usable = sum(1 for c in all_chunks if c["text"].strip())
        if usable == 0 and not parse_failures:
            _close_attempt(conn, attempt, AttemptOutcome.ERROR,
                           "parse kullanilabilir icerik uretmedi")
            raise RuntimeError(
                "parse kullanilabilir icerik uretmedi -- mevcut indeks "
                "korundu")

        # Everything this run writes is STAGED under a generation number
        # that is immutably THIS ATTEMPT'S OWN (an atomic counter, never
        # active+1: reusing a number let consecutive different contents
        # merge into one staging generation). Retrieval keeps serving the
        # observed active generation until promotion.
        new_generation = db.allocate_generation(conn, document_id, attempt)

        # Chunks stored by ANY earlier attempt, findable by content key --
        # but reusable ONLY under this run's exact embedding fingerprint:
        # a text match alone once carried a superseded model's vectors
        # into the new generation.
        fingerprint = embedding_fingerprint()
        already = db.existing_content_keys(conn, document_id, fingerprint)
        if already:
            print(f"[INGEST] bu belgeden {len(already)} icerik anahtari "
                  f"ayni gomme sozlesmesiyle zaten var, degismeyenler "
                  f"tekrar gomulmeyecek")

        # Extend the lease BEFORE the long part, not only during it: the
        # embed loop can run for minutes and a lease that expired at its
        # first batch would be taken over while this run was still
        # working. It extends the lease and nothing more -- the right to
        # WRITE is checked inside each write's own transaction.
        db.heartbeat_attempt(conn, attempt)
        try:
            batch, copy_pairs, manifest, skipped = [], [], set(), 0
            for i, c in enumerate(all_chunks):
                if not c["text"].strip():
                    continue
                content_key = _content_key(document_id, c, i)
                chunk_id = _chunk_id(document_id, c, i, new_generation)
                manifest.add(chunk_id)
                if content_key in already:
                    skipped += 1
                    # THIS parse's payload travels with the copy; only the
                    # vectors are inherited (copy_chunks_into_generation)
                    copy_pairs.append({
                        "id": chunk_id,
                        "document_id": document_id,
                        "type": c["type"],
                        "text": c["text"],
                        "source_tag": c["source_tag"],
                        "page": c["page"],
                        "headings": c["headings"],
                        "table_data": c.get("table_data"),
                        "generation": new_generation,
                        "content_key": content_key,
                        "embedding_fingerprint": fingerprint,
                    })
                    continue
                sparse_indices, sparse_values = _retry(
                    lambda: embed_sparse(c["text"]))
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
                    "sparse": db.sparse_to_literal(sparse_indices,
                                                   sparse_values),
                    "generation": new_generation,
                    "content_key": content_key,
                    "embedding_fingerprint": fingerprint,
                })
                if len(batch) >= 32:
                    db.upsert_chunks(conn, batch, attempt)
                    batch = []
                    db.heartbeat_attempt(conn, attempt)
                    print(f"  {i+1}/{len(all_chunks)} yazildi...")
            if batch:
                db.upsert_chunks(conn, batch, attempt)
            # earlier attempts' embeddings join this generation by COPY --
            # the expensive call is skipped and no served row ever moves
            db.copy_chunks_into_generation(conn, copy_pairs, attempt)
        except Exception as failure:
            _close_attempt(conn, attempt, AttemptOutcome.ERROR,
                           type(failure).__name__)
            # what has been written stays STAGED: ids are content-derived,
            # so re-running picks up from here instead of starting over,
            # and the active generation was never touched
            print(f"[INGEST] HATA -- yazilanlar evrede korundu, ayni komutu "
                  f"tekrar calistirarak kaldigi yerden devam edebilirsin")
            raise

        # Promotion is the ONLY place old rows die, and it exists only for
        # a COMPLETE parse: manifest membership + three-part CAS + lease
        # release + the attempt's DONE + sweep, one transaction. A partial
        # run leaves its rows staged and the previous complete generation
        # still serving -- so "partial" means exactly "the last full
        # version is what you are searching", never "two editions mixed"
        # and never "and I deleted good rows".
        if parse_failures:
            note = json.dumps(parse_failures, ensure_ascii=False)
            _close_attempt(conn, attempt, AttemptOutcome.PARTIAL, note)
            verdict = (AttemptOutcome.PARTIAL, note)
            stale = 0
        else:
            verdict = (AttemptOutcome.DONE, None)
            stale = db.promote_generation(
                conn, document_id, new_generation,
                expected_active=observed_active,
                manifest_ids=manifest,
                content_sha256=attempt.candidate_sha,
                candidate_id=attempt.candidate_id,
                attempt_id=attempt.attempt_id)

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE document_id = %s",
                        (document_id,))
            count = cur.fetchone()[0]
        extra = []
        if skipped:
            extra.append(f"{skipped} atlandi")
        if stale:
            extra.append(f"{stale} onceki nesle ait chunk silindi")
        suffix = f" ({', '.join(extra)})" if extra else ""
        if parse_failures:
            print(f"\n[INGEST] KISMI tamamlandi ({len(parse_failures)} parca "
                  f"hatali). Evrelenen nesil bekliyor, aktif surum degismedi. "
                  f"Vektor sayisi: {count}{suffix}")
            for failure in parse_failures:
                print(f"  eksik: {failure['kaynak']} ({failure['asama']}): "
                      f"{failure['hata']}")
        else:
            print(f"\n[INGEST] Tamamlandi, nesil {new_generation} aktif. "
                  f"Bu belge icin vektor sayisi: {count}{suffix}")
        return verdict
    finally:
        conn.close()


def main(path, expected_candidate=None, attempt=None):
    """Snapshot a path and run the core under an attempt.

    Two callers reach the core through here, and they differ in WHO takes
    the lease:

      * the API passes ``attempt``. It must take the lease itself, before
        anything is parsed, because a candidate that is still STAGED has
        to come back as 409 from the endpoint rather than as a failure
        halfway through an ingest.
      * a caller holding only the row it read passes
        ``expected_candidate`` -- ``(candidate_id, content_sha256)`` --
        and this wrapper verifies both halves before taking the lease.
        The bytes on disk must still be the recorded candidate's, and the
        recorded candidate must still be the one the caller read: an
        audited race had an old ingest quietly re-record old bytes over a
        newer authorised upload and promote them.

    The CLI does not come through here at all -- it is a first-class
    publisher with its own protocol (see ``cli_main``).

    Returns whatever the core returned: this run's own ``(outcome,
    note)``."""
    filename = Path(path).name
    try:
        snapshot, snapshot_dir, content_sha256 = _snapshot_of(path)
    except BaseException as failure:
        # WHEN THE CALLER ALREADY HOLDS THE LEASE, this is not merely a
        # failed call. The API takes the attempt BEFORE calling, so that
        # a candidate still being published can be answered 409 instead
        # of discovered halfway through an ingest -- which means an
        # unreadable path or a full temp volume here used to raise with
        # the lease still held, and nobody could retry the document
        # until it EXPIRED. Only the exception's TYPE is recorded: a
        # path or an OS message is not something to copy into the
        # attempt record.
        if attempt is not None:
            abandon_attempt(attempt, type(failure).__name__)
        raise
    try:
        if attempt is None:
            if expected_candidate is not None:
                _cid, expected_sha = expected_candidate
                if content_sha256 != expected_sha:
                    # refused BEFORE parsing: the disk no longer carries
                    # the recorded candidate (a newer upload landed, or
                    # the file was touched) -- processing it would bind
                    # the wrong bytes
                    raise RuntimeError(
                        "disk icerigi kayitli adayla uyusmuyor; islenmedi "
                        "-- yeni bir yukleme araya girmis ya da dosya "
                        "degismis olabilir")
            conn = db.get_conn()
            try:
                db.init_schema(conn)
                row = db.lookup_document(conn, filename)
                if row is None:
                    raise RuntimeError(
                        "belge kaydi yok; once aday yayimlanmali")
                if (expected_candidate is not None
                        and row.get("candidate_id") != expected_candidate[0]):
                    raise RuntimeError(
                        "aday kimligi artik kayitli olan degil; bu kosu "
                        "iptal edildi, indeks el degmedi")
                attempt = db.begin_attempt(conn, row["id"])
            finally:
                conn.close()
        return ingest_attempt(snapshot, attempt)
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


# Not one of the frozen contract codes: those name the five decisions the
# protocol can make, and "you typed the command wrong" is not one of them.
USAGE_EXIT = 1

USAGE = ("kullanim: python -m pipeline.index.ingest <dosya> [--replace]\n"
         "  --replace   ayni ada FARKLI icerik yayimlamaya acik yetki verir")


def _parsed_argv(argv):
    """``(path, allow_replace)``, or None when the command line is not one.

    Parsed by hand rather than with argparse: argparse ends the PROCESS
    with status 2 on a usage error, and 2 is already
    ``CANDIDATE_CONFLICT`` -- a mistyped command would be
    indistinguishable from a refused replacement, which is exactly the
    confusion the frozen codes exist to prevent."""
    allow_replace = False
    targets = []
    for token in argv:
        if token == "--replace":
            allow_replace = True
        elif token.startswith("-"):
            return None
        else:
            targets.append(token)
    if len(targets) != 1:
        return None
    return targets[0], allow_replace


def cli_main(argv):
    """The CLI: publish a candidate, take an attempt, index UNDER it.

    The CLI is a FIRST-CLASS PUBLISHER, through the same internal
    protocol the API uses and in the same order:

        publish_candidate -> begin_attempt -> ingest_attempt(attempt)

    It used to have no publication step at all: it parsed whatever was on
    the path and then knocked the candidate gate itself, with a hash that
    equalled the SERVED one -- a legitimate arm of the gate -- so a stale
    snapshot could mint a fresh candidate id over a newer authorised
    upload and promote itself. Publishing FIRST is what removes that
    ordering entirely; there is no longer a moment where indexed bytes
    have not been recorded.

    Replacement is authorised by ``--replace`` and by nothing else. There
    is no environment variable for it: a global flag would grant every
    document at once what the caller meant for one.

    A CONTRACT REFUSAL IS A RETURN VALUE, never a traceback. A caller
    cannot tell a crash from a policy decision, so the five decisions the
    protocol can make each have their own frozen code (``ExitCode``).
    Anything genuinely unexpected still propagates -- a crash should look
    like a crash."""
    parsed = _parsed_argv(list(argv))
    if parsed is None:
        print(USAGE, file=sys.stderr)
        return USAGE_EXIT
    target, allow_replace = parsed

    source = Path(target)
    try:
        body = source.read_bytes()
    except OSError as error:
        print(f"[INGEST] dosya okunamadi ({type(error).__name__})",
              file=sys.stderr)
        return USAGE_EXIT

    # ONE read of the path, used for BOTH the publication and the parse:
    # reading it twice is how an ABA swap got content B indexed under
    # content A's hash. What is published and what is parsed are the same
    # bytes because they are the same object in memory.
    file_type = source.suffix.lower().lstrip(".")
    snapshot, snapshot_dir = _snapshot_bytes(body, source.name)
    try:
        conn = db.get_conn()
        try:
            db.init_schema(conn)
            document_id, _candidate_id, canonical = (
                publication.publish_candidate(
                    conn, source.name, file_type, body,
                    allow_replace=allow_replace))
            print(f"[INGEST] aday yayimlandi: {canonical}")
            attempt = db.begin_attempt(conn, document_id)
        finally:
            conn.close()
        ingest_attempt(snapshot, attempt)
    except CandidateConflict:
        print("[INGEST] ayni ada farkli icerik zaten kayitli; degistirmek "
              "bilincliyse --replace ver", file=sys.stderr)
        return ExitCode.CANDIDATE_CONFLICT
    except (CandidateNotPublished, CandidateSuperseded):
        print("[INGEST] aday yayimlanmis degil; islenmedi", file=sys.stderr)
        return ExitCode.CANDIDATE_NOT_PUBLISHED
    except AttemptAlreadyRunning:
        print("[INGEST] bu belge icin canli bir kosu var; lease baskasinda",
              file=sys.stderr)
        return ExitCode.ATTEMPT_UNAVAILABLE
    except (AttemptFenced, AttemptLeaseLost):
        print("[INGEST] bu kosu cevrildi ya da lease'i devredildi; indeks "
              "el degmedi", file=sys.stderr)
        return ExitCode.ATTEMPT_LOST
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    return ExitCode.OK


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
