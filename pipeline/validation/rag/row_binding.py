"""Row binding over the EXTRACTION's structure: "same row" as a fact.

Twelve audit rounds established that flat text cannot carry a row relation
losslessly -- a segmenter must GUESS record boundaries and a scorer must
GUESS which words are entities, and each guess has a wrong side that ends in
a false annotation. This module is the structural alternative those rounds
legislated for: it reads the ``table_data`` that ingest already stores on
every table chunk (headers + rows, straight from the verified extraction
pipeline), where row membership is not inferred from text but RECORDED.

The check: for each figure the answer states, find where it OCCURS in the
retrieved tables (exact fact, via numeric normalization), and find which row
the question BINDS to (stem-tolerant term overlap over the row's cells --
still lexical, but over one row's cells rather than guessed segments). A
figure that occurs only OUTSIDE every best-matching row, while a
best-matching row offers a different figure in the same column, is flagged.

One-sided, like every check in this package: no term signal, no figure
occurrence, or an equal-best tie means silence, not a guess.

DECLARED LIMITS of v0, stated rather than discovered later:
  * ROWS only. A wrong-COLUMN binding (right entity, wrong year) passes,
    because an occurrence inside the bound row counts as bound. The column
    dimension needs the question's time/attribute qualifier resolved
    against headers, and that is its own measured step.
  * Unpriced on saved runs. The stored eval rows carry flattened context
    text, not ``table_data``; the false-review cost on real answers can
    only be measured on a future run that carries structure into the
    context. Until that number exists this module follows the same
    measure-first gate as the flat-text binding signal: ANNOTATION-ONLY,
    never in the publication path, and the import-graph tests pin that.
    (That sibling module is deliberately not named here: a static test bans
    its token outside its own file, and the ban brooks no exceptions.)
  * Only as good as extraction. A table the consensus pipeline mis-read is
    mis-read here too; ``confidence``/``needs_review`` travel in the same
    ``table_data`` and a consumer may weigh them.
"""
import re

from pipeline.lang.tr_notation import fold, normalize, numbers, stems

WRONG_ROW_BINDING = "yanlis_satir_baglama"

_WORD = re.compile(r"[a-z0-9]+")
_ANSWER_PAGE = re.compile(r"sayfa\s*\d+", re.IGNORECASE)
_FIGURE_TOKEN = re.compile(
    r"(?<![a-zA-Z0-9])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d[\d.,]*\d|\d)"
    r"(?:\s*(?:bin|milyon|milyar|trilyon))?"
)
# Question machinery and units; deliberately the same list philosophy as the
# flat-text signal, duplicated so the two experimental modules stay
# independently auditable.
_STOPWORDS = frozenset("""
    nedir kactir kacti kac hangi kadar oldu olan olarak gore icin ile veya
    yuzde oran orani oraninda milyon milyar bin trilyon adet ton kwh mwp
    yil yili yilinda yilindaki sonu sonunda itibariyla toplam tam tamamen
    ne zaman nasil neden nerede neresidir kimdir mi mu tane sayi sayisi
    """.split())


def _question_terms(question: str) -> frozenset:
    terms = set()
    for word in _WORD.findall(fold(question or "")):
        if len(word) < 3 or word.isdigit() or word in _STOPWORDS:
            continue
        terms.add(word)
    return frozenset(terms)


def _answer_figures(answer: str):
    body = _ANSWER_PAGE.sub(" ", answer or "")
    for match in _FIGURE_TOKEN.finditer(body):
        forms = frozenset(numbers(normalize(match.group())))
        if forms:
            yield match.group().strip(), forms


def _cell_numbers(cell) -> frozenset:
    return frozenset(numbers(normalize(str(cell))))


def _row_score(row, terms: frozenset) -> int:
    """Question terms present anywhere in the row's cell text, stems meeting
    stems so inflection differences do not break the match."""
    union = set()
    for cell in row:
        for word in _WORD.findall(fold(str(cell))):
            if len(word) >= 3:
                union |= stems(word)
    return sum(1 for term in terms if stems(term) & union)


def _structured_tables(chunks):
    for chunk in chunks or ():
        if not isinstance(chunk, dict):
            continue
        data = chunk.get("table_data")
        if not isinstance(data, dict):
            continue
        headers = data.get("headers")
        rows = data.get("rows")
        if not isinstance(headers, list) or not isinstance(rows, list):
            continue
        clean = [row for row in rows if isinstance(row, list)]
        if clean:
            yield [str(h) for h in headers], clean


def _table_verdict(headers, rows, terms, forms):
    """One table's opinion on one figure: 'bound', 'misbound' or None.

    None everywhere the fact pattern is incomplete -- the figure absent,
    no term signal, an equal-best tie, or a best row offering nothing
    different. Only a complete wrong-row shape accuses."""
    occurrences = [
        (row_index, cell_index)
        for row_index, row in enumerate(rows)
        for cell_index, cell in enumerate(row)
        if forms & _cell_numbers(cell)
    ]
    if not occurrences:
        return None
    scores = [_row_score(row, terms) for row in rows]
    best = max(scores)
    if best == 0:
        return None
    best_rows = {i for i, score in enumerate(scores) if score == best}
    if len(best_rows) > 1:
        return None  # ambiguous binding: silence, never a guess
    if any(row_index in best_rows for row_index, _ in occurrences):
        return "bound"
    # the figure sits outside the bound row; accuse only if the bound row
    # OFFERS a different figure in one of the same columns
    occurrence_columns = {cell_index for _, cell_index in occurrences}
    (bound_row,) = best_rows
    for cell_index in occurrence_columns:
        if cell_index >= len(rows[bound_row]):
            continue
        offered = _cell_numbers(rows[bound_row][cell_index])
        if offered and not (offered & forms):
            return "misbound"
    return None


def check_row_binding(question: str, answer: str, chunks) -> list:
    """Diagnostics for figures bound to the wrong ROW of a retrieved table.

    ``chunks`` are the retrieval dictionaries (the same objects
    ``build_rag_context`` consumes); chunks without structured
    ``table_data`` are ignored, so plain-text contexts produce silence,
    not errors."""
    terms = _question_terms(question)
    if not terms:
        return []
    tables = list(_structured_tables(chunks))
    if not tables:
        return []
    misbound = []
    for token, forms in _answer_figures(answer):
        verdicts = [
            _table_verdict(headers, rows, terms, forms)
            for headers, rows in tables
        ]
        if "bound" in verdicts:
            continue  # some table places the figure in the bound row
        if "misbound" in verdicts:
            misbound.append(token)
    return [(WRONG_ROW_BINDING, tuple(sorted(misbound)))] if misbound else []
