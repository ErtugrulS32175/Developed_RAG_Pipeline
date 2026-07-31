"""Row-count verification from a table's own serial-number column.

Both directions of row miscounting have been seen from VLMs on dense tables: a
run of identical records read one row SHORT, and a run read one row LONG. Both
are silent -- every extracted value is correct, only the number of rows is
wrong -- so neither the number-fidelity check (which sees no bad digits) nor
model consensus (both models can miscount the same way) notices.

A form that numbers its rows carries the answer inside itself. This module finds
such a column and reports what it contradicts: a gap means a row was dropped, a
repeat means one was duplicated, and an unnumbered row between numbered ones
means a row was invented. No external knowledge of the expected row count is
needed -- the sequence states it.

Detection is deliberately conservative: a table without a serial column produces
no findings at all rather than guesses. Nothing is ever corrected here; findings
are flagged for a human, like every other check in the pipeline.
"""
import re

_INT_RE = re.compile(r"^\d{1,4}$")
# Below this the "sequence" is too short to distinguish from an ordinary numeric
# column that happens to ascend.
MIN_INDEXED_ROWS = 3
# Fraction of rows that must carry a number for the column to be an index at all.
MIN_COVERAGE = 0.6


def _as_int(cell):
    s = str(cell).strip()
    return int(s) if _INT_RE.match(s) else None


def find_index_column(rows):
    """Column holding a serial number, or None.

    Wants a column that ascends from the top of the table and covers most rows.
    Ascending is checked non-strictly so a DUPLICATED row -- the very defect this
    module exists to catch -- doesn't disqualify the column it shows up in.
    """
    if not rows:
        return None
    width = max(len(r) for r in rows)
    for col in range(width):
        values = [_as_int(r[col]) if col < len(r) else None for r in rows]
        nums = [v for v in values if v is not None]
        if len(nums) < MIN_INDEXED_ROWS or len(nums) / len(rows) < MIN_COVERAGE:
            continue
        # a serial column starts at the beginning and never descends
        if nums[0] > 2 or any(b < a for a, b in zip(nums, nums[1:])):
            continue
        return col
    return None


def check(rows):
    """(issues, cells) for the serial column, if the table has one.

    `cells` are (row, col) positions to highlight, so a reviewer lands on the
    exact row the sequence complains about instead of scanning the table.
    """
    col = find_index_column(rows)
    if col is None:
        return [], set()

    values = [_as_int(r[col]) if col < len(r) else None for r in rows]
    numbered = [i for i, v in enumerate(values) if v is not None]
    issues, cells = [], set()

    # A trailing unnumbered row is normal -- totals and footers carry no serial.
    # An unnumbered row BETWEEN numbered ones is a row the form never had.
    last = numbered[-1]
    for i in range(numbered[0], last):
        if values[i] is None:
            issues.append(f"satir {i + 1}: sira numarasi yok "
                          f"(fazladan satir olabilir)")
            cells.add((i, col))

    seen = {}
    for i in numbered:
        v = values[i]
        if v in seen:
            issues.append(f"sira no {v} iki kez geciyor "
                          f"(satir {seen[v] + 1} ve {i + 1} - satir tekrarlanmis olabilir)")
            cells.add((i, col))
            cells.add((seen[v], col))
        else:
            seen[v] = i

    ordered = [(i, values[i]) for i in numbered]
    for (pi, pv), (ni, nv) in zip(ordered, ordered[1:]):
        if nv > pv + 1:
            missing = nv - pv - 1
            issues.append(f"sira no {pv} ile {nv} arasinda {missing} satir eksik")
            cells.add((ni, col))

    if values[numbered[0]] > 1:
        issues.append(f"sira numarasi {values[numbered[0]]} ile basliyor "
                       "(bastaki satirlar eksik olabilir)")
        cells.add((numbered[0], col))

    return issues, cells
