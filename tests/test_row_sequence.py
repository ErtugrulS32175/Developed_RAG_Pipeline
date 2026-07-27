"""A table that numbers its own rows can prove its row count is wrong."""
from pipeline.row_sequence import check, find_index_column


def _rows(indices, width=3):
    """One row per index; an index of None leaves that cell blank."""
    return [[("" if i is None else str(i))] + ["x"] * (width - 1) for i in indices]


# --- detection ---

def test_finds_a_serial_column():
    assert find_index_column(_rows([1, 2, 3, 4])) == 0


def test_ignores_a_table_without_one():
    rows = [["a", "10,00"], ["b", "20,00"], ["c", "30,00"]]
    assert find_index_column(rows) is None


def test_ignores_a_column_that_does_not_start_near_one():
    """Years ascend too, but they are not a row count."""
    rows = [[str(y), "x"] for y in (2019, 2020, 2021, 2022)]
    assert find_index_column(rows) is None


def test_ignores_a_descending_column():
    assert find_index_column(_rows([1, 5, 3, 4])) is None


def test_ignores_too_short_a_run():
    assert find_index_column(_rows([1, 2])) is None


# --- what it reports ---

def test_clean_sequence_reports_nothing():
    issues, cells = check(_rows([1, 2, 3, 4, 5]))
    assert issues == []
    assert cells == set()


def test_trailing_unnumbered_row_is_allowed():
    """A totals row carries no serial and must not be flagged."""
    issues, cells = check(_rows([1, 2, 3, 4, None]))
    assert issues == []
    assert cells == set()


def test_unnumbered_row_between_numbered_ones_is_flagged():
    """The observed defect: a model invented a row and had no number for it."""
    issues, cells = check(_rows([1, None, 2, 3, 4]))
    assert len(issues) == 1
    assert "sira numarasi yok" in issues[0]
    assert (1, 0) in cells


def test_gap_in_the_sequence_reports_missing_rows():
    issues, cells = check(_rows([1, 2, 5, 6]))
    assert any("2 ile 5 arasinda 2 satir eksik" in i for i in issues)
    assert (2, 0) in cells


def test_repeated_serial_reports_a_duplicated_row():
    issues, cells = check(_rows([1, 2, 2, 3]))
    assert any("iki kez geciyor" in i for i in issues)
    assert (2, 0) in cells and (1, 0) in cells


def test_sequence_not_starting_at_one_is_flagged():
    issues, _ = check(_rows([2, 3, 4, 5]))
    assert any("ile basliyor" in i for i in issues)


def test_no_serial_column_means_no_findings():
    issues, cells = check([["a", "1,00"], ["b", "2,00"], ["c", "3,00"]])
    assert issues == [] and cells == set()


def test_empty_table_is_safe():
    assert check([]) == ([], set())
