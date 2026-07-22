from pipeline.number_verify import is_financial, verify


def test_is_financial_only_monetary_values():
    # monetary / decimal values -> financial
    assert is_financial("12,34")
    assert is_financial("0,00")
    assert is_financial("-7,89")
    assert is_financial("1.234,56")
    # NOT financial: bare int (row index), year, month, date, thousands-only, text
    assert not is_financial("5")
    assert not is_financial("3000")
    assert not is_financial("2000-11")
    assert not is_financial("1.234")
    assert not is_financial("abc")


def test_verify_skips_row_indices_and_dates():
    headers = ["No", "Tutar", "Ay"]
    rows = [["1", "12,34", "2000-11"], ["2", "56,78", "2000-12"]]
    # OCR reading has the money but NOT the narrow index / date columns
    fidelity, flags = verify(headers, rows, "12,34 56,78")
    assert fidelity == 1.0            # both financial cells matched
    assert flags == []               # "1"/"2" and the dates are NOT flagged


def test_verify_flags_hallucinated_money_only():
    headers = ["No", "Tutar"]
    rows = [["1", "12,34"], ["2", "99,99"]]
    fidelity, flags = verify(headers, rows, "12,34")   # 99,99 absent from OCR
    assert len(flags) == 1 and flags[0][2] == "99,99"
    assert fidelity == 0.5
