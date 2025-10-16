from src.core.parse_pdf import normalize_for_chunking


def test_normalize_for_chunking_preserves_table_spacing():
    raw_text = "Column\tAmount\r\nPolicy   Value\r\nTotal\t$ 299,997.00\r\r\n"
    normalized = normalize_for_chunking(raw_text)

    assert "Column Amount" in normalized
    assert "Policy  Value" in normalized  # triple spaces collapsed to two
    assert "$ 299,997.00" in normalized
    assert "\n\n" in normalized  # capped blank lines
    assert not normalized.startswith("\n")
    assert not normalized.endswith("\n")
