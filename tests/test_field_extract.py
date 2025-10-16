from src.core.field_extract import extract_fields


def test_extract_fields_detects_all_values():
    sample = (
        "POLICY NUMBER: NCBA330004911965\n"
        "Estimated Total Premium\n"
        "$ 299,997.00\n"
        "Premium shown is payable at inception:\n"
        "$ 150,000.00\n"
    )

    result = extract_fields(sample)

    assert result["policy_number"] == "NCBA330004911965"
    assert result["estimated_total_premium"] == "$ 299,997.00"
    assert result["premium_at_inception"] == "$ 150,000.00"


def test_extract_fields_handles_missing():
    assert extract_fields("") == {}
