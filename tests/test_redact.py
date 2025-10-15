from src.core.redact import redact_text


def test_redact_patterns():
    text = "Contact us at person@example.com or 555-123-4567. Policy 1234-5678 at 10 Main Street."
    redacted = redact_text(text, enabled=True)
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_PHONE]" in redacted
    assert "[REDACTED_POLICY_ID]" in redacted
    assert "[REDACTED_ADDRESS]" in redacted


def test_redaction_disabled():
    text = "Email: person@example.com"
    assert redact_text(text, enabled=False) == text
