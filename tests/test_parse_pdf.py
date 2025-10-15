import src.core.parse_pdf as parse_pdf
from src.core.parse_pdf import extract_text_from_pdf


def _make_pdf_bytes(text: str) -> bytes:
    header = b"%PDF-1.4\n"
    objects = [
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n",
        b"2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj\n",
        (
            b"3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>endobj\n"
        ),
    ]
    stream = f"BT /F1 12 Tf 72 72 Td ({text}) Tj ET".encode("utf-8")
    objects.append(
        b"4 0 obj<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>stream\n"
        + stream
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n")

    body = b""
    offsets = []
    current = len(header)
    for obj in objects:
        offsets.append(current)
        body += obj
        current += len(obj)

    xref = b"xref\n0 " + str(len(objects) + 1).encode("ascii") + b"\n"
    xref += b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode("ascii")
    trailer = b"trailer<< /Root 1 0 R /Size " + str(len(objects) + 1).encode("ascii") + b" >>\n"
    startxref = len(header) + len(body)
    eof = b"startxref\n" + str(startxref).encode("ascii") + b"\n%%EOF"
    return header + body + xref + trailer + eof


def test_extract_text_from_pdf():
    pdf_bytes = _make_pdf_bytes("Policy Document")
    text = extract_text_from_pdf(pdf_bytes)
    assert "Policy Document" in text


def test_extract_text_empty_bytes():
    assert extract_text_from_pdf(b"") == ""


def test_extract_text_pdfminer_fallback(monkeypatch):
    monkeypatch.setattr(parse_pdf, "_extract_with_pypdf", lambda _bytes: "")
    monkeypatch.setattr(parse_pdf, "_extract_with_pdfminer", lambda _bytes: " fallback text ")

    text = extract_text_from_pdf(b"pdf-bytes", filename="document.pdf")

    assert text == "fallback text"
