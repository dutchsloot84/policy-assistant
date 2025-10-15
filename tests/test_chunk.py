from src.core.chunk import chunk_text


def test_chunk_simple_split():
    text = "Sentence one. Sentence two is a bit longer. Sentence three."
    chunks = chunk_text(text, max_chars=30, overlap=5)
    assert chunks
    # ensure overlap not exceed, and chunk ordering
    for chunk in chunks:
        assert len(chunk.text) <= 30
    assert chunks[0].text.startswith("Sentence one")


def test_chunk_handles_long_sentence():
    text = "A" * 2500
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) >= 5
    assert all(len(chunk.text) <= 500 for chunk in chunks)
