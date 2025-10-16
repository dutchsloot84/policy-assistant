import importlib

from src.core.chunk import Chunk, chunk_text, format_page_label, map_offsets_to_page_range


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


def test_chunk_env_defaults(monkeypatch):
    from src.core import chunk as chunk_module

    monkeypatch.setenv("CHUNK_MAX_CHARS", "42")
    monkeypatch.setenv("CHUNK_OVERLAP", "7")
    importlib.reload(chunk_module)

    assert chunk_module.DEFAULT_MAX_CHARS == 42
    assert chunk_module.DEFAULT_OVERLAP == 7

    monkeypatch.delenv("CHUNK_MAX_CHARS", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP", raising=False)
    importlib.reload(chunk_module)


def test_map_offsets_to_page_range_handles_boundaries():
    page_breaks = [0, 14, 42]

    first = Chunk(id="a", text="", start=0, end=10)
    assert map_offsets_to_page_range(first, page_breaks) == (1, 1)

    second = Chunk(id="b", text="", start=16, end=30)
    assert map_offsets_to_page_range(second, page_breaks) == (2, 2)

    spanning = Chunk(id="c", text="", start=10, end=50)
    assert map_offsets_to_page_range(spanning, page_breaks) == (1, 3)


def test_format_page_label():
    assert format_page_label(None, None) == ""
    assert format_page_label(1, 1) == "Page 1"
    assert format_page_label(2, 3) == "Pages 2–3"
