from __future__ import annotations

import json
import sys
import types

import pytest


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class DummyStreamlit:
    def __init__(self) -> None:
        self.captured: dict[str, object] = {}
        self.sidebar = types.SimpleNamespace(
            header=lambda *a, **k: None,
            slider=lambda *a, **k: 3,
            checkbox=lambda *a, **k: False,
            text=lambda *a, **k: None,
        )

    def set_page_config(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        return None

    def title(self, *args, **kwargs) -> None:  # noqa: D401 - stub
        return None

    def tabs(self, labels):
        return (DummyContext(), DummyContext())

    def subheader(self, *args, **kwargs) -> None:
        return None

    def file_uploader(self, *args, **kwargs):
        return None

    def spinner(self, *args, **kwargs) -> DummyContext:
        return DummyContext()

    def success(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None

    def text_input(self, *args, **kwargs) -> str:
        return ""

    def button(self, *args, **kwargs) -> bool:
        return False

    def markdown(self, *args, **kwargs) -> None:
        return None

    def expander(self, *args, **kwargs) -> DummyContext:
        return DummyContext()

    def caption(self, *args, **kwargs) -> None:
        return None

    def json(self, value) -> None:
        self.captured["json"] = value

    def code(self, *args, **kwargs) -> None:
        return None

    def info(self, message) -> None:
        self.captured["info"] = message


class DummyRequests:
    def post(self, *args, **kwargs):
        raise AssertionError("Network should not be called in history test")


@pytest.fixture(autouse=True)
def patch_streamlit(monkeypatch):
    dummy = DummyStreamlit()
    module_name = "streamlit"
    original_requests = sys.modules.get("requests")
    monkeypatch.setitem(sys.modules, module_name, dummy)
    monkeypatch.setitem(sys.modules, "requests", DummyRequests())
    yield dummy
    sys.modules.pop(module_name, None)
    if original_requests is not None:
        sys.modules["requests"] = original_requests
    else:
        sys.modules.pop("requests", None)


def test_history_tab_loads_summary(monkeypatch, tmp_path, patch_streamlit):
    ledger_path = tmp_path / "ledger.jsonl"
    ingest_payload = json.dumps(
        {
            "kind": "ingest",
            "filename": "doc.pdf",
            "chunks": 1,
            "embed_batches": 1,
            "duration_ms": 1,
        }
    )
    query_payload = json.dumps(
        {
            "kind": "query",
            "query": "hi",
            "top_k": 1,
            "hits": [],
            "model": "gpt-4o-mini",
            "max_tokens": 300,
            "temperature": 0.2,
            "latency_ms": 1,
            "answer_chars": 0,
        }
    )
    ledger_path.write_text((ingest_payload + "\n" + query_payload + "\n"), encoding="utf-8")
    monkeypatch.setenv("HIST_LEDGER", str(ledger_path))

    import importlib

    module = importlib.import_module("src.ui.app_streamlit")

    assert patch_streamlit.captured["json"]["ingest_events"] == 1
    assert patch_streamlit.captured["json"]["query_events"] == 1
    assert "Ledger" not in patch_streamlit.captured.get("info", "")
    assert module is not None
