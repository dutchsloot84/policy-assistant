"""Streamlit UI for the policy assistant POC."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import requests  # type: ignore[import-untyped]
import streamlit as st
from dotenv import load_dotenv

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.historian.export import summarize
else:  # pragma: no cover - runtime import resolution
    from ..historian.export import summarize

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _format_page_label(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return ""
    if page_end is None or page_end == page_start:
        return f"Page {page_start}"
    return f"Pages {page_start}–{page_end}"

st.set_page_config(page_title="Policy Assistant", layout="wide")
st.title("Policy Assistant Chatbot")

chat_tab, history_tab = st.tabs(["Chat", "History"])

with chat_tab:
    st.sidebar.header("Settings")
    default_top_k = int(os.getenv("TOP_K", "3"))
    selected_top_k = st.sidebar.slider("Top K", min_value=1, max_value=10, value=default_top_k)
    redact_toggle = st.sidebar.checkbox(
        "Redact PII", value=os.getenv("REDACT_PII", "true").lower() in {"true", "1"}
    )
    st.sidebar.text(f"Model: {os.getenv('CHAT_MODEL', 'gpt-4o-mini')}")

    st.subheader("Upload policy PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
    if uploaded_file is not None:
        with st.spinner("Uploading and ingesting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(f"{API_URL}/ingest", files=files, timeout=60)
        if response.ok:
            chunk_count = response.json().get("chunks")
            elapsed = response.json().get("elapsed_sec")
            st.success(f"Ingested {chunk_count} chunks in {elapsed}s")
        else:
            st.error(f"Failed to ingest: {response.text}")

    st.subheader("Ask a question")
    query_text = st.text_input("Enter your question")
    if st.button("Submit") and query_text:
        payload = {"query": query_text, "top_k": selected_top_k, "redact": redact_toggle}
        with st.spinner("Thinking..."):
            resp = requests.post(f"{API_URL}/query", json=payload, timeout=60)
        if resp.ok:
            data: Dict[str, Any] = resp.json()
            st.markdown(f"**Answer:** {data.get('answer', 'No response')}")
            st.markdown("---")
            with st.expander("Retrieved context"):
                sources = data.get("sources", [])
                snippets = data.get("snippets", [])
                for source, snippet in zip(sources, snippets, strict=False):
                    label = source["source"]
                    chunk_id = source["chunk_id"]
                    score = source["score"]
                    page_label = _format_page_label(source.get("page_start"), source.get("page_end"))
                    if page_label:
                        title = f"**{label} ({page_label}, chunk {chunk_id})**"
                    else:
                        title = f"**{label} (chunk {chunk_id})**"
                    st.markdown(f"{title} — score {score:.4f}")
                    st.caption(snippet)
        else:
            st.error(f"Query failed: {resp.text}")

with history_tab:
    ledger_path = Path(os.getenv("HIST_LEDGER", "data/historian/ledger.jsonl"))
    st.subheader("Historian Summary")
    if ledger_path.exists():
        summary = summarize(ledger_path)
        st.json(summary)
        with st.expander("Raw events"):
            with ledger_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    st.code(line, language="json")
    else:
        st.info("Ledger not created yet. Run an ingest or query to populate history.")
