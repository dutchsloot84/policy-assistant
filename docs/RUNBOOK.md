# Runbook

## Overview

This runbook covers day-to-day operations for the Policy Assistant POC running locally on a
developer laptop. All timestamps should use the America/Phoenix timezone for consistency.

## Services

- **API (FastAPI):** `make run-api` (port 8000)
- **UI (Streamlit):** `make run-ui` (port 8501)

## Logs

- FastAPI logs print to stdout. Use `LOG_TZ=America/Phoenix` in `.env` to keep timestamps
  aligned. Redirect to a file if long-running.
- Streamlit logs to the terminal session running `make run-ui`.

## Common operations

### Rebuilding the FAISS index

1. Stop the API/UI services.
2. Delete `data/index.faiss`, `data/meta.pkl`, and optionally `data/emb_cache.pkl`.
3. Restart the API and re-ingest policy PDFs.

### Deleting a document by filename

1. Stop the API.
2. Open `data/meta.pkl` with a Python shell and filter out entries whose `source` matches
   the filename.
3. Rebuild the FAISS index by removing the matching vectors (`data/index.faiss`) and
   re-ingest the remaining documents.

### Mocking OpenAI for demos/tests

- Set `OPENAI_API_KEY` to a dummy value.
- Override `OpenAIClient.chat` and `EmbeddingService.embed_query`/`embed_documents` in a
  wrapper script or via dependency injection to avoid live API calls (see `tests/test_api_query.py`).

### Handling API rate limits

- Rate limiter defaults to 2 RPS. Adjust `RATE_LIMIT_RPS` cautiously.
- Circuit breaker trips after repeated failures; wait for the reset timeout (default 60s) or
  restart the service to reset counters.

### Timezone considerations

- Ensure system timezone is set to America/Phoenix when interpreting logs or scheduling
  ingestion jobs.

## Incident response

- **OpenAI outages:** Circuit breaker will trip. Switch to cached answers or queue requests
  until service restoration.
- **Corrupt FAISS index:** Remove the index and metadata files, then re-ingest documents.
- **High costs:** Review logs for request counts, lower `TOP_K`, and ensure dedupe is working.

## Contact

- Engineering owner: Platform Engineering
- Escalation: Slack `#policy-assistant` channel or email platform-oncall@example.com
