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

### Rotate or clear the historian ledger

1. Confirm the target file via `echo $HIST_LEDGER` (defaults to `data/historian/ledger.jsonl`).
2. Run `pwsh -File ./scripts/Dev.ps1 hist-clear` to remove the active ledger. Rotation will
   automatically rename the file to `ledger.rN.jsonl` when it grows beyond `HIST_ROTATE_MB`.
3. Restart ingestion/query flows; a fresh ledger file is created on the next successful event.

### Create a CSV/summary snapshot

1. Ensure the ledger exists (ingest/query at least once).
2. Generate a JSON summary via `pwsh -File ./scripts/Dev.ps1 hist-snapshot` or by running:

   ```bash
   python - <<'PY'
   from historian.export import summarize
   import os
   path = os.getenv('HIST_LEDGER', 'data/historian/ledger.jsonl')
   print(summarize(path))
   PY
   ```

3. For spreadsheet analysis, convert the JSONL into CSV with the standard library:

   ```bash
   python - <<'PY'
   import csv
   import json
   import os

   path = os.getenv('HIST_LEDGER', 'data/historian/ledger.jsonl')
   with open(path, 'r', encoding='utf-8') as handle:
       rows = [json.loads(line) for line in handle if line.strip()]

   if rows:
       fieldnames = sorted({key for row in rows for key in row})
       with open('ledger_export.csv', 'w', encoding='utf-8', newline='') as out:
           writer = csv.DictWriter(out, fieldnames=fieldnames)
           writer.writeheader()
           writer.writerows(rows)
       print('Wrote ledger_export.csv with', len(rows), 'rows')
   else:
       print('No ledger rows to export')
   PY
   ```

### Confirm field extraction

1. After ingesting a policy, open the **History** tab in the Streamlit UI and locate the
   matching ingest event. A `Note` marker reading `Extracted fields: …` confirms regex hits.
2. Run `pwsh -File ./scripts/Dev.ps1 hist-snapshot` to export the latest events and verify the
   structured values without opening the JSONL manually.
3. If regex patterns are updated, remove the affected document from the FAISS store (see
   "Rebuilding the FAISS index") and re-upload the PDF so fresh metadata and historian notes
   are captured.

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
