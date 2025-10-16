# Policy Assistant Chatbot POC

A 4-week proof of concept for a local-first Retrieval Augmented Generation (RAG) chatbot
that answers questions about policy PDFs. The system stores all data on disk, uses the
OpenAI API for embeddings and chat with conservative defaults, and can be lifted to AWS
with minimal changes.

## How it works

1. Upload policy PDFs via the Streamlit UI or API.
2. PDFs are parsed locally (pypdf with pdfminer fallback) and normalized.
3. Text is chunked (sentence-aware when available) with overlap for context continuity.
4. Chunks are deduplicated, embedded in batches via OpenAI `text-embedding-3-small`, and
   cached on disk.
5. Embeddings are stored in a local FAISS index alongside chunk metadata.
6. Queries embed locally, retrieve the top-3 chunks, and call `gpt-4o-mini` with a grounded
   prompt that demands citations. Answers include snippets and sources.
7. A cost guard enforces rate limits, exponential backoff, token budgets, and a simple
   circuit breaker to avoid runaway API usage.

### Table-aware chunking & field extractor

- PDF text is normalized with `normalize_for_chunking` so carriage returns become newlines,
  tabs become spaces, and table columns keep a hint of padding instead of collapsing into a
  single run of words.
- Default chunking now favours dense schedules: 550-character max chunks with 90-character
  overlap (still overridable via environment variables) to keep labels and numeric values in
  the same chunk.
- During ingest we run lightweight regexes to capture policy numbers, estimated total
  premiums, and inception premiums. These values are stored in chunk metadata, surfaced to the
  historian, and used to answer common questions without invoking the chat model.

## Cost controls

- Low-cost default models (`text-embedding-3-small`, `gpt-4o-mini`).
- Dedupe identical chunks before embedding and cache vectors on disk.
- Batched embedding requests (default batch size 64) with strict rate limiting.
- Per-chunk embedding guardrail configurable via `MAX_EMBED_TOKENS` (default 6000 tokens).
- Text chunking defaults adjustable via `CHUNK_MAX_CHARS` and `CHUNK_OVERLAP` (defaults 550/90).
- Temperature 0.2 and max tokens 300 for concise answers.
- Circuit breaker that halts requests after repeated failures.
- No secrets or sensitive data logged; basic PII redaction available on retrieved context.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI API key
make run-api
# in another terminal
make run-ui
```

Open http://localhost:8501 to access the Streamlit UI.

## Usage flow

1. Upload a policy PDF from the UI. Ingestion parses, chunks, embeds, and stores vectors.
2. Ask a question in the chat box. The system retrieves the top-k chunks and sends them to
   the OpenAI chat model with a grounded prompt.
3. The response includes citations by filename and chunk id. Expand the "Retrieved context"
   panel to see snippets and scores.

### Chunking configuration

- `CHUNK_MAX_CHARS` sets the upper bound for chunk size. Reducing it creates shorter,
  sentence-aware snippets that improve recall for dense or structured content such as policy
  tables, bullet lists, or numbered clauses where relevant details are tightly scoped.
- `CHUNK_OVERLAP` preserves trailing context between adjacent chunks. Increase it when
  important fields span multiple sentences; decrease it for highly structured text to avoid
  redundant embeddings.

Fine-tuning these values can increase retrieval quality for structured fields by ensuring the
retriever surfaces the precise clause or table row that matches a query instead of a large
composite chunk.

## Historian (Audit Trail)

- Every successful ingest and query is appended to a local JSONL ledger with timestamps,
  filenames, chunk counts, retrieval hits (including previews and scores), and the model
  parameters used for the call.
- The ledger lives at `data/historian/ledger.jsonl` by default and automatically rotates to
  `ledger.rN.jsonl` when it reaches the configured megabytes threshold (`HIST_ROTATE_MB`,
  default 10 MB).
- Open the **History** tab in the Streamlit UI to review a live summary and drill into the
  raw JSON events. The PowerShell helpers `hist-snapshot` and `hist-clear` provide quick CLI
  inspection and maintenance for Windows-first developers.
- Retrieval previews honour the `REDACT_PII` toggle, so enable `REDACT_PII=true` when logging
  sensitive content to keep ledger entries scrubbed.

## Troubleshooting

- **PDF is scanned or image-only:** Text extraction will be empty. Use OCR tools (e.g.,
  `ocrmypdf`) before ingesting.
- **Rate limit errors:** Increase `RATE_LIMIT_RPS` cautiously or wait and retry. The
  circuit breaker will reopen after the reset timeout.
- **Corporate TLS proxy / self-signed certs:** Set `OPENAI_CA_BUNDLE`,
  `REQUESTS_CA_BUNDLE`, or `SSL_CERT_FILE` in `.env` (or the process environment) to the
  path of your trusted PEM bundle before starting the API so the OpenAI client can verify
  TLS handshakes.
- **Empty answer:** Ensure the PDF produced text and that the query is relevant. Check the
  retrieved context for coverage.
- **Embeddings re-run on same PDF:** Ensure the file contents have not changed. The cache
  deduplicates based on text hash; re-ingesting identical content should be a no-op.

## AWS mapping

| Local Component | AWS Equivalent |
| --------------- | -------------- |
| Local disk storage (`data/`) | Amazon S3 or EFS for persistence |
| FAISS index | Amazon OpenSearch Serverless, Aurora PostgreSQL with pgvector, or Faiss on ECS/EKS |
| FastAPI app | AWS Lambda via API Gateway, AWS Fargate on ECS, or App Runner |
| Streamlit UI | Amplify Hosting, S3 + CloudFront, or containerized on ECS |
| `.env` secrets | AWS Secrets Manager or Systems Manager Parameter Store with KMS |
| Local network calls | VPC Endpoints + PrivateLink for API access |

Future lift-and-shift would mount S3 storage, replace FAISS with a managed vector DB, and
run the FastAPI/Streamlit containers on ECS/Fargate behind load balancers. Secrets migrate
to Secrets Manager with IAM roles.

## Cost tips

- Stick with the default small embedding model and only increase when necessary.
- Keep `TOP_K` small (default 3) to limit prompt size.
- Avoid re-ingesting unchanged PDFs; caching prevents duplicate embeddings.
- Batch embeddings via `EMBED_BATCH_SIZE` and respect rate limits.
- Monitor request counts and circuit breaker state via logs.

## Testing & quality

- `make test` runs pytest with coverage (≥70% on `src/`).
- `make lint` enforces ruff, black, and mypy.
- CI is configured via GitHub Actions (`.github/workflows/ci.yml`).

## Repository structure

```
policy-bot/
  src/ ...
  tests/ ...
  docs/ ...
  samples/README.md
```

See `docs/ARCHITECTURE.md` and `docs/RUNBOOK.md` for deeper details.
