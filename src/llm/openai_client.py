"""Thin OpenAI API client with retries and guardrails."""

from __future__ import annotations

import logging
import os
import time
from typing import Iterable, List, Sequence

from dotenv import load_dotenv
from openai import APIError, OpenAI, OpenAIError

from ..core.cost_guard import CostGuard, exponential_backoff

LOGGER = logging.getLogger(__name__)

load_dotenv()

DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
FALLBACK_CHAT_MODEL = "gpt-4o-mini-2024-07-18"
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = (
    "Answer strictly from provided context; if unknown, say you don't know; "
    "cite sources by filename and chunk id; be concise."
)


class OpenAIClient:
    def __init__(self, *, api_key: str | None = None) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            LOGGER.warning("OPENAI_API_KEY missing; client will fail on live calls")
        self.client = OpenAI(api_key=api_key)
        self.cost_guard = CostGuard.from_env()

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        model = EMBED_MODEL
        retries = exponential_backoff()
        last_error: OpenAIError | None = None
        for attempt in range(5):
            try:
                response = self.client.embeddings.create(model=model, input=list(texts))
                return [list(item.embedding) for item in response.data]
            except OpenAIError as exc:  # pragma: no cover - network path
                last_error = exc
                if attempt == 4:
                    break
                wait = next(retries)
                LOGGER.warning(
                    "Embedding request failed; retrying",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                time.sleep(wait)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to retrieve embeddings after retries")

    def chat(self, *, query: str, context_blocks: Iterable[str]) -> str:
        model = DEFAULT_CHAT_MODEL or FALLBACK_CHAT_MODEL
        prompt = self._build_prompt(context_blocks)
        self.cost_guard.enforce_budget(prompt=prompt, completion=query)
        retries = exponential_backoff()
        for attempt in range(5):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Query: {query}\n\nContext:\n{prompt}"},
                    ],
                    temperature=self.cost_guard.temperature,
                    max_tokens=self.cost_guard.max_tokens,
                    timeout=30,
                    extra_headers={"User-Agent": "policy-bot-poc"},
                )
                message = response.choices[0].message.content or ""
                self.cost_guard.after_success(tokens_used=len(message))
                return message
            except APIError as exc:  # pragma: no cover - network path
                self.cost_guard.after_failure(error=exc)
                wait = next(retries)
                LOGGER.warning(
                    "Chat request failed; retrying",
                    extra={"attempt": attempt + 1, "error": str(exc)},
                )
                time.sleep(wait)
        raise RuntimeError("Failed to obtain chat completion after retries")

    @staticmethod
    def _build_prompt(context_blocks: Iterable[str]) -> str:
        return "\n\n".join(context_blocks)
