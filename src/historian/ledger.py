"""Append-only ledger writer with simple rotation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .schema import LedgerConfig

_BYTES_IN_MB = 1024 * 1024


class Ledger:
    """Manage an append-only JSONL ledger."""

    def __init__(self, cfg: LedgerConfig | None = None) -> None:
        self.cfg = cfg or LedgerConfig()
        self.path = Path(self.cfg.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate(self) -> None:
        """Rotate the ledger file when the size exceeds the configured limit."""
        if not self.path.exists():
            return

        max_bytes = self.cfg.rotate_mb * _BYTES_IN_MB
        if max_bytes <= 0:
            return

        current_size = self.path.stat().st_size
        if current_size < max_bytes:
            return

        index = 1
        while True:
            rotated = self.path.with_name(f"{self.path.stem}.r{index}{self.path.suffix}")
            if not rotated.exists():
                self.path.rename(rotated)
                break
            index += 1

    def append(self, obj: Mapping[str, Any]) -> None:
        """Append a JSON serializable mapping to the ledger."""
        self._rotate()
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(obj, handle, ensure_ascii=False)
            handle.write("\n")


__all__ = ["Ledger"]
