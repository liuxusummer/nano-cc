from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import InboxMessage, TodoValidationError


class MessageBus:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else Path(".agent/teammates")

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def _ensure_root_dir(self) -> None:
        if self._root_dir.exists() and not self._root_dir.is_dir():
            raise TodoValidationError(
                "invalid_teammates_root_dir",
                "teammates root_dir is not a directory",
                {"path": str(self._root_dir)},
            )
        self._root_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, teammate_id: str) -> Path:
        # InboxMessage will validate id characters; we only map to filename.
        return self._root_dir / f"{teammate_id}.jsonl"

    def send(self, to: str, sender: str, content: str) -> dict[str, Any] | None:
        # Validate using InboxMessage to reuse field checks.
        msg = InboxMessage.from_mapping({"to": to, "sender": sender, "content": content})

        self._ensure_root_dir()
        inbox_path = self._inbox_path(msg.to)

        # Append-only JSONL; single write to reduce interleaving risk.
        line = json.dumps(msg.to_dict(), ensure_ascii=False)
        with inbox_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()

        return msg.to_dict()

    def read_inbox(self, teammate_id: str) -> dict[str, Any]:
        # Validate teammate_id via constructing InboxMessage mapping minimally
        try:
            InboxMessage.from_mapping({"to": teammate_id, "sender": "_", "content": "_"})
        except TodoValidationError as exc:
            raise TodoValidationError(
                "invalid_message_to",
                "invalid recipient id",
                {"to": teammate_id, "error": exc.to_dict()},
            ) from exc

        inbox_path = self._inbox_path(teammate_id)
        if not inbox_path.exists():
            legacy = self._legacy_inbox_path(teammate_id)
            if legacy is None or not legacy.exists():
                return {"messages": [], "stats": {"lines": 0, "parsed": 0, "skipped": 0}}
            inbox_path = legacy

        lines = 0
        parsed = 0
        skipped = 0
        messages: list[dict[str, Any]] = []

        with inbox_path.open("r", encoding="utf-8") as f:
            for raw in f:
                lines += 1
                text = raw.strip()
                if not text:
                    skipped += 1
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                try:
                    msg = InboxMessage.from_mapping(data)
                except TodoValidationError:
                    skipped += 1
                    continue
                messages.append(msg.to_dict())
                parsed += 1

        # Truncate inbox as atomically as possible: replace with empty file
        # Write temp file and atomically swap in place
        tmp_path = inbox_path.with_suffix(inbox_path.suffix + ".tmp")
        tmp_path.write_text("", encoding="utf-8")
        try:
            os.replace(tmp_path, inbox_path)
        finally:
            if tmp_path.exists():
                # Best-effort cleanup if replace fails
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

        return {"messages": messages, "stats": {"lines": lines, "parsed": parsed, "skipped": skipped}}

    def _legacy_inbox_path(self, teammate_id: str) -> Path | None:
        if self._root_dir.name != "teammates":
            return None
        if self._root_dir.parent.name != ".agent":
            return None
        project_root = self._root_dir.parent.parent
        return project_root / ".trae" / "teammates" / f"{teammate_id}.jsonl"
