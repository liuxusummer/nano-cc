from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class TranscriptRef:
    transcript_id: str
    entry: str

    def to_dict(self) -> dict[str, str]:
        return {"transcript_id": self.transcript_id, "entry": self.entry}


class TranscriptStore:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self._root_dir = Path(root_dir) if root_dir is not None else Path(".trae/transcripts")

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    def new_transcript_id(self) -> str:
        millis = int(time.time() * 1000)
        return f"{millis}_{uuid4().hex[:8]}"

    def path_for(self, transcript_id: str) -> Path:
        normalized = str(transcript_id).strip()
        if not normalized:
            raise ValueError("transcript_id is required")
        if "/" in normalized or "\\" in normalized or ".." in normalized:
            raise ValueError("invalid transcript_id")
        return self._root_dir / f"{normalized}.jsonl"

    def append(self, transcript_id: str, entry_type: str, payload: Any) -> TranscriptRef:
        path = self.path_for(transcript_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry_id = uuid4().hex
        record = {"ts": _utc_iso(), "type": str(entry_type), "payload": payload, "id": entry_id}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return TranscriptRef(transcript_id=transcript_id, entry=entry_id)

    def read_entries(self, transcript_id: str) -> list[dict[str, Any]]:
        path = self.path_for(transcript_id)
        if not path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
        return entries
