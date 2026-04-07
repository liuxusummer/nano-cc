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
class SessionRef:
    session_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session: SessionRef
    updated_at: str
    entry_count: int
    last_input: str


class SessionStore:
    def __init__(self, root_dir: str | Path) -> None:
        self._root_dir = Path(root_dir)

    @property
    def sessions_dir(self) -> Path:
        return self._root_dir / ".agent" / "cli" / "sessions"

    @property
    def _legacy_sessions_dir(self) -> Path:
        return self._root_dir / ".trae" / "cli" / "sessions"

    @property
    def task_graph_sessions_dir(self) -> Path:
        return self._root_dir / ".agent" / "task_graph" / "sessions"

    @property
    def legacy_task_graph_dir(self) -> Path:
        return self._root_dir / ".agent" / "task_graph"

    def new_session(self) -> SessionRef:
        millis = int(time.time() * 1000)
        session_id = f"{millis}_{uuid4().hex[:8]}"
        path = self.sessions_dir / f"{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        return SessionRef(session_id=session_id, path=path)

    def append_round(self, session: SessionRef, payload: dict[str, Any]) -> None:
        record = {"ts": _utc_iso(), **payload}
        session.path.parent.mkdir(parents=True, exist_ok=True)
        with session.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def find_last_session(self) -> SessionRef | None:
        candidates: list[Path] = []
        for directory in (self.sessions_dir, self._legacy_sessions_dir):
            if not directory.exists():
                continue
            try:
                candidates.extend(list(directory.glob("*.jsonl")))
            except Exception:
                continue
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        path = candidates[0]
        return SessionRef(session_id=path.stem, path=path)

    def get_session(self, session_id: str) -> SessionRef | None:
        normalized = str(session_id).strip()
        if not normalized or "/" in normalized or "\\" in normalized or ".." in normalized:
            return None
        for directory in (self.sessions_dir, self._legacy_sessions_dir):
            path = directory / f"{normalized}.jsonl"
            if path.exists():
                return SessionRef(session_id=normalized, path=path)
        return None

    def read_entries(self, session: SessionRef) -> list[dict[str, Any]]:
        if not session.path.exists():
            return []
        entries: list[dict[str, Any]] = []
        for line in session.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                entries.append(loaded)
        return entries

    def task_graph_root(self, session: SessionRef | None) -> Path:
        if session is None:
            return self.task_graph_sessions_dir / "__missing__"
        return self.task_graph_sessions_dir / session.session_id

    def resolve_task_graph_root(self, session: SessionRef | None, *, allow_legacy_fallback: bool = False) -> Path:
        root = self.task_graph_root(session)
        if root.exists() or not allow_legacy_fallback:
            return root
        return self.legacy_task_graph_dir

    def list_sessions(self, limit: int | None = None) -> list[SessionRef]:
        candidates: list[Path] = []
        for directory in (self.sessions_dir, self._legacy_sessions_dir):
            if not directory.exists():
                continue
            try:
                candidates.extend(list(directory.glob("*.jsonl")))
            except Exception:
                continue
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        refs = [SessionRef(session_id=path.stem, path=path) for path in candidates]
        if limit is not None and limit >= 0:
            return refs[:limit]
        return refs

    def list_session_summaries(self, limit: int | None = None) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for session in self.list_sessions(limit=limit):
            entries = self.read_entries(session)
            last_input = ""
            for entry in reversed(entries):
                raw_input = str(entry.get("raw_input", "")).strip()
                user_input = str(entry.get("user_input", "")).strip()
                candidate = raw_input or user_input
                if candidate:
                    last_input = candidate
                    break
            try:
                stat = session.path.stat()
                updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            except Exception:
                updated_at = ""
            summaries.append(
                SessionSummary(
                    session=session,
                    updated_at=updated_at,
                    entry_count=len(entries),
                    last_input=last_input,
                )
            )
        return summaries
