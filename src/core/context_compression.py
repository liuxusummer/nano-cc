from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .model_provider import ModelProvider, PlanState
from .transcript_store import TranscriptStore, TranscriptRef


@dataclass(frozen=True, slots=True)
class CompactResult:
    summary: str
    ref: TranscriptRef
    replacement_trace: list[dict[str, Any]]

    def to_tool_output(self) -> dict[str, Any]:
        return {"tool": "compact", "ref": self.ref.to_dict(), "summary": self.summary}


class ContextCompression:
    def __init__(
        self,
        transcript_store: TranscriptStore,
        *,
        model_provider: ModelProvider | None = None,
        token_threshold: int | None = None,
    ) -> None:
        self._transcript_store = transcript_store
        self._model_provider = model_provider
        self._token_threshold = token_threshold

    @property
    def token_threshold(self) -> int:
        if self._token_threshold is not None:
            return self._token_threshold
        env = os.getenv("TRAE_COMPACT_TOKEN_THRESHOLD", "").strip()
        if env:
            try:
                parsed = int(env)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return 8000

    def estimate_tokens(self, value: Any) -> int:
        try:
            serialized = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            serialized = str(value)
        chars = len(serialized)
        return (chars + 3) // 4

    def should_auto_compact(self, trace: list[dict[str, Any]]) -> bool:
        return self.estimate_tokens(trace) > self.token_threshold

    def micro_compact(self, trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
        last_tool_idx = -1
        for i in range(len(trace) - 1, -1, -1):
            step = trace[i].get("step")
            if isinstance(step, dict) and step.get("type") == "tool":
                last_tool_idx = i
                break

        if last_tool_idx <= 0:
            return trace

        updated: list[dict[str, Any]] = []
        for i, item in enumerate(trace):
            if i >= last_tool_idx:
                updated.append(item)
                continue
            step = item.get("step")
            if not isinstance(step, dict) or step.get("type") != "tool":
                updated.append(item)
                continue
            observation = item.get("observation")
            if not isinstance(observation, dict):
                updated.append(item)
                continue
            result = observation.get("result")
            if not isinstance(result, dict):
                updated.append(item)
                continue
            output = result.get("output")
            if isinstance(output, dict) and output.get("__compact__") is True:
                updated.append(item)
                continue

            ref = observation.get("ref")
            if not isinstance(ref, dict) or not str(ref.get("transcript_id", "")).strip():
                updated.append(item)
                continue
            placeholder = {"__compact__": True, "kind": "tool_result", "ref": dict(ref)}
            compacted = dict(item)
            compacted_observation = dict(observation)
            compacted_result = dict(result) if isinstance(result, dict) else {}
            compacted_result["output"] = placeholder
            compacted_observation["result"] = compacted_result
            compacted["observation"] = compacted_observation
            updated.append(compacted)
        return updated

    def auto_compact(self, transcript_id: str, trace: list[dict[str, Any]]) -> CompactResult:
        return self._compact(transcript_id, trace, reason="auto")

    def manual_compact(self, transcript_id: str, trace: list[dict[str, Any]], reason: str | None = None) -> CompactResult:
        return self._compact(transcript_id, trace, reason=str(reason or "manual").strip() or "manual")

    def _compact(self, transcript_id: str, trace: list[dict[str, Any]], reason: str) -> CompactResult:
        summary = self._summarize_transcript(transcript_id)
        ref = self._transcript_store.append(
            transcript_id,
            "compact",
            {"type": "compact_summary", "summary": summary, "reason": reason, "ref": {"transcript_id": transcript_id}},
        )
        replacement_trace = [
            {
                "step": {
                    "type": "compact_summary",
                    "summary": summary,
                    "ref": {"transcript_id": transcript_id},
                    "reason": reason,
                },
                "observation": {},
            }
        ]
        return CompactResult(summary=summary, ref=ref, replacement_trace=replacement_trace)

    def _summarize_transcript(self, transcript_id: str) -> str:
        entries = self._transcript_store.read_entries(transcript_id)
        if self._model_provider is None:
            return self._fallback_summary(entries)
        state: PlanState = {
            "goal": "generate compact summary for transcript",
            "trace": [],
            "observation": {"transcript_id": transcript_id, "entries": entries},
            "constraints": {"purpose": "compact_summary", "only_return_json": True},
        }
        action = self._model_provider.plan_next(state)
        if isinstance(action, dict) and str(action.get("type", "")).strip() == "answer":
            content = str(action.get("content", "")).strip()
            if content:
                return content
        return self._fallback_summary(entries)

    def _fallback_summary(self, entries: list[dict[str, Any]]) -> str:
        if not entries:
            return "no transcript entries"
        tail = entries[-20:]
        lines: list[str] = []
        for item in tail:
            entry_type = str(item.get("type", "")).strip() or "unknown"
            payload = item.get("payload")
            try:
                payload_text = json.dumps(payload, ensure_ascii=False, default=str)
            except TypeError:
                payload_text = str(payload)
            payload_text = payload_text[:200]
            lines.append(f"{entry_type}: {payload_text}")
        return "\n".join(lines)
