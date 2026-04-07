from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EnvLoadResult:
    loaded: bool
    path: Path
    keys: list[str]


def _strip_quotes(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        return text[1:-1]
    return text


def parse_env_lines(lines: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        parsed[key] = _strip_quotes(value)
    return parsed


def load_env_file(path: Path, *, override: bool = False) -> EnvLoadResult:
    normalized = Path(path).expanduser()
    if not normalized.exists() or not normalized.is_file():
        return EnvLoadResult(loaded=False, path=normalized, keys=[])
    content = normalized.read_text(encoding="utf-8")
    parsed = parse_env_lines(content.splitlines())
    applied: list[str] = []
    for k, v in parsed.items():
        if not override and k in __import__("os").environ:
            continue
        __import__("os").environ[k] = v
        applied.append(k)
    return EnvLoadResult(loaded=True, path=normalized, keys=sorted(applied))

