from __future__ import annotations

from pathlib import Path

from .models import TodoValidationError


class SkillsEngine:
    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path("doc/skills")

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def list_skills(self) -> list[str]:
        if not self._base_dir.exists():
            return []
        if not self._base_dir.is_dir():
            raise TodoValidationError(
                "invalid_skills_base_dir",
                "skills base_dir is not a directory",
                {"path": str(self._base_dir)},
            )
        skills: list[str] = []
        for path in self._base_dir.glob("*.md"):
            if path.is_file():
                skills.append(path.stem)
        return sorted(set(skills))

    def load_skill(self, name: str) -> str:
        normalized = str(name).strip()
        if not normalized:
            raise TodoValidationError("invalid_skill_name", "skill name is required", {"name": name})
        if "/" in normalized or "\\" in normalized or ".." in normalized:
            raise TodoValidationError(
                "invalid_skill_name",
                "skill name contains invalid characters",
                {"name": normalized},
            )

        path = self._base_dir / f"{normalized}.md"
        if not path.exists():
            raise TodoValidationError(
                "skill_not_found",
                "skill not found",
                {"name": normalized, "path": str(path)},
            )
        if not path.is_file():
            raise TodoValidationError(
                "skill_not_found",
                "skill path is not a file",
                {"name": normalized, "path": str(path)},
            )
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise TodoValidationError(
                "skill_unreadable",
                "skill cannot be read",
                {"name": normalized, "path": str(path), "error": str(exc)},
            ) from exc
