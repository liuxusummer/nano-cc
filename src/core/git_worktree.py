from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models import TodoValidationError


@dataclass(slots=True)
class GitRunResult:
    returncode: int
    stdout: str
    stderr: str


GitRunner = Callable[[list[str], Path | None], GitRunResult]


def _default_runner(args: list[str], cwd: Path | None) -> GitRunResult:
    completed = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True, capture_output=True, check=False)
    return GitRunResult(returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


class GitWorktree:
    def __init__(self, *, runner: GitRunner | None = None) -> None:
        self._runner = runner or _default_runner

    def add(
        self,
        repo_dir: str | Path,
        worktree_dir: str | Path,
        *,
        name: str,
        base_ref: str | None = None,
        branch: str | None = None,
        force: bool = False,
    ) -> GitRunResult:
        repo = Path(repo_dir)
        target = Path(worktree_dir)
        normalized_name = str(name).strip()
        if not str(repo).strip():
            raise TodoValidationError("invalid_repo_dir", "repo_dir is required")
        if not str(target).strip():
            raise TodoValidationError("invalid_worktree_dir", "worktree_dir is required")
        if not normalized_name:
            raise TodoValidationError("invalid_worktree_name", "name is required")
        args: list[str] = ["git", "-C", str(repo), "worktree", "add"]
        if force:
            args.append("--force")
        branch_name = str(branch).strip() if branch is not None else ""
        base_ref_name = str(base_ref).strip() if base_ref is not None else ""
        if branch_name:
            args.extend(["-b", branch_name])
        args.append(str(target))
        if base_ref_name:
            args.append(base_ref_name)
        result = self._runner(args, cwd=repo)
        if result.returncode != 0:
            raise TodoValidationError(
                "git_worktree_add_failed",
                "git worktree add failed",
                {"args": args, "code": result.returncode, "stderr": result.stderr},
            )
        return result

    def remove(
        self,
        repo_dir: str | Path,
        worktree_dir: str | Path,
        *,
        force: bool = False,
    ) -> bool:
        repo = Path(repo_dir)
        target = Path(worktree_dir)
        if not str(repo).strip():
            raise TodoValidationError("invalid_repo_dir", "repo_dir is required")
        if not str(target).strip():
            raise TodoValidationError("invalid_worktree_dir", "worktree_dir is required")
        args: list[str] = ["git", "-C", str(repo), "worktree", "remove", str(target)]
        if force:
            args.append("--force")
        result = self._runner(args, cwd=repo)
        if result.returncode != 0:
            raise TodoValidationError(
                "git_worktree_remove_failed",
                "git worktree remove failed",
                {"args": args, "code": result.returncode, "stderr": result.stderr},
            )
        return True
