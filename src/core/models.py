from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Callable, Mapping


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class TaskStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class BackgroundTaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TodoValidationError(ValueError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


@dataclass(slots=True)
class TaskGraphItem:
    id: str
    content: str
    status: TaskStatus = TaskStatus.PENDING
    blockedBy: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    owner: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TaskGraphItem:
        task_id = str(data.get("id", "")).strip()
        if not task_id:
            raise TodoValidationError("invalid_task_id", "task id is required")
        if "/" in task_id or "\\" in task_id or ".." in task_id:
            raise TodoValidationError("invalid_task_id", "task id contains invalid characters", {"id": task_id})

        content = str(data.get("content", "")).strip()
        if not content:
            raise TodoValidationError("invalid_task_content", "task content is required", {"id": task_id})

        status_raw = data.get("status", TaskStatus.PENDING)
        try:
            if isinstance(status_raw, str):
                normalized_status = status_raw.strip().lower()
                normalized_status = normalized_status.replace("-", "_").replace(" ", "_")
                normalized_status = {
                    "ready": "pending",
                    "todo": "pending",
                    "doing": "in_progress",
                    "inprogress": "in_progress",
                    "done": "completed",
                    "finished": "completed",
                }.get(normalized_status, normalized_status)
                status = TaskStatus(normalized_status)
            else:
                status_value = str(status_raw)
                status = status_raw if isinstance(status_raw, TaskStatus) else TaskStatus(status_value)
        except ValueError as exc:
            raise TodoValidationError(
                "invalid_task_status",
                f"invalid task status: {status_raw}",
                {"id": task_id},
            ) from exc

        blocked_by_raw = data.get("blockedBy", data.get("blocked_by", []))
        blocks_raw = data.get("blocks", [])
        owner_raw = data.get("owner")

        blocked_by = _normalize_id_list(blocked_by_raw, field_name="blockedBy", task_id=task_id)
        blocks = _normalize_id_list(blocks_raw, field_name="blocks", task_id=task_id)

        owner = None
        if owner_raw is not None:
            owner_str = str(owner_raw).strip()
            owner = owner_str or None

        return cls(id=task_id, content=content, status=status, blockedBy=blocked_by, blocks=blocks, owner=owner)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status.value,
            "blockedBy": list(self.blockedBy),
            "blocks": list(self.blocks),
            "owner": self.owner if self.owner is not None else None,
        }


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolResult:
    success: bool
    output: Any = None
    error: Any = None


ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackgroundTaskRecord:
    task_id: str
    status: str
    created_at: datetime = field(default_factory=utc_now)
    command: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    exit_code: int | None = None
    output_tail: str | None = None
    error_tail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "command": self.command,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "exit_code": self.exit_code,
            "output_tail": self.output_tail,
            "error_tail": self.error_tail,
        }


@dataclass(slots=True)
class BackgroundTaskNotification:
    task_id: str
    status: str
    finished_at: datetime = field(default_factory=utc_now)
    exit_code: int | None = None
    output_tail: str | None = None
    error_tail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "finished_at": self.finished_at.isoformat(),
            "exit_code": self.exit_code,
            "output_tail": self.output_tail,
            "error_tail": self.error_tail,
        }


def _normalize_id_list(value: Any, field_name: str, task_id: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise TodoValidationError(
            "invalid_task_dependencies",
            f"{field_name} must be an array of strings",
            {"id": task_id, "field": field_name},
        )
    normalized: list[str] = []
    for raw in value:
        item = str(raw).strip()
        if not item:
            raise TodoValidationError(
                "invalid_task_dependencies",
                f"{field_name} contains empty id",
                {"id": task_id, "field": field_name},
            )
        normalized.append(item)
    return normalized


class TeammateStatus(StrEnum):
    ACTIVE = "active"
    INACTIVE = "inactive"


@dataclass(slots=True)
class TeammateRecord:
    id: str
    email: str
    name: str
    status: TeammateStatus = TeammateStatus.ACTIVE
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TeammateRecord:
        teammate_id = str(data.get("id", "")).strip()
        if not teammate_id:
            raise TodoValidationError("invalid_teammate_id", "teammate id is required")
        if "/" in teammate_id or "\\" in teammate_id or ".." in teammate_id:
            raise TodoValidationError(
                "invalid_teammate_id",
                "teammate id contains invalid characters",
                {"id": teammate_id},
            )

        email = str(data.get("email", "")).strip()
        if not email:
            raise TodoValidationError(
                "invalid_teammate_email", "email is required", {"id": teammate_id}
            )

        name = str(data.get("name", "")).strip()
        if not name:
            raise TodoValidationError(
                "invalid_teammate_name", "name is required", {"id": teammate_id}
            )

        status_raw = data.get("status", TeammateStatus.ACTIVE)
        try:
            status_value = status_raw if isinstance(status_raw, TeammateStatus) else TeammateStatus(str(status_raw))
        except ValueError as exc:
            raise TodoValidationError(
                "invalid_teammate_status",
                f"invalid teammate status: {status_raw}",
                {"id": teammate_id},
            ) from exc

        created_raw = data.get("created_at")
        if isinstance(created_raw, datetime):
            created_at = created_raw
        elif isinstance(created_raw, (str, bytes)) and str(created_raw).strip():
            try:
                created_at = datetime.fromisoformat(str(created_raw))
            except ValueError as exc:
                raise TodoValidationError(
                    "invalid_teammate_created_at",
                    f"invalid created_at: {created_raw}",
                    {"id": teammate_id},
                ) from exc
        else:
            created_at = utc_now()

        return cls(
            id=teammate_id,
            email=email,
            name=name,
            status=status_value,
            created_at=created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
        }


@dataclass(slots=True)
class InboxMessage:
    to: str
    sender: str
    content: str
    created_at: datetime = field(default_factory=utc_now)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> InboxMessage:
        to_value = str(data.get("to", "")).strip()
        if not to_value:
            raise TodoValidationError("invalid_message_to", "message 'to' is required")
        if "/" in to_value or "\\" in to_value or ".." in to_value:
            raise TodoValidationError("invalid_message_to", "invalid recipient id", {"to": to_value})

        sender_value = str(data.get("sender", "")).strip()
        if not sender_value:
            raise TodoValidationError("invalid_message_sender", "sender is required", {"to": to_value})

        content_value = str(data.get("content", "")).strip()
        if not content_value:
            raise TodoValidationError("invalid_message_content", "content is required", {"to": to_value})

        created_raw = data.get("created_at")
        if isinstance(created_raw, datetime):
            created_at = created_raw
        elif isinstance(created_raw, (str, bytes)) and str(created_raw).strip():
            try:
                created_at = datetime.fromisoformat(str(created_raw))
            except ValueError as exc:
                raise TodoValidationError(
                    "invalid_message_created_at",
                    f"invalid created_at: {created_raw}",
                    {"to": to_value},
                ) from exc
        else:
            created_at = utc_now()

        return cls(to=to_value, sender=sender_value, content=content_value, created_at=created_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "to": self.to,
            "sender": self.sender,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }
