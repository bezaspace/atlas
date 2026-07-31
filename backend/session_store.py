"""In-memory and file-backed session management for the Atlas backend.

Follows the patterns in victordibia/designing-multiagent-systems
(PicoAgents) webui sessions.
"""

from __future__ import annotations

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlascore.context import AgentContext


@dataclass
class Session:
    """A single backend execution session."""

    session_id: str
    entity_type: str
    entity_id: str
    context: AgentContext = field(default_factory=AgentContext)
    status: str = "created"
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    events: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancellation_token: Optional[Any] = None
    task: Optional[asyncio.Task] = None
    approval_event: asyncio.Event = field(default_factory=asyncio.Event)


class SessionStore(ABC):
    """Abstract base for session storage backends."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Session]:
        """Retrieve a session by ID."""
        pass

    @abstractmethod
    async def save(self, session_id: str, session: Session) -> None:
        """Persist a session."""
        pass

    @abstractmethod
    async def list(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List sessions with metadata."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Delete a session."""
        pass

    @abstractmethod
    async def clear_all(self) -> int:
        """Clear all sessions."""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory session store (lost on restart)."""

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}

    async def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def save(self, session_id: str, session: Session) -> None:
        self._sessions[session_id] = session

    async def list(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sessions = []
        for sid, session in self._sessions.items():
            if entity_id and session.entity_id != entity_id:
                continue
            sessions.append(
                {
                    "id": sid,
                    "entity_id": session.entity_id,
                    "entity_type": session.entity_type,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.metadata.get(
                        "last_activity", session.created_at
                    ).isoformat(),
                }
            )
        sessions.sort(key=lambda s: s["last_activity"], reverse=True)
        return sessions

    async def delete(self, session_id: str) -> bool:
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if session.task and not session.task.done():
                session.task.cancel()
            del self._sessions[session_id]
            return True
        return False

    async def clear_all(self) -> int:
        count = len(self._sessions)
        for session in self._sessions.values():
            if session.task and not session.task.done():
                session.task.cancel()
        self._sessions.clear()
        return count


class FileSessionStore(SessionStore):
    """File-backed session store with metadata and context only (no queued events)."""

    def __init__(self, storage_dir: str = ".atlas_sessions") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._meta: Dict[str, Dict[str, Any]] = {}
        self._contexts: Dict[str, AgentContext] = {}

    def _path(self, session_id: str) -> Path:
        return self.storage_dir / f"{session_id}.json"

    async def get(self, session_id: str) -> Optional[Session]:
        if session_id not in self._meta:
            path = self._path(session_id)
            if not path.exists():
                return None
            try:
                import json

                data = json.loads(path.read_text(encoding="utf-8"))
                self._meta[session_id] = data["metadata"]
                self._contexts[session_id] = AgentContext(**data["context"])
            except Exception:
                return None

        meta = self._meta[session_id]
        return Session(
            session_id=session_id,
            entity_type=meta.get("entity_type", "unknown"),
            entity_id=meta.get("entity_id", "unknown"),
            context=self._contexts[session_id],
            status=meta.get("status", "created"),
            created_at=datetime.fromisoformat(meta["created_at"]),
            metadata=meta.get("metadata", {}),
        )

    async def save(self, session_id: str, session: Session) -> None:
        self._meta[session_id] = {
            "entity_type": session.entity_type,
            "entity_id": session.entity_id,
            "status": session.status,
            "created_at": session.created_at.isoformat(),
            "metadata": session.metadata,
        }
        self._contexts[session_id] = session.context
        try:
            import json

            self._path(session_id).write_text(
                json.dumps(
                    {
                        "metadata": self._meta[session_id],
                        "context": session.context.model_dump(),
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    async def list(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sessions = []
        for path in self.storage_dir.glob("*.json"):
            session_id = path.stem
            session = await self.get(session_id)
            if session is None:
                continue
            if entity_id and session.entity_id != entity_id:
                continue
            sessions.append(
                {
                    "id": session_id,
                    "entity_id": session.entity_id,
                    "entity_type": session.entity_type,
                    "status": session.status,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.metadata.get(
                        "last_activity", session.created_at
                    ).isoformat(),
                }
            )
        sessions.sort(key=lambda s: s["last_activity"], reverse=True)
        return sessions

    async def delete(self, session_id: str) -> bool:
        if session_id in self._meta:
            del self._meta[session_id]
        if session_id in self._contexts:
            del self._contexts[session_id]
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    async def clear_all(self) -> int:
        count = 0
        for path in self.storage_dir.glob("*.json"):
            path.unlink()
            count += 1
        self._meta.clear()
        self._contexts.clear()
        return count


class SessionManager:
    """Manages backend sessions with pluggable storage."""

    def __init__(self, store: Optional[SessionStore] = None) -> None:
        self.store = store or InMemorySessionStore()

    def create_session_id(self) -> str:
        return str(uuid.uuid4())

    async def get_or_create(
        self,
        session_id: str,
        entity_id: str,
        entity_type: str = "research",
    ) -> Session:
        session = await self.store.get(session_id)
        if session is None:
            session = Session(
                session_id=session_id,
                entity_id=entity_id,
                entity_type=entity_type,
                context=AgentContext(
                    session_id=session_id,
                    metadata={
                        "entity_id": entity_id,
                        "entity_type": entity_type,
                    },
                ),
            )
            await self.store.save(session_id, session)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        return await self.store.get(session_id)

    async def update(self, session_id: str, session: Session) -> None:
        session.metadata["last_activity"] = datetime.now()
        await self.store.save(session_id, session)

    async def list(self, entity_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return await self.store.list(entity_id)

    async def delete(self, session_id: str) -> bool:
        return await self.store.delete(session_id)

    async def clear_all(self) -> int:
        return await self.store.clear_all()

    async def get_approval_event(self, session_id: str) -> asyncio.Event:
        session = await self.get_or_create(session_id, "unknown")
        return session.approval_event
