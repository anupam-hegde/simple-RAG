"""
Chat History Service — persists conversation sessions to a JSON file.

Each session holds an ordered list of messages (user + assistant),
including source references returned by the RAG pipeline.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)

HISTORY_FILENAME = "chat_history.json"


# ── Data Models ──────────────────────────────────────────────────────────────

@dataclass
class SourceRef:
    text: str
    filename: str
    page: int


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    sources: list[SourceRef] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = "New Chat"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    messages: list[Message] = field(default_factory=list)


# ── Service ──────────────────────────────────────────────────────────────────

class ChatHistoryService:
    """Thread-safe, file-backed chat history store."""

    def __init__(self) -> None:
        settings = get_settings()
        data_dir = Path(settings.UPLOAD_DIR).parent  # …/data
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / HISTORY_FILENAME
        self._sessions: dict[str, Session] = {}
        self._load()

    # ── Public API ───────────────────────────────────────────────────────

    def create_session(self, title: Optional[str] = None) -> Session:
        session = Session(title=title or "New Chat")
        self._sessions[session.id] = session
        self._save()
        logger.info(f"Created session {session.id}")
        return session

    def list_sessions(self) -> list[dict]:
        """Return lightweight session summaries sorted newest-first."""
        return sorted(
            [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at,
                    "message_count": len(s.messages),
                }
                for s in self._sessions.values()
            ],
            key=lambda x: x["created_at"],
            reverse=True,
        )

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[list[dict]] = None,
    ) -> Message:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id} not found")

        source_refs = [
            SourceRef(
                text=s.get("text", ""),
                filename=s.get("filename", "Unknown"),
                page=s.get("page", 0),
            )
            for s in (sources or [])
        ]

        msg = Message(role=role, content=content, sources=source_refs)
        session.messages.append(msg)

        # Auto-title from the first user message
        if role == "user" and len(session.messages) == 1:
            session.title = content[:80] + ("…" if len(content) > 80 else "")

        self._save()
        return msg

    # ── Persistence ──────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            data = {sid: asdict(s) for sid, s in self._sessions.items()}
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"Failed to save chat history: {e}")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for sid, sdata in raw.items():
                messages = [
                    Message(
                        role=m["role"],
                        content=m["content"],
                        sources=[SourceRef(**s) for s in m.get("sources", [])],
                        timestamp=m.get("timestamp", ""),
                    )
                    for m in sdata.get("messages", [])
                ]
                self._sessions[sid] = Session(
                    id=sdata["id"],
                    title=sdata.get("title", "New Chat"),
                    created_at=sdata.get("created_at", ""),
                    messages=messages,
                )
            logger.info(f"Loaded {len(self._sessions)} sessions from disk")
        except Exception as e:
            logger.warning(f"Could not load chat history: {e}")


# Singleton accessor
_instance: Optional[ChatHistoryService] = None


def get_chat_history_service() -> ChatHistoryService:
    global _instance
    if _instance is None:
        _instance = ChatHistoryService()
    return _instance
