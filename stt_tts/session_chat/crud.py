"""
crud.py - Session Service CRUD + Redis Session Manager

Redis keys:
    session:{session_id}          →  JSON session metadata  (TTL = SESSION_TTL)
    session:{session_id}:active   →  "1" flag, means session is live in Redis
"""
import json
import os
import uuid
from datetime import datetime
from typing import Optional, List

from sqlalchemy.orm import Session as DBSession
from models import ChatSession

# ── Redis ─────────────────────────────────────────────────────────────────────

SESSION_TTL = int(os.getenv("SESSION_TTL", 86400))

try:
    import redis
    _redis = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        db=int(os.getenv("REDIS_DB", "0")),
        decode_responses=True,
    )
    _redis.ping()
    _redis_ok = True
except Exception:
    _redis = None
    _redis_ok = False


class RedisSessionManager:
    """Redis-backed session cache. Falls back to no-op if Redis is unavailable."""

    def set(self, session):
        if not _redis_ok:
            return
        key = f"session:{session.id}"
        _redis.setex(key, SESSION_TTL, json.dumps(session.to_dict()))
        _redis.setex(f"{key}:active", SESSION_TTL, "1")

    def get(self, session_id) -> Optional[dict]:
        if not _redis_ok:
            return None
        raw = _redis.get(f"session:{session_id}")
        return json.loads(raw) if raw else None

    def delete(self, session_id):
        if not _redis_ok:
            return
        _redis.delete(f"session:{session_id}", f"session:{session_id}:active")

    def refresh_ttl(self, session_id):
        if not _redis_ok:
            return
        key = f"session:{session_id}"
        _redis.expire(key, SESSION_TTL)
        _redis.expire(f"{key}:active", SESSION_TTL)

    def exists(self, session_id) -> bool:
        if not _redis_ok:
            return False
        return _redis.exists(f"session:{session_id}:active") > 0


# Singleton
redis_session_manager = RedisSessionManager()


# ── CRUD ──────────────────────────────────────────────────────────────────────

class SessionCRUD:
    def __init__(self, db: DBSession):
        self.db  = db
        self.rsm = redis_session_manager

    # ── Create ───────────────────────────────────────────────────
    def create(self, user_id: str, title: str = None) -> ChatSession:
        session = ChatSession(user_id=uuid.UUID(user_id), title=title)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        self.rsm.set(session)
        return session

    # ── Read ─────────────────────────────────────────────────────
    def get_by_id(self, session_id: str) -> Optional[ChatSession]:
        self.rsm.refresh_ttl(session_id)
        session = self.db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        if session and not self.rsm.exists(session_id):
            self.rsm.set(session)
        return session

    def get_cached(self, session_id: str) -> Optional[dict]:
        """Return metadata from Redis only — no DB hit."""
        return self.rsm.get(session_id)

    def list_by_user(self, user_id: str) -> List[ChatSession]:
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == uuid.UUID(user_id))
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

    def exists(self, session_id: str) -> bool:
        if self.rsm.exists(session_id):
            return True
        return (
            self.db.query(ChatSession.id)
            .filter(ChatSession.id == uuid.UUID(session_id))
            .scalar() is not None
        )

    # ── Update ───────────────────────────────────────────────────
    def update_title(self, session_id: str, title: str) -> Optional[ChatSession]:
        session = self.get_by_id(session_id)
        if not session:
            return None
        session.title      = title
        session.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(session)
        self.rsm.set(session)
        return session

    def touch(self, session_id: str):
        """Bump updated_at — called by history-service after each message."""
        session = self.db.query(ChatSession).filter(
            ChatSession.id == uuid.UUID(session_id)
        ).first()
        if session:
            session.updated_at = datetime.utcnow()
            self.db.commit()
            self.rsm.set(session)

    # ── Delete ───────────────────────────────────────────────────
    def delete(self, session_id: str) -> bool:
        session = self.get_by_id(session_id)
        if not session:
            return False
        self.db.delete(session)
        self.db.commit()
        self.rsm.delete(session_id)
        return True