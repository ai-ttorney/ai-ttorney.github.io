from datetime import datetime
from typing import List, Optional

from models.models import ChatMessage, ChatSession
from sqlalchemy.orm import Session, joinedload


def create_chat_session(
    db: Session, user_id: str, session_name: Optional[str] = None
) -> ChatSession:
    db_session = ChatSession(user_id=user_id, session_name=session_name)
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session


def create_chat_message(
    db: Session,
    session_id: int,
    role: str,
    content: str,
    metadata: Optional[dict] = None,
) -> ChatMessage:
    db_message = ChatMessage(
        session_id=session_id, role=role, content=content, message_metadata=metadata
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    return db_message


def get_user_sessions(db: Session, user_id: str) -> List[ChatSession]:
    return (
        db.query(ChatSession)
        .options(joinedload(ChatSession.messages))
        .filter(ChatSession.user_id == user_id)
        .all()
    )


def get_session_messages(db: Session, session_id: int) -> List[ChatMessage]:
    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.timestamp)
        .all()
    )


def get_session_by_id(db: Session, session_id: int) -> Optional[ChatSession]:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def update_session_name(
    db: Session, session_id: int, new_name: str
) -> Optional[ChatSession]:
    session = get_session_by_id(db, session_id)
    if session:
        session.session_name = new_name
        db.commit()
        db.refresh(session)
    return session


def delete_session(db: Session, session_id: int) -> bool:
    session = get_session_by_id(db, session_id)
    if session:
        db.delete(session)
        db.commit()
        return True
    return False


def delete_message(db: Session, message_id: int) -> bool:
    message = db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
    if message:
        db.delete(message)
        db.commit()
        return True
    return False
