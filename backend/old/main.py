from datetime import datetime
from typing import List

from crud import (
    create_chat_message,
    create_chat_session,
    delete_session,
    get_session_messages,
    get_user_sessions,
)
from database import get_db
from fastapi import Depends, FastAPI, HTTPException
from models import Base
from pydantic import BaseModel
from sqlalchemy.orm import Session

app = FastAPI()


Base.metadata.create_all(bind=get_db().__next__().get_bind())


class ChatMessageCreate(BaseModel):
    role: str
    content: str
    metadata: dict = None


class ChatSessionCreate(BaseModel):
    user_id: str
    session_name: str = None


@app.post("/sessions/", response_model=dict)
def create_session(session: ChatSessionCreate, db: Session = Depends(get_db)):
    db_session = create_chat_session(db, session.user_id, session.session_name)
    return {"session_id": db_session.id, "message": "Session created successfully"}


@app.post("/sessions/{session_id}/messages/", response_model=dict)
def add_message(
    session_id: int, message: ChatMessageCreate, db: Session = Depends(get_db)
):
    db_message = create_chat_message(
        db, session_id, message.role, message.content, message.metadata
    )
    return {"message_id": db_message.id, "message": "Message added successfully"}


@app.get("/users/{user_id}/sessions/", response_model=List[dict])
def get_sessions(user_id: str, db: Session = Depends(get_db)):
    sessions = get_user_sessions(db, user_id)
    return [
        {
            "id": session.id,
            "session_name": session.session_name,
            "created_at": session.created_at,
            "message_count": len(session.messages),
        }
        for session in sessions
    ]


@app.get("/sessions/{session_id}/messages/", response_model=List[dict])
def get_messages(session_id: int, db: Session = Depends(get_db)):
    messages = get_session_messages(db, session_id)
    return [
        {
            "id": msg.id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "metadata": msg.message_metadata,
        }
        for msg in messages
    ]


@app.delete("/sessions/{session_id}/", response_model=dict)
def remove_session(session_id: int, db: Session = Depends(get_db)):
    if delete_session(db, session_id):
        return {"message": "Session deleted successfully"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
