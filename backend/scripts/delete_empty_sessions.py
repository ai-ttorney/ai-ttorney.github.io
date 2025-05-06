from database import get_db
from models import ChatMessage, ChatSession
from sqlalchemy import func


def delete_empty_sessions():
    db = next(get_db())
    try:

        empty_sessions = (
            db.query(ChatSession)
            .outerjoin(ChatMessage)
            .group_by(ChatSession.id)
            .having(func.count(ChatMessage.id) == 0)
            .all()
        )

        for session in empty_sessions:
            db.delete(session)
            print(f"Deleted empty session with ID: {session.id}")

        db.commit()
        print(f"Successfully deleted {len(empty_sessions)} empty sessions")

    except Exception as e:
        print(f"Error deleting empty sessions: {str(e)}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    delete_empty_sessions()
