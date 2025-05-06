from typing import Generator

from database import get_db
from models import ChatMessage, ChatSession
from sqlalchemy.orm import Session


def check_database():
    print("\n=== Checking Database Contents ===")

    db: Generator[Session, None, None] = get_db()
    session = next(db)

    try:

        sessions = session.query(ChatSession).all()
        print(f"\nFound {len(sessions)} chat sessions:")

        for chat_session in sessions:
            print(f"\nSession ID: {chat_session.id}")
            print(f"User ID: {chat_session.user_id}")
            print(f"Session Name: {chat_session.session_name}")
            print(f"Created At: {chat_session.created_at}")
            print(f"Updated At: {chat_session.updated_at}")

            messages = (
                session.query(ChatMessage)
                .filter(ChatMessage.session_id == chat_session.id)
                .order_by(ChatMessage.timestamp)
                .all()
            )

            print(f"\nMessages in this session ({len(messages)}):")
            for msg in messages:
                print(f"  - [{msg.timestamp}] {msg.role}: {msg.content[:100]}...")
                if msg.message_metadata:
                    print(f"    Metadata: {msg.message_metadata}")

            print("-" * 50)

    except Exception as e:
        print(f"Error checking database: {str(e)}")
    finally:
        session.close()


if __name__ == "__main__":
    check_database()
