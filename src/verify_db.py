from database import Base, engine
from models import ChatMessage, ChatSession
from sqlalchemy import text


def verify_database():
    print("Verifying database setup...")

    try:

        Base.metadata.create_all(bind=engine)
        print("✓ Tables created/verified successfully!")

        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """
                )
            )
            tables = [row[0] for row in result]

            print("\nExisting tables:")
            for table in tables:
                print(f"- {table}")

            if "chat_sessions" in tables:
                result = connection.execute(text("SELECT COUNT(*) FROM chat_sessions"))
                count = result.scalar()
                print(f"\nNumber of chat sessions: {count}")

                result = connection.execute(
                    text(
                        """
                    SELECT cs.id, cs.user_id, cs.session_name, cs.created_at, cs.updated_at,
                           cm.id as message_id, cm.role, cm.content, cm.timestamp
                    FROM chat_sessions cs
                    LEFT JOIN chat_messages cm ON cs.id = cm.session_id
                    ORDER BY cs.id, cm.timestamp
                """
                    )
                )

                current_session = None
                for row in result:
                    if current_session != row[0]:
                        if current_session is not None:
                            print("-" * 50)
                        current_session = row[0]
                        print(f"\nSession ID: {row[0]}")
                        print(f"User ID: {row[1]}")
                        print(f"Session Name: {row[2]}")
                        print(f"Created At: {row[3]}")
                        print(f"Updated At: {row[4]}")
                        print("\nMessages:")

                    if row[5] is not None:
                        print(f"  - [{row[7]}] {row[6]}: {row[7]}")

                if current_session is not None:
                    print("-" * 50)

            if "chat_messages" in tables:
                result = connection.execute(text("SELECT COUNT(*) FROM chat_messages"))
                count = result.scalar()
                print(f"\nTotal number of chat messages: {count}")

    except Exception as e:
        print(f"✗ Error verifying database: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your .env file has correct credentials")
        print("3. Verify the database 'chat_history' exists")
        print("4. Check if the user has proper permissions")


if __name__ == "__main__":
    verify_database()
