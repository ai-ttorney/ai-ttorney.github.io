import time

from database import Base, engine
from models import ChatMessage, ChatSession
from sqlalchemy import text


def init_database():
    print("Initializing database...")

    try:

        Base.metadata.create_all(bind=engine)
        print("✓ Database tables created successfully!")

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

            required_tables = {"chat_sessions", "chat_messages"}
            existing_tables = set(tables)

            if required_tables.issubset(existing_tables):
                print("✓ All required tables exist:")
                for table in required_tables:
                    print(f"  - {table}")
            else:
                missing_tables = required_tables - existing_tables
                print(f"✗ Missing tables: {missing_tables}")

    except Exception as e:
        print(f"✗ Database initialization failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your database credentials")
        print("3. Verify you have proper permissions")
        print("4. Check if the database exists")


if __name__ == "__main__":
    init_database()
