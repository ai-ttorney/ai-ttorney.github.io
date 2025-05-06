import time

from database import engine
from sqlalchemy import text


def test_connection():
    print("Testing database connection...")

    try:

        with engine.connect() as connection:

            result = connection.execute(text("SELECT version();"))
            version = result.scalar()
            print(f"✓ Successfully connected to PostgreSQL!")
            print(f"PostgreSQL version: {version}")

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

    except Exception as e:
        print(f"✗ Connection failed: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check your .env file has correct credentials:")
        print("   POSTGRES_USER=your_username")
        print("   POSTGRES_PASSWORD=your_password")
        print("   POSTGRES_HOST=localhost")
        print("   POSTGRES_PORT=5432")
        print("   POSTGRES_DB=chat_history")
        print("3. Verify the database 'chat_history' exists")
        print("4. Check if the user has proper permissions")


if __name__ == "__main__":
    test_connection()
