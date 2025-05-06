import json
import time

import requests

BASE_URL = "http://localhost:8000"


def test_chat_persistence():

    user_id = "test_user_123"
    session_name = "Test Session"
    messages = [
        {"role": "user", "content": "Hello!", "metadata": {}},
        {"role": "assistant", "content": "Hi there!", "metadata": {}},
        {"role": "user", "content": "How are you?", "metadata": {}},
    ]

    print("1. Creating a new chat session...")
    session_response = requests.post(
        f"{BASE_URL}/sessions/", json={"user_id": user_id, "session_name": session_name}
    )
    session_id = session_response.json()["session_id"]
    print(f"Session created with ID: {session_id}")

    print("\n2. Adding messages to the session...")
    for msg in messages:
        response = requests.post(
            f"{BASE_URL}/sessions/{session_id}/messages/", json=msg
        )
        print(f"Response status: {response.status_code}")
        print(f"Response body: {response.text}")
        print(f"Added message: {msg['content']}")

    print("\n3. Retrieving session messages...")
    messages_response = requests.get(f"{BASE_URL}/sessions/{session_id}/messages/")
    stored_messages = messages_response.json()
    print("Current messages:")
    for msg in stored_messages:
        print(f"- {msg['role']}: {msg['content']} (ID: {msg['id']})")
    print(f"Total messages in session: {len(stored_messages)}")

    print("\n4. Getting user sessions...")
    sessions_response = requests.get(f"{BASE_URL}/users/{user_id}/sessions/")
    print("User sessions:")
    for session in sessions_response.json():
        print(
            f"- Session {session['id']}: {session['session_name']} ({session['message_count']} messages)"
        )

        session_messages = requests.get(
            f"{BASE_URL}/sessions/{session['id']}/messages/"
        ).json()
        for msg in session_messages:
            print(f"  └─ {msg['role']}: {msg['content']}")

    print("\nTest completed successfully!")


if __name__ == "__main__":
    print("Starting database persistence test...")
    test_chat_persistence()
