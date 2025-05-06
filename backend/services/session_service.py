import logging
import traceback
from datetime import datetime

from database.database import get_db
from flask import jsonify, request
from langchain_community.chat_message_histories import ChatMessageHistory
from models.crud import create_chat_session, get_session_messages
from models.models import ChatMessage, ChatSession
from services.chat_service import thread_metadata, user_queues, user_threads

logger = logging.getLogger(__name__)


class ThreadMetadata:
    def __init__(self, thread_id: str, user_id: str):
        self.thread_id = thread_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        self.title = "New Conversation"
        self.last_message = None
        self.is_initial = True
        self.db_session_id = None


def create_thread_service():
    """Create a new thread for a user (only if explicitly forced)"""
    try:
        data = request.json
        user_id = data.get("user_id")
        force = data.get("force", False)

        logger.warning(f"\n🚨 CREATE_THREAD called at {datetime.now().isoformat()}")
        logger.warning(f"🚨 Data received: {data}")
        logger.warning("🚨 Call stack:\n" + "".join(traceback.format_stack(limit=5)))

        if not user_id or not force:
            logger.warning("🚫 Thread creation blocked — missing force flag or user ID")
            return jsonify({"error": "Thread creation not allowed"}), 400

        db = next(get_db())
        try:
            new_session = create_chat_session(db, user_id, "New Chat")
            db.commit()

            thread_id = str(new_session.id)

            user_threads[thread_id] = ChatMessageHistory()
            user_queues[thread_id] = None
            thread_metadata[thread_id] = ThreadMetadata(thread_id, user_id)
            thread_metadata[thread_id].db_session_id = new_session.id
            thread_metadata[thread_id].title = "New Chat"

            logger.warning(f"✅ Thread created: {thread_id}")
            return jsonify(
                {
                    "thread_id": thread_id,
                    "title": "New Chat",
                    "created_at": new_session.created_at.isoformat(),
                    "last_updated": new_session.updated_at.isoformat(),
                    "is_initial": True,
                }
            )

        except Exception as e:
            logger.error(f"❌ DB error creating thread: {str(e)}")
            logger.error(traceback.format_exc())
            db.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

    except Exception as e:
        logger.error(f"❌ Error in create_thread_service: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def get_all_threads_service():
    """Get all threads for a specific user"""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"error": "User ID is required"}), 400

        db = next(get_db())
        try:
            sessions = (
                db.query(ChatSession).filter(ChatSession.user_id == user_id).all()
            )
            threads_data = []

            for session in sessions:
                last_message = (
                    db.query(ChatMessage)
                    .filter(ChatMessage.session_id == session.id)
                    .order_by(ChatMessage.timestamp.desc())
                    .first()
                )

                threads_data.append(
                    {
                        "thread_id": str(session.id),
                        "title": session.session_name or "New Chat",
                        "created_at": session.created_at.isoformat(),
                        "last_updated": session.updated_at.isoformat(),
                        "last_message": last_message.content if last_message else None,
                        "is_initial": not last_message,
                    }
                )

            threads_data.sort(key=lambda x: x["last_updated"], reverse=True)
            return jsonify({"threads": threads_data})

        except Exception as e:
            logger.error(f"Error getting threads: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in get_all_threads_service: {str(e)}")
        return jsonify({"error": str(e)}), 500


def get_thread_history_service():
    """Get chat history for a specific thread"""
    try:
        data = request.json
        thread_id = data.get("thread_id")
        user_id = data.get("user_id")

        if not thread_id or not user_id:
            return jsonify({"error": "Thread ID and User ID are required"}), 400

        db = next(get_db())
        try:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == int(thread_id), ChatSession.user_id == user_id
                )
                .first()
            )

            if not session:
                return jsonify({"error": "Thread not found"}), 404

            messages = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == session.id)
                .order_by(ChatMessage.timestamp.asc())
                .all()
            )

            history = []
            for msg in messages:
                history.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                    }
                )

            if thread_id in user_threads:
                user_threads[thread_id].clear()
                for msg in history:
                    if msg["role"] == "user":
                        user_threads[thread_id].add_user_message(msg["content"])
                    else:
                        user_threads[thread_id].add_ai_message(msg["content"])

            return jsonify(
                {
                    "history": history,
                    "title": session.session_name or "New Chat",
                    "thread_id": thread_id,
                }
            )

        except Exception as e:
            logger.error(f"Error getting thread history: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in get_thread_history_service: {str(e)}")
        return jsonify({"error": str(e)}), 500


def delete_thread_service():
    """Delete a thread for a specific user"""
    try:
        data = request.json
        thread_id = data.get("thread_id")
        user_id = data.get("user_id")

        if not thread_id or not user_id:
            return jsonify({"error": "Thread ID and User ID are required"}), 400

        db = next(get_db())
        try:
            session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == int(thread_id), ChatSession.user_id == user_id
                )
                .first()
            )

            if not session:
                return jsonify({"error": "Thread not found"}), 404

            db.delete(session)
            db.commit()

            if thread_id in user_threads:
                del user_threads[thread_id]
            if thread_id in user_queues:
                del user_queues[thread_id]
            if thread_id in thread_metadata:
                del thread_metadata[thread_id]

            return jsonify({"message": "Thread deleted successfully"})

        except Exception as e:
            logger.error(f"Error deleting thread: {str(e)}")
            logger.error(traceback.format_exc())
            db.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Error in delete_thread_service: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
