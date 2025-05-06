from flask import request
from flask_restx import Namespace, Resource
from services.chat_service import (
    clear_chat_service,
    generate_chat_service,
    process_chat_service,
)

chat_ns = Namespace("chat", description="Chat related operations", path="/api")


@chat_ns.route("/process_chat")
class ProcessChat(Resource):
    def post(self):
        """Process the chat message for a given thread"""
        try:
            thread_id = request.args.get("thread_id")
            return process_chat_service(thread_id)
        except Exception:
            return {
                "error": "Unable to process chat. Please refresh and try again."
            }, 500


@chat_ns.route("/generate")
class GenerateChat(Resource):
    def post(self):
        """Generate a new chat message"""
        try:
            return generate_chat_service()
        except Exception:
            return {"error": "Chat generation failed. Please try again later."}, 500


@chat_ns.route("/clear")
class ClearChat(Resource):
    def post(self):
        """Clear chat history"""
        try:
            return clear_chat_service()
        except Exception:
            return {"error": "Failed to clear chat history."}, 500
