from flask import request
from flask_restx import Namespace, Resource
from services.session_service import create_thread_service, get_all_threads_service, get_thread_history_service, delete_thread_service

session_ns = Namespace("session", description="Session related operations", path="/api")


@session_ns.route("/create_thread")
class CreateThread(Resource):
    def post(self):
        """Create a new thread"""
        try:
            return create_thread_service()
        except Exception:
            return {"error": "Thread could not be created."}, 500


@session_ns.route("/get_all_threads")
class GetAllThreads(Resource):
    def get(self):
        """Get all threads for a user"""
        try:
            return get_all_threads_service()
        except Exception:
            return {"error": "Could not retrieve chat history."}, 500


@session_ns.route("/get_thread_history")
class GetThreadHistory(Resource):
    def post(self):
        """Get chat history for a specific thread"""
        try:
            return get_thread_history_service()
        except Exception:
            return {"error": "Failed to load thread."}, 500


@session_ns.route("/delete_thread")
class DeleteThread(Resource):
    def post(self):
        """Delete a specific thread"""
        try:
            return delete_thread_service()
        except Exception:
            return {"error": "Thread deletion failed."}, 500