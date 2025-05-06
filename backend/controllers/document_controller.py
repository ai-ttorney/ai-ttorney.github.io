from flask import Blueprint, request, jsonify
from services.document_service import upload_document_service

document_bp = Blueprint('document', __name__)

@document_bp.route('/api/upload', methods=['POST'])
def upload_document():
    try:
        return upload_document_service()
    except Exception:
        return jsonify({"error": "Document upload failed. Please try again."}), 500