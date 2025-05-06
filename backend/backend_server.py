import os
from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS
from flask_restx import Api

# RESTX namespaces
from controllers.chat_controller import chat_ns
from controllers.session_controller import session_ns

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.getenv("SECRET_KEY", "your_secret_key")

# Initialize Flask-RESTX API (with Swagger docs at /docs)
api = Api(app, doc="/docs", title="AI-TTORNEY API", description="Legal assistant API")

# Register namespaces for RESTX
api.add_namespace(chat_ns)
api.add_namespace(session_ns)

# Run app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
