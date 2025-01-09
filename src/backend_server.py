from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
import os

load_dotenv()


app = Flask(__name__)

CORS(app)

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # Construct the 'messages' array for ChatCompletion
        messages = [
            {"role": "system", "content": "AI-ttorney is a financial law advice chatbot and will only answer questions related to Turkey's financial law. For all other questions, it must indicate that he is only trained for Turkey's financial law."},
            {"role": "user", "content": prompt}
        ]

        # Use ChatCompletion endpoint
        response = openai.ChatCompletion.create(
            model="ft:gpt-4o-mini-2024-07-18:aittorney:aittorney-test2:AjWMVzNY",  # Replace with your fine-tuned model name if applicable
            messages=messages,
            max_tokens=100
        )

        # Extract and return the assistant's reply
        return jsonify({"response": response['choices'][0]['message']['content']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
