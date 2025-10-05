#!/usr/bin/env python3
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "chemeng-tutor"

SYSTEM_PROMPT = """You are an expert Chemical Engineering tutor.

FORMATTING:
- Use LaTeX for math: $inline$ or $$display$$
- Use markdown: **bold**, lists, code blocks
- Always include units

TEACHING:
- Be clear, accurate, and concise
- Provide step-by-step explanations for complex topics
- Give numerical examples when helpful
- For simple questions, answer directly

THINKING (OPTIONAL):
- For complex multi-step problems, you MAY briefly show reasoning in <thinking> tags
- Keep thinking SHORT and focused on key steps
- For simple questions, skip thinking and answer directly

RESTRICTIONS:
- ONLY Chemical Engineering topics
- If asked other topics: "I specialize in Chemical Engineering topics!"

Maintain high accuracy while being efficient."""

def query_ollama(prompt, conversation_history=None):
    try:
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\n".join([
                f"{'Student' if msg['role'] == 'user' else 'Tutor'}: {msg['content'][:300]}" 
                for msg in conversation_history[-4:]
            ])
            full_prompt = f"Recent context:\n{history_text}\n\nStudent: {prompt}"
        else:
            full_prompt = prompt
        
        payload = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": 0.6,
                "top_p": 0.9,
                "num_ctx": 6144,
                "num_predict": 1536,
                "top_k": 40
            }
        }
        
        logger.info(f"Sending request to Ollama (balanced speed/quality)...")
        
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=150
        )
        
        if response.status_code == 200:
            result = response.json()["response"]
            logger.info(f"Response generated ({len(result)} chars)")
            return result
        else:
            logger.error(f"Ollama failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    try:
        test_response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        
        if test_response.status_code == 200:
            return jsonify({
                "status": "healthy",
                "ollama": "connected",
                "model": MODEL_NAME
            }), 200
        else:
            return jsonify({
                "status": "unhealthy",
                "ollama": "disconnected"
            }), 503
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        history = data.get('history', [])
        
        if not user_message:
            return jsonify({"error": "Empty message"}), 400
        
        logger.info(f"Message: {user_message[:100]}...")
        
        bot_response = query_ollama(user_message, history)
        
        if bot_response:
            return jsonify({
                "response": bot_response,
                "model": MODEL_NAME
            })
        else:
            return jsonify({
                "error": "Failed to generate response"
            }), 500
            
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "Internal error"}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🧪 Chemical Engineering Chatbot API Server")
    print("=" * 70)
    print(f"Model: qwen3:235b-a22b → chemeng-tutor")
    print(f"Server: http://localhost:5001")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=5001, debug=True)
