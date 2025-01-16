import os

from dotenv import load_dotenv
from flask import Flask, request, jsonify

from ChatBotProxy.main_engine.import_docu import get_document_links, fetch_documents
from ChatBotProxy.main_engine.query_ollama import query_ollama

app = Flask(__name__)

# Load environment variables from the .env file
load_dotenv()

# Apply configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')

config = {
    'llm': os.getenv('LLM_MODEL'),
    'url': os.getenv('DOCUSAURUS_URL'),
    'path': os.getenv('DOCUSAURUS_BASE_PATH'),
    'embedding_model': os.getenv('EMBEDDING_MODEL'),
}


@app.route('/update', methods=['GET'])
def handle_update():
    fetch_documents(config['url'], config['path'], config['embedding_model'])
    return "done", 200


@app.route('/chat', methods=['POST'])
def handle_post():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No JSON payload provided"}), 400
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    document_links = get_document_links(config['url'], config['path'], config['embedding_model'])
    return jsonify(data | query_ollama(data['question'], config['llm'], config['embedding_model'], document_links)), 200


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port)
