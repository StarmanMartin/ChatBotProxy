import os

from threading import Thread
from dotenv import load_dotenv, find_dotenv
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

from ChatBotProxy.main_engine.import_docu import ContextManager
from ChatBotProxy.main_engine.query_ollama import query_ollama, build_question_prompt

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Load environment variables from the .env file
load_dotenv(find_dotenv(usecwd=True))

# Apply configuration from environment variables
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')
socketio = SocketIO(app)

config = {
    'llm': os.getenv('LLM_MODEL'),
    'url': os.getenv('DOCUSAURUS_URL'),
    'path': os.getenv('DOCUSAURUS_BASE_PATH'),
    'embedding_model': os.getenv('EMBEDDING_MODEL'),
}

ContextManager().setup(config['embedding_model'], config['url'], config['llm'], config['path'])

MAIN_DOCU_THREAD = None




def send_update(method_type: str, value: dict):
    if method_type == 'meta':
        socketio.emit('meda_data', {'links_len': value['len']})
    if method_type in ['links', 'index', 'links-meta', 'generate_questions', 'generated_questions']:
        socketio.emit(method_type, value)

@app.route('/update', methods=['GET'])
def handle_update():
    global MAIN_DOCU_THREAD
    if MAIN_DOCU_THREAD is not None and MAIN_DOCU_THREAD.is_alive():
        return render_template("index.html", header="Process is already running")

    MAIN_DOCU_THREAD = Thread(target=ContextManager().fetch_documents, args=(send_update,))
    MAIN_DOCU_THREAD.daemon = True
    MAIN_DOCU_THREAD.start()
    return render_template("index.html", header="Starting new process")



@app.route('/index_chunks', methods=['GET'])
def index_chunks():
    global MAIN_DOCU_THREAD
    if MAIN_DOCU_THREAD is not None and MAIN_DOCU_THREAD.is_alive():
        return render_template("index.html", header="Process is already running")

    MAIN_DOCU_THREAD = Thread(target=ContextManager().index_chunks, args=(send_update,))
    MAIN_DOCU_THREAD.daemon = True
    MAIN_DOCU_THREAD.start()
    return render_template("index.html", header="Starting new process")



@app.route('/generate_questions', methods=['GET'])
def generate_questions():
    global MAIN_DOCU_THREAD
    if MAIN_DOCU_THREAD is not None and MAIN_DOCU_THREAD.is_alive():
        return render_template("index_questions.html", header="Process is already running")

    MAIN_DOCU_THREAD = Thread(target=ContextManager().generate_questions, args=(send_update,))
    MAIN_DOCU_THREAD.daemon = True
    MAIN_DOCU_THREAD.start()
    return render_template("index_questions.html", header="Starting new process")


@app.route('/chat', methods=['POST'])
def handle_post():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No JSON payload provided"}), 400
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400
    prompt = build_question_prompt(data['question'])
    return jsonify(data | query_ollama(prompt, config['llm'])), 200


if __name__ == '__main__':
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 5000))
    app.run(host=host, port=port)
