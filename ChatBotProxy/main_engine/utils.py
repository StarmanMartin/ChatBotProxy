import os

import requests


def query_ollama(prompt: str, model_name: str, stream: bool = False) -> dict[str:str]:
    if os.getenv('ONLY_SAMPLE_ANSWER', 'f') .lower() == 'true':
        with open(os.path.join(os.path.dirname(__file__), 'sample_answer.md'), 'r') as f:
            return {'answer': f.read()}

    response = requests.post(
        "http://localhost:11434/api/generate",
        json= {"prompt": prompt, "model": model_name, "stream": stream}
    )
    try:
        return {"answer": response.json().get("response")}
    except ValueError as e:
        print("JSON parsing error:", e)
        print("Raw response content:", response.text)
        return {"error": "Invalid JSON response"}