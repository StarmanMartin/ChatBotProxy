import faiss
import requests
import os

__all__ = ['query_ollama']

from ChatBotProxy.main_engine.import_docu import docu_root, get_embedding_model

def search_index(query, index, model, doc_text_links: list[str], top_k=10):
    """Search the FAISS index with a query and return top_k results."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        with open(doc_text_links[idx], 'r') as f:
            results = [(f.read(), distances[0][i])]
    return results


def _build_prompt(question: str, model_name: str, embedding_model:str, doc_text_links: list[str]):
    model = get_embedding_model(embedding_model)
    # Save the index for later use
    index = faiss.read_index(os.path.join(docu_root(), "faiss_index.bin"))

    # Search FAISS index
    results = search_index(question, index, model, doc_text_links, top_k=10)

    context = "\n".join([r[0] for r in results])
    # Command to send the POST request on the remote server
    prompt = f"Based on the following context, answer the question:\n\nContext: {context}\n\nQuestion: {question}"
    return {"prompt": prompt, "model": model_name, "stream": False}


def query_ollama(question: str, model_name: str, embedding_model: str, doc_text_links: list[str]) -> dict[str:str]:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=_build_prompt(question, model_name, embedding_model, doc_text_links)
    )
    try:
        return {"answer": response.json().get("response")}
    except ValueError as e:
        print("JSON parsing error:", e)
        print("Raw response content:", response.text)
        return {"error": "Invalid JSON response"}

