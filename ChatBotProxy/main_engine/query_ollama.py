import faiss
from ChatBotProxy.main_engine.utils import query_ollama as ql
import os

__all__ = ['query_ollama', 'build_question_prompt']

from ChatBotProxy.main_engine.import_docu import ContextManager

def search_index(query, index, model, doc_text_links: list[str], top_k=10):
    """Search the FAISS index with a query and return top_k results."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for i, idx in enumerate(indices[0]):
        with open(doc_text_links[idx], 'r') as f:
            results = [(f.read(), distances[0][i])]
    return results


def build_question_prompt(question: str):
    model = ContextManager().get_embedding_model()
    # Save the index for later use
    index = faiss.read_index(os.path.join(ContextManager().docu_root(), "faiss_index.bin"))

    # Search FAISS index
    results = search_index(question, index, model, ContextManager().get_document_links(), top_k=10)

    context = "\n".join([r[0] for r in results])
    # Command to send the POST request on the remote server
    prompt = f"Based on the following context, answer the question about Chemotion:\n\nContext: {context}\n\nQuestion: {question}"
    return prompt


def query_ollama(prompt: str, model_name: str):
    return ql(prompt, model_name)
