import json
import os
import re
import shutil

import faiss
import requests
from bs4 import BeautifulSoup

import html2text
from sentence_transformers import SentenceTransformer

__all__ = ['fetch_documents', 'get_document_links', 'docu_root', 'get_embedding_model']


_MODEL = None
_DOCU_LINKS = []
def _get_all_website_links(base_url: str, base_path: str = '/', urls: list | None = None, idx: int = 0):
    if urls is None:
        urls = list()
        urls.append(base_path)

    url = urls[idx]
    response = requests.get(base_url + url)
    soup = BeautifulSoup(response.text, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            continue
        if href.startswith(base_url):
            href = href.replace(base_url, '')
        href = href.split('#')[0]
        if href.startswith(base_path) and href not in urls:
            urls.append(href)

    idx += 1
    if idx < len(urls):
        return _get_all_website_links(base_url, base_path, urls, idx)
    return list(dict.fromkeys(urls))


def _make_file_name(text, idx, prefix):
    if text.startswith(prefix):
        text = text[len(prefix):]
    return re.sub(r'/', '_', text) + '.' + str(idx) + '.txt'


def _extract_text_from_web(url_docu):
    response = requests.get(url_docu)
    soup = BeautifulSoup(response.text, 'html.parser')
    div = soup.find('div', {'class': 'theme-doc-markdown'})
    h = html2text.HTML2Text()
    if not div:
        res = h.handle(str(soup))
    else:
        res = h.handle(str(div))
    return re.sub(r'^\n|\n$', '', re.sub('â|Â ', '', res))

def _fp(path_name: str) -> str:
    return os.path.join(os.path.dirname(__file__), path_name)

def docu_root() -> str:
    return _fp('docu')

def fetch_documents(url: str, base_path: str, embedding_model: str):
    links = _get_all_website_links(url, base_path)
    if os.path.isdir(docu_root()):
        shutil.rmtree(docu_root())
    os.makedirs(docu_root(), exist_ok=True)
    index = {'links': _DOCU_LINKS}
    text_chunks = []
    for link in links:
        text = _extract_text_from_web(url + link)
        main_header = text.split('\n')[0]
        for idx, text_part in enumerate(re.split(r'\n## ', text)):
            text_part = main_header + '\n## ' + text_part.strip('#')
            fn = _make_file_name(link, idx, url)
            text_chunks.append(text_part)
            file_path = os.path.join(docu_root(), fn)
            with open(file_path, 'w+') as f:
                f.write(text_part)
            index['links'].append(file_path)
    with open(os.path.join(docu_root(), 'index.json'), 'w+') as f:
        f.write(json.dumps(index))

    # Load pre-trained model
    model = get_embedding_model(embedding_model)# Lightweight and efficient

    # Generate embeddings
    embeddings = model.encode(text_chunks, convert_to_tensor=True)

    # Convert embeddings to numpy array
    embeddings_np = embeddings.cpu().detach().numpy()

    # Create a FAISS index
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
    index.add(embeddings_np)

    # Save the index for later use
    faiss.write_index(index, os.path.join(docu_root(), "faiss_index.bin"))


def get_document_links(url: str, base_path: str, embedding_model: str):
    global _DOCU_LINKS
    if len(_DOCU_LINKS) == 0:
        json_path = os.path.join(docu_root(), 'index.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                _DOCU_LINKS = json.loads(f.read())['links']
        else:
            fetch_documents(url, base_path, embedding_model)
    return _DOCU_LINKS


def get_embedding_model(embedding_model: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(embedding_model)
    return _MODEL