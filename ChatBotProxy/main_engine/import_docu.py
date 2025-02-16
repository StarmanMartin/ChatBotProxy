import json
import os
import re
import shutil
import threading
from enum import Enum
from typing import Callable

import faiss
import requests
from bs4 import BeautifulSoup

import html2text
from sentence_transformers import SentenceTransformer

from ChatBotProxy.main_engine.utils import query_ollama


class ThreadSafeSingleton(type):
    _instances = {}
    _singleton_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # double-checked locking pattern (https://en.wikipedia.org/wiki/Double-checked_locking)
        if cls not in cls._instances:
            with cls._singleton_lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super(ThreadSafeSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ContextTypes(Enum):
    DOCUSAURUS = 1


class ContextManager(metaclass=ThreadSafeSingleton):

    def __init__(self):
        self._base_url = self._base_path = self.context_types = self._embedding_model_name = None
        self._embedding_model = self._embedding_model = self._docu_links = None
        self._llm = None

        self._chunk_size = int(os.environ.get('CHUNK_SIZE', 2000))

    def setup(self, embedding_model: str, base_url: str, llm: str, base_path: str = '/',
              context_types: ContextTypes = ContextTypes.DOCUSAURUS):
        self._base_url, self._base_path = base_url, base_path
        self.context_types = context_types
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        self._docu_links = []
        self._llm = llm


    def _get_html_selector(self):
        if self.context_types == ContextTypes.DOCUSAURUS:
            return {'class': 'theme-doc-markdown'}
        return {}

    def _get_all_website_links(self, urls: list | None = None, idx: int = 0):
        if urls is None:
            urls = list()
            urls.append(self._base_path)

        url = urls[idx]
        response = requests.get(self._base_url + url)
        soup = BeautifulSoup(response.text, "html.parser")
        for a_tag in soup.findAll("a"):
            href = a_tag.attrs.get("href")
            if href == "" or href is None:
                continue
            if href.startswith(self._base_url):
                href = href.replace(self._base_url, '')
            href = href.split('#')[0]
            if href.startswith(self._base_path) and href not in urls:
                urls.append(href)

        idx += 1
        if idx < len(urls):
            return self._get_all_website_links(urls, idx)
        return list(dict.fromkeys(urls))

    def _make_file_name(self, text, idx=None):
        if text.startswith(self._base_url):
            text = text[len(self._base_url):]
        if text.endswith('.txt'):
            text = text[:-4]
        if idx is None:
            return re.sub(r'/', '_', text) + '.txt'
        return re.sub(r'/', '_', text) + '.' + str(idx) + '.txt'

    def _extract_text_from_web(self, url_docu):
        if not url_docu.startswith(self._base_url):
            url_docu = self._base_url + url_docu
        response = requests.get(url_docu)
        soup = BeautifulSoup(response.text, 'html.parser')
        div = soup.find('div', self._get_html_selector())
        h = html2text.HTML2Text()
        if not div:
            res = h.handle(str(soup))
        else:
            res = h.handle(str(div))
        return re.sub(r'^\n|\n$', '', re.sub('â|Â ', '', res))

    @staticmethod
    def _fp(path_name: str) -> str:
        return os.path.join(os.getcwd(), path_name)

    @classmethod
    def docu_root(cls) -> str:
        return cls._fp('chat_bot_docu')

    def _handle_text_chunk(self, link: str, txt: str, log_handler: Callable[[str, dict], None] | None = None):
        fn = self._make_file_name(link)
        file_path = os.path.join(self.docu_root(), fn)
        txt = self._prepare_text(txt)
        with open(file_path, 'w+') as f:
            f.write(txt)
        self._docu_links.append(file_path)
        log_handler and log_handler('links-meta', {'text': f"{file_path} (Length {len(txt)})"})
        return [txt]

    def _handle_long_text_chunk(self, link: str, text: str, log_handler: Callable[[str, dict], None] | None = None):  #
        main_header = text.split('\n')[0]
        text_chunks = []
        new_text = main_header
        sub_sections = re.split(r'\n## ', text)
        for idx, text_part in enumerate(sub_sections):
            new_text += '\n## ' + text_part.strip('#')

            if len(sub_sections) - 1 == idx or len(new_text) > self._chunk_size:
                chunk_idx = 0
                chunk_step = int(self._chunk_size * 1.2 // 1)
                while chunk_idx < len(new_text):
                    fn = self._make_file_name(link, f"{idx}_{chunk_idx // self._chunk_size}")
                    chunk_end = chunk_idx + chunk_step
                    text_chunks += self._handle_text_chunk(fn, new_text[chunk_idx:chunk_end], log_handler)
                    chunk_idx += self._chunk_size
                new_text = main_header

        return text_chunks

    def fetch_documents(self, log_handler: Callable[[str, dict], None] | None = None):
        links = self._get_all_website_links()
        if os.path.isdir(self.docu_root()):
            shutil.rmtree(self.docu_root())
        os.makedirs(self.docu_root(), exist_ok=True)
        index = {'links': self._docu_links}
        log_handler and log_handler('meta', {'len': str(len(links))})
        text_chunks = []
        for _idx, link in enumerate(links):
            text = self._extract_text_from_web(link)
            log_handler and log_handler('links', {'text': f'[{_idx + 1}/{len(links)}] {link} (Lenght: {len(text)})',
                                                  'idx': _idx})
            if len(text) > self._chunk_size:
                text_chunks += self._handle_long_text_chunk(link, text, log_handler)
            else:
                text_chunks += self._handle_text_chunk(link, text, log_handler)

        with open(os.path.join(self.docu_root(), 'index.json'), 'w+') as f:
            f.write(json.dumps(index))
        log_handler and log_handler(f'chunks_path', {'text': self.docu_root()})
        self._index_chunks(log_handler, text_chunks)

    def index_chunks(self, log_handler):
        text_chunks = []
        for dl in self.get_document_links():
            with open(dl, 'r') as f:
                text_chunks.append(f.read())
        self._index_chunks(log_handler, text_chunks)

    def generate_questions(self, log_handler):
        q_root = os.path.join(self.docu_root(), 'questions')
        shutil.rmtree(q_root, ignore_errors=True)
        os.makedirs(q_root, exist_ok=True)
        for dl in self.get_document_links():
            log_handler and log_handler(f'generate_questions', {'text': f'Generateing questions for: {dl}'})
            with open(dl, 'r') as f:
                question_chunk = self._generate_questions(f.read())
            log_handler and log_handler(f'generated_questions', {'text': question_chunk})
            q_file_path = os.path.join(q_root, os.path.basename(dl))
            with open(q_file_path, 'w+') as f:
                f.write(question_chunk)



    def get_document_links(self):
        if len(self._docu_links) == 0:
            json_path = os.path.join(self.docu_root(), 'index.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self._docu_links = json.loads(f.read())['links']
            else:
                self.fetch_documents()
        return self._docu_links

    def get_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _index_chunks(self, log_handler, text_chunks):
        # Load pre-trained model
        model = self.get_embedding_model()  # Lightweight and efficient
        # Generate embeddings
        embeddings = model.encode(text_chunks, convert_to_tensor=True)
        # Convert embeddings to numpy array
        embeddings_np = embeddings.cpu().detach().numpy()
        # Create a FAISS index
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
        index.add(embeddings_np)
        # Save the index for later use
        idx_bin_path = os.path.join(self.docu_root(), "faiss_index.bin")
        faiss.write_index(index, idx_bin_path)
        log_handler and log_handler(f'index', {'text': f'FAISS index path {idx_bin_path}'})

    def _prepare_text(self, text: str) -> str:
        if os.getenv('ONLY_SAMPLE_ANSWER', 'f').lower() == 'true':
            return text
        prompt = f"The following text is the documentation chunk for the Chemotion ELN. Summarize the following text into a concise, high-quality text while retaining key details: {text}"
        text = query_ollama(prompt, self._llm, False)['answer']
        prompt = f"Rewrite the following text to be scientifically sound such that it give a clear information to its reader: {text}"
        text = query_ollama(prompt, self._llm, False)['answer']
        prompt = f"Does the following text need more context or additional details? If yes, suggest improvements: {text}"
        text = query_ollama(prompt, self._llm, False)['answer']
        prompt = f"Eliminate redundant information from the following text while preserving meaning: {text}"
        text = query_ollama(prompt, self._llm, False)['answer']
        prompt = f"Include missing domain knowledge: {text}"
        return query_ollama(prompt, self._llm, False)['answer']

    def _generate_questions(self, text: str) -> str:
        if os.getenv('ONLY_SAMPLE_ANSWER', 'f').lower() == 'true':
            return text
        # "Generate a list of 10 realistic questions that a user of the Chemotion ELN system might ask based on this documentation.\nFor each question:\n1. Ensure it reflects practical, system-related scenarios.\n2.Provide a clear, accurate, and concise answer that an IT support staff member would typically deliver to address the query.\nMake the questions user-focused and answers professional yet accessible to someone with basic technical knowledge.\n Using the provided Chemotion ELN documentation excerpt:{context}"

        prompt = f"Based on the provided Chemotion ELN documentation chunk:\n\n<context>\n\n {text} \n\n</context>\n\n, generate a list of 10 realistic questions that a system user might ask. For each question, provide a clear, accurate, and concise answer that an IT support staff member would typically give in response."
        return query_ollama(prompt, self._llm)['answer']
