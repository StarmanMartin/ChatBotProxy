# ChatBotProxy

This tool is a proxy system between a chatbot server and an Ollama instance.
It retrieves all information from a docusaurus instance and prepares it in such a way that the
ollama LLM can process it as efficiently as possible.

The last part of preprocessing is indexing with FIASS pipline.

The ollama instance must be accessible from the ChatBotProxy at http://localhost:11434/api/generate

## Install

1) Clone the project
2) cd into the project
3) run:
```shell
$ pip install dist/chatbotproxy-0.1.0.tar.gz
```

## CLI Tool

Usage: ChatBotProxy \[OPTIONS] COMMAND \[ARGS]...

Options:<br>
--help ->  Show this message and exit.

Commands:
- answer  Ask ollama<br>
  Args:
  - -q, --question | TEXT | Question to be answered
  - --llm | TEXT | LLM model name
  - -u, --url | TEXT | Docusaurus url
  - -p, --path | TEXT | Docusaurus url base path
  - -em, --embedding_model | TEXT | FIASS model
  - --help           ->            Show this message and exit.

- update  Fetch and update documentation of Chemotion<br>
  Args:
    - -em, --embedding_model | TEXT | FIASS model
    - -u, --url | TEXT | Docusaurus url
    - -p, --path | TEXT  | Docusaurus url base path
    - --help        ->            Show this message and exit.
- serve   Serve proxy server (needs .env)<br>
  Args:
    - --help          ->          Show this message and exit.


The arguments only need to be set if there is no .env file. See the following section Run Server

## RUN Server

For all option generate a .env file in your cwd:

```dotenv
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=phi4
DOCUSAURUS_URL=https://chemotion.net
DOCUSAURUS_BASE_PATH=/docs

FLASK_ENV=development
FLASK_APP=ChatBotProxy.app:app
FLASK_DEBUG=True
SECRET_KEY=your_secret_key
HOST=0.0.0.0
PORT=8000
```

### Option 1

Clone the project

```shell
 gunicorn -c ChatBotProxy/gunicorn_config.py ChatBotProxy.app:app
```

### Option 2

Clone the project

```shell
pip install -r requirements.txt
python ChatBotProxy/run_gunicorn.py 
```


### Option 3

Install the project

```shell
ChatBotProxy serve
```
