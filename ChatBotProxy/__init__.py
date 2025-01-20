import os

import click
from dotenv import load_dotenv, find_dotenv

from ChatBotProxy.main_engine.import_docu import ContextManager
from ChatBotProxy.main_engine.query_ollama import query_ollama, build_question_prompt
from ChatBotProxy.run_gunicorn import run

# Load environment variables from the .env file
load_dotenv(find_dotenv(usecwd=True))


@click.group()
def cli():
    pass


@cli.command(help="Fetch and update documentation of Chemotion")
@click.option('--url', '-u', default=os.getenv('DOCUSAURUS_URL'), help="Docusaurus url")
@click.option('--path', '-p', default=os.getenv('DOCUSAURUS_BASE_PATH'), help="Docusaurus url base path")
@click.option('--embedding_model', '-em', default=os.getenv('EMBEDDING_MODEL'), help="Docusaurus url base path")
@click.option('--llm_model', '-llm', default=os.getenv('LLM_MODEL'), help="Docusaurus url base path")
def update(url, path, embedding_model, llm_model):
    ContextManager().setup(embedding_model, url, llm_model, path)

    def print_update(method_type: str, value: dict):
        if method_type == 'meta':
            click.echo(f"Number of all links: {value['len']}")
        if method_type in ['links', 'index', 'links-meta']:
            click.echo(f"{method_type} -> {value['text']}")

    ContextManager().fetch_documents(print_update)


@cli.command(help="Ask ollama")
@click.option('--question', '-q', prompt="What do you want to know?", help="Question to be answered")
@click.option('--llm_model', '-llm', default=os.getenv('LLM_MODEL'), help="LLM model name")
@click.option('--url', '-u', default=os.getenv('DOCUSAURUS_URL'), help="Docusaurus url")
@click.option('--path', '-p', default=os.getenv('DOCUSAURUS_BASE_PATH'), help="Docusaurus url base path")
@click.option('--embedding_model', '-em', default=os.getenv('EMBEDDING_MODEL'), help="Docusaurus url base path")
def answer(url, path, embedding_model, question, llm_model):
    ContextManager().setup(embedding_model, url, llm_model, path)
    prompt = build_question_prompt(question)
    res = query_ollama(prompt, llm_model)
    click.echo(res['answer'])


@cli.command(help="Serve proxy server")
def serve():
    run()

if __name__ == '__main__':
    cli()
