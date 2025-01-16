import os

import click
from dotenv import load_dotenv

from ChatBotProxy.main_engine.import_docu import fetch_documents, get_document_links
from ChatBotProxy.main_engine.query_ollama import query_ollama
from ChatBotProxy.run_gunicorn import run

# Load environment variables from the .env file
load_dotenv()


@click.group()
def cli():
    pass


@cli.command(help="Fetch and update documentation of Chemotion")
@click.option('--url', '-u', default=os.getenv('DOCUSAURUS_URL'), help="Docusaurus url")
@click.option('--path', '-p', default=os.getenv('DOCUSAURUS_BASE_PATH'), help="Docusaurus url base path")
@click.option('--embedding_model', '-em', default=os.getenv('EMBEDDING_MODEL'), help="Docusaurus url base path")
def update(url, path, embedding_model):
    fetch_documents(url, path, embedding_model)


@cli.command(help="Ask ollama")
@click.option('--question', '-q', prompt="What do you want to know?", help="Question to be answered")
@click.option('--llm', default=os.getenv('LLM_MODEL'), help="LLM model name")
@click.option('--url', '-u', default=os.getenv('DOCUSAURUS_URL'), help="Docusaurus url")
@click.option('--path', '-p', default=os.getenv('DOCUSAURUS_BASE_PATH'), help="Docusaurus url base path")
@click.option('--embedding_model', '-em', default=os.getenv('EMBEDDING_MODEL'), help="Docusaurus url base path")
def answer(url, path, embedding_model, question, llm):
    query_ollama(question, llm, embedding_model, get_document_links(url, path, embedding_model))


@cli.command(help="Serve proxy server")
def serve():
    run()

if __name__ == '__main__':
    cli()
