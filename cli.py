import os
import click
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

from embed import embed
from query import query
from get_vector_database import get_vector_db

TEMP_FOLDER = os.getenv('TEMP_FOLDER', 'tmp')
os.makedirs(TEMP_FOLDER, exist_ok=True)

logging.basicConfig(filename='cli.log', level=logging.INFO)

@click.group()
def cli():
    """A command-line interface for managing vector database operations."""
    pass

@cli.command("add")
@click.argument("file_path", type=click.Path(exists=True, readable=True))
def add_file(file_path):
    """
    Add a PDF file to the vector database.

    Example:
        python cli.py add ./data/myfile.pdf
    """
    file_path = Path(file_path)
    if not file_path.suffix.lower().endswith(".pdf"):
        click.echo("Only PDF files are supported.")
        return

    class DummyFile:
        """A lightweight wrapper to simulate a file-like object for embed()."""
        def __init__(self, path):
            self.filename = path.name
            self.path = path
        def save(self, dest):
            os.system(f"cp '{self.path}' '{dest}'")

    dummy_file = DummyFile(file_path)
    click.echo(f"Embedding file: {file_path.name} ...")

    success = embed(dummy_file)
    if success:
        click.echo("File embedded successfully.")
    else:
        click.echo("Embedding failed.")


@cli.command("delete")
def delete_collection():
    """
    Delete a collection from the vector database.

    Example:
        python cli.py delete
    """
    db = get_vector_db()

    db.delete_collection()
    click.echo(f"Collection deleted successfully.")

@cli.command("query")
@click.argument("file_path", type=click.Path(exists=True, readable=True))
@click.argument("output", type=click.File('w'))
def query_rag(file_path, output):
    """
    file_path: a path to file with questions to ask from the RAG system.
    Each line will be its own question.
    output: name of file to write query and response to (jsonl)

    Example:
        python cli.py query ./data/myfile.txt output.jsonl"
    """
    file_path = Path(file_path)
    with file_path.open() as f:
        for line in f:
            query_cleaned = line.strip().strip('\n')
            response = query(query_cleaned)
            if response:
                output.write(json.dumps({"query": query_cleaned, "message": response}) + '\n')
            logging.exception(response)


if __name__ == "__main__":
    cli()