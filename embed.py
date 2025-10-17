import os
from datetime import datetime
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from get_vector_database import get_vector_db

# Chunking parameters
CHUNK_SIZE = os.getenv('CHUNK_SIZE', 7500)
CHUNK_OVERLAP = os.getenv('CHUNK_OVERLAP', 100)

# Default temp folder from environment variable
TEMP_FOLDER = os.getenv("TEMP_FOLDER", "tmp")


def allowed_file(filename: str, allowed_extensions: set[str] | None = None) -> bool:
    """
    Check if the uploaded file has an allowed extension.

    Args:
        filename (str): The name of the uploaded file.
        allowed_extensions (set[str], optional): Allowed file extensions.
            Defaults to {'pdf'}.

    Returns:
        bool: True if the file extension is allowed, False otherwise.
    """
    allowed_extensions = allowed_extensions or {"pdf"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions


def save_file(file, temp_folder: str | None = None) -> str:
    """
    Save the uploaded file to a temporary folder with a unique timestamped name.

    Args:
        file: The uploaded file object (e.g., from Flask request.files).
        temp_folder (str, optional): Directory to save temporary files.
            Defaults to the 'TEMP_FOLDER' environment variable or 'tmp'.

    Returns:
        str: The full path to the saved file.
    """
    temp_folder = temp_folder or TEMP_FOLDER
    os.makedirs(temp_folder, exist_ok=True)

    timestamp = datetime.now().timestamp()
    filename = f"{timestamp}_{secure_filename(file.filename)}"
    file_path = os.path.join(temp_folder, filename)
    file.save(file_path)

    return file_path


def load_and_split_data(
    file_path: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    loader_class=UnstructuredPDFLoader,
    splitter_class=RecursiveCharacterTextSplitter,
):
    """
    Load a PDF file and split its content into text chunks for embedding.

    Args:
        file_path (str): Path to the PDF file.
        chunk_size (int, optional): Maximum number of characters per chunk.
            Defaults to the CHUNK_SIZE environment variable or 7500.
        chunk_overlap (int, optional): Number of overlapping characters between chunks.
            Defaults to the CHUNK_OVERLAP environment variable or 100.
        loader_class (type, optional): Loader class used to read the PDF file.
            Defaults to UnstructuredPDFLoader.
        splitter_class (type, optional): Text splitter class for chunking.
            Defaults to RecursiveCharacterTextSplitter.

    Returns:
        list: A list of document chunks ready for embedding.
    """
    # Environment variables are strings; need to cast as int
    chunk_size = chunk_size or int(CHUNK_SIZE)
    chunk_overlap = chunk_overlap or int(CHUNK_OVERLAP)

    loader = loader_class(file_path=file_path)
    data = loader.load()

    splitter = splitter_class(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(data)

    return chunks


def embed(
    file,
    db_factory=get_vector_db,
    temp_folder: str | None = None,
    allowed_extensions: set[str] | None = None,
) -> bool:
    """
    Handle the embedding process for an uploaded file:
    validate, save, process, embed, and clean up.

    Args:
        file: The uploaded file object.
        db_factory (callable, optional): Function returning a Chroma database instance.
            Defaults to get_vector_db.
        temp_folder (str, optional): Temporary directory for saving files.
            Defaults to 'TEMP_FOLDER' environment variable or 'tmp'.
        allowed_extensions (set[str], optional): Allowed file types.
            Defaults to {'pdf'}.

    Returns:
        bool: True if embedding was successful, False otherwise.
    """
    if file and file.filename and allowed_file(file.filename, allowed_extensions):
        file_path = save_file(file, temp_folder=temp_folder)
        try:
            chunks = load_and_split_data(file_path)
            db = db_factory()
            db.add_documents(chunks)
            db.persist()
        finally:
            os.remove(file_path)

        return True

    return False
