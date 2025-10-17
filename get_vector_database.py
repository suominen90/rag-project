import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.chroma import Chroma


def get_vector_db(
    chroma_path: str | None = None,
    collection_name: str | None = None,
    text_embedding_model: str | None = None,
    show_progress: bool = True,
) -> Chroma:
    """
    Initialize and return a Chroma vector database using Ollama embeddings.

    This function sets up a Chroma vector store for efficient similarity search
    and retrieval in Retrieval-Augmented Generation (RAG) pipelines. It supports
    dependency injection by allowing configuration values to be passed as parameters,
    while still providing environment variable fallbacks.

    Args:
        chroma_path (str, optional): Directory for Chroma persistence.
            Defaults to the 'CHROMA_PATH' environment variable or "chroma".
        collection_name (str, optional): Name of the Chroma collection.
            Defaults to 'COLLECTION_NAME' environment variable or "local-rag".
        text_embedding_model (str, optional): Name of the embedding model used.
            Defaults to 'TEXT_EMBEDDING_MODEL' environment variable or "nomic-embed-text".
        show_progress (bool, optional): Whether to display progress while generating embeddings.
            Defaults to True.

    Returns:
        Chroma: An initialized Chroma vector database configured with
        the specified or environment-based settings.
    """

    chroma_path = chroma_path or os.getenv("CHROMA_PATH", "chroma")
    collection_name = collection_name or os.getenv("COLLECTION_NAME", "local-rag")
    text_embedding_model = text_embedding_model or os.getenv("TEXT_EMBEDDING_MODEL", "nomic-embed-text")

    embedding = OllamaEmbeddings(
        model=text_embedding_model,
        show_progress=show_progress,
    )

    db = Chroma(
        collection_name=collection_name,
        persist_directory=chroma_path,
        embedding_function=embedding,
    )

    return db
