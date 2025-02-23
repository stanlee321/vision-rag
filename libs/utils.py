import os
import re

from db.chroma import ChromaDBClient
from smart_llm_loader import SmartLLMLoader

from llama_index.llms.openai import OpenAI
# from llama_index.llms.groq import Groq
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import Document as LlamaDocument
from llama_index.vector_stores.chroma import ChromaVectorStore

from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex, download_loader, StorageContext

from typing import Union, Tuple

from anyio import to_thread

def sanitize_metadata(metadata: dict, doc_type: str) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = str(value)
    # If there's a file_path, add file_name as its basename
    if "file_path" in sanitized and isinstance(sanitized["file_path"], str):
        sanitized["file_name"] = os.path.basename(sanitized["file_path"])
    
    # Add doc_type to metadata
    sanitized["doc_type"] = doc_type
    return sanitized

def transform_metadata(raw_metadata: dict, doc_type: Union[str, None]) -> list:
    uuid_regex = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    )
    results = []
    for key, metadata in raw_metadata.items():
        if isinstance(metadata, dict):
            if isinstance(key, str) and uuid_regex.match(key):
                entry = {"doc_id": key}
            else:
                entry = {"doc_id": None}
            # Sanitize metadata and add file_name if available
            if doc_type is not None:
                entry.update(sanitize_metadata(metadata, doc_type))
            else:
                entry.update(metadata)
            results.append(entry)
    return results


def convert_langchain_to_llama_docs(lc_docs, doc_type: str):
    return [
        LlamaDocument(
            text=doc.page_content, 
            metadata=sanitize_metadata(doc.metadata, doc_type))
        for doc in lc_docs
    ]


def get_llm(provider: str, model_name: str):

    return OpenAI(model_name=model_name, api_key=os.environ["OPENAI_API_KEY"])
    
    
def get_embed_model(provider: str):
    if provider == "openai":
        return OpenAIEmbedding(
            model_name="text-embedding-3-large", 
            api_key=os.environ["OPENAI_API_KEY"]
        )
    elif provider == "ollama":
        return OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
    else:
        return "No provider selected"
    
    

async def process_pdf(
    chroma_client: ChromaDBClient,
    file_path: str, 
    collection_name: str, 
    loader_type: str = "pymupdf", 
    vision_model: str = "gemini/gemini-1.5-flash",
    doc_type: str = "GENERIC",
    api_key: str = None
) -> Tuple[VectorStoreIndex, str]:
    """
    Process a PDF file and return a VectorStoreIndex.

    Args:
        chroma_client: The ChromaDB client.
        file_path: The path to the PDF file.
        collection_name: The name of the collection to upload the document to.
        loader_type: The type of loader to use to load the document.
        vision_model: The vision model to use to load the document.
        doc_type: The type of the document.
        api_key: The API key to use to load the document.

    Returns:
        A VectorStoreIndex.
    """
    # Get (or create) a collection in ChromaDB
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    documents_size = 0
    # Choose loader based on loader_type query parameter
    if loader_type.lower() == "smart":
        loader = SmartLLMLoader(
            file_path=file_path,
            chunk_strategy="contextual",
            model=vision_model,
            api_key = api_key
        )
        # Run loader.load_and_split() in a separate thread to avoid nested event loops
        docs = await to_thread.run_sync(loader.load_and_split)
        docs = convert_langchain_to_llama_docs(docs, doc_type)
        documents_size = len(docs)
    else:
        PyMuPDFReader = download_loader("PyMuPDFReader")
        docs = PyMuPDFReader().load_data(file_path)
        
        # Add doc_type to metadata
        for doc in docs:
            doc.metadata["doc_type"] = doc_type
        
        documents_size = len(docs)

    # Build (or update) the index using the parsed documents
    index = VectorStoreIndex.from_documents(
        docs, 
        vector_store=vector_store,
        storage_context=storage_context,
        transformations=[
            SentenceSplitter(chunk_size=1000, chunk_overlap=200)
        ],
        show_progress=True
    )
    return index, documents_size
