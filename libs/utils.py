import os
import re


from llama_index.llms.openai import OpenAI
# from llama_index.llms.groq import Groq
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding


from typing import Union, List


def sanitize_metadata(metadata: dict, doc_type: str) -> dict:
    sanitized = {}
    print(f"Sanitizing metadata: {metadata}")
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

def transform_metadata(raw_metadata: dict, doc_type: Union[str, None]) -> List[dict]:
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


def get_llm(provider: str, model_name: str):
    return OpenAI(model_name=model_name, api_key=os.environ["OPENAI_API_KEY"])

    
def get_embed_model(provider: str, llm_embeddings_model: str):
    if provider == "openai":
        return OpenAIEmbedding(
            model_name=llm_embeddings_model, 
            api_key=os.environ["OPENAI_API_KEY"]
        )
    elif provider == "ollama":
        return OllamaEmbedding(
            model_name=llm_embeddings_model,
            base_url="http://localhost:11434",
            ollama_additional_kwargs={"mirostat": 0},
        )
    else:
        return "No provider selected"
    
 
