import os
import asyncio
import nest_asyncio
nest_asyncio.apply()  # Enable nested asyncio event loops

from fastapi import FastAPI, UploadFile, File, Query, Form, Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from db.chroma import ChromaDBClient
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from libs.data import template
from libs.rag import RagAPI

from dotenv import load_dotenv

load_dotenv()

# Add API token configuration
API_TOKEN = os.getenv("API_TOKEN", "1234")  # Default token is "1234", but better to set in .env
security = HTTPBearer()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_CLIENT_AUTH_CREDENTIALS = os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")
CHROMA_SERVER_AUTHN_PROVIDER = os.getenv("CHROMA_SERVER_AUTHN_PROVIDER")
CHROMA_AUTH_TOKEN_TRANSPORT_HEADER = os.getenv("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TIMEOUT = int(os.getenv("TIMEOUT", "600"))
USE_METADATA = int(os.getenv("USE_METADATA", 0))

use_metadata_pipeline = True if USE_METADATA==1 else False


# Print all environment variables
print("Environment variables:")
for key, value in os.environ.items():
    print(f"{key}: {value}")

qa_template = PromptTemplate(template)
chroma_client = ChromaDBClient(
    host=CHROMA_HOST, 
    port=CHROMA_PORT, 
    auth_credentials=CHROMA_CLIENT_AUTH_CREDENTIALS,
    auth_provider=CHROMA_SERVER_AUTHN_PROVIDER,
    auth_token_transport_header=CHROMA_AUTH_TOKEN_TRANSPORT_HEADER
)

rag_api = RagAPI(chroma_client, qa_template, OPENAI_API_KEY, VISION_MODEL, use_metadata_pipeline = use_metadata_pipeline)

# Settings.llm = get_llm(provider=AI_PROVIDER, model_name=LLM_MODEL)
# Settings.embed_model = get_embed_model(provider=AI_PROVIDER)

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> bool:
    """
    Verify the Bearer token from the request
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

app = FastAPI()

@app.post("/v1/rag/upload")
async def upload_endpoint(
    file: UploadFile = File(..., description="PDF file to upload"),
    collection_name: str = Form(
        default="default_collection",
        description="Name of the collection to store the document"
    ),
    doc_type: str = Form(
        default="GENERIC",
        description="Type of the document being uploaded"
    ),
    loader: str = Form(
        default="pymupdf",
        description="Loader to use for processing the document"
    ),
    authenticated: bool = Depends(verify_token)
):
    print(f"Uploading document to collection: {collection_name}")
    print(f"Document type: {doc_type}")
    print(f"Loader: {loader}")
    print(f"File: {file}")

    result = await asyncio.wait_for(
        rag_api.upload_document(file, collection_name, doc_type, loader),
        timeout=TIMEOUT
    )
    return result

@app.get("/v1/rag/query")
def query_endpoint(
    q: str = Query(...),
    doc_type: str = Query(None),
    collection_name: str = Query("default_collection"),
    response_mode: str = Query("compact"),
    authenticated: bool = Depends(verify_token)
):
    print(f"Querying document: {q}")
    print(f"Document type: {doc_type}")
    print(f"Collection name: {collection_name}")
    print(f"Response mode: {response_mode}")
    return rag_api.query_documents(q, doc_type, collection_name, response_mode)

@app.get("/v1/rag/info")
def info_endpoint(authenticated: bool = Depends(verify_token)):
    return rag_api.get_info()

@app.get("/v1/rag/collections")
def collections_endpoint(authenticated: bool = Depends(verify_token)):
    return rag_api.list_all_collections()


# Delete collection by name
@app.delete("/v1/rag/collections/{collection_name}")
def delete_collection_endpoint(collection_name: str, authenticated: bool = Depends(verify_token)):
    print(f"Deleting collection: {collection_name}")
    return rag_api.delete_collection(collection_name)

@app.post("/v1/translate/to-spanish")
def translate_to_spanish_endpoint(
    text: str = Query(..., description="Text to translate to Spanish"),
    authenticated: bool = Depends(verify_token)
):
    """
    Translate text from any language to Spanish.
    
    Args:
        text: The text to translate.
        
    Returns:
        The translated text in Spanish.
    """
    print(f"Translating text to Spanish: {text[:100]}...")
    return rag_api.translate_to_spanish(text)

@app.post("/v1/translate")
def translate_endpoint(
    text: str = Query(..., description="Text to translate"),
    target_language: str = Query("Spanish", description="Target language for translation"),
    authenticated: bool = Depends(verify_token)
):
    """
    Translate text from any language to the specified target language.
    
    Args:
        text: The text to translate.
        target_language: The target language for translation (default: Spanish).
        
    Returns:
        The translated text in the target language.
    """
    print(f"Translating text to {target_language}: {text[:100]}...")
    return rag_api.translate_text(text, target_language)