import os

from db.chroma import ChromaDBClient
from fastapi import FastAPI, UploadFile, File, Query

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate

from libs.data import template
from libs.utils import get_llm, get_embed_model
from libs.rag import RagAPI

from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_CLIENT_AUTH_CREDENTIALS = os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")
CHROMA_SERVER_AUTHN_PROVIDER = os.getenv("CHROMA_SERVER_AUTHN_PROVIDER")
CHROMA_AUTH_TOKEN_TRANSPORT_HEADER = os.getenv("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

rag_api = RagAPI(chroma_client, qa_template, OPENAI_API_KEY, VISION_MODEL)

Settings.llm = get_llm(provider=AI_PROVIDER, model_name=LLM_MODEL)
Settings.embed_model = get_embed_model(provider=AI_PROVIDER)


app = FastAPI()

@app.post("/v1/rag/upload")
async def upload_endpoint(
    file: UploadFile = File(...),
    collection_name: str = Query("default_collection"),
    doc_type: str = Query("GENERIC"),
    loader: str = Query("pymupdf")
):
    return await rag_api.upload_document(file, collection_name, doc_type, loader)

@app.get("/v1/rag/query")
def query_endpoint(
    q: str = Query(...),
    doc_type: str = Query(None),
    collection_name: str = Query("default_collection"),
    response_mode: str = Query("compact")
):
    return rag_api.query_documents(q, doc_type, collection_name, response_mode)

@app.get("/v1/rag/info")
def info_endpoint():
    return rag_api.get_info()

@app.get("/v1/rag/collections")
def collections_endpoint():
    return rag_api.list_all_collections()