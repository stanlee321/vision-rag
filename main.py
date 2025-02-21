import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from tempfile import NamedTemporaryFile

from db.chroma import ChromaDBClient

from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters


from libs.data import (response_mode_dict, 
                  template)

from libs.utils import (get_llm, 
                        get_embed_model, 
                        transform_metadata,
                        process_pdf)

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

VISION_MODEL = os.getenv("VISION_MODEL", "openai/gpt-4o")
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
CHROMA_CLIENT_AUTH_CREDENTIALS = os.getenv("CHROMA_CLIENT_AUTH_CREDENTIALS")
CHROMA_SERVER_AUTHN_PROVIDER = os.getenv("CHROMA_SERVER_AUTHN_PROVIDER")
CHROMA_AUTH_TOKEN_TRANSPORT_HEADER = os.getenv("CHROMA_AUTH_TOKEN_TRANSPORT_HEADER")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(CHROMA_HOST, CHROMA_PORT, CHROMA_CLIENT_AUTH_CREDENTIALS, CHROMA_SERVER_AUTHN_PROVIDER, CHROMA_AUTH_TOKEN_TRANSPORT_HEADER, OPENAI_API_KEY)

qa_template = PromptTemplate(template)

# Initialize chromadb client (persistent storage if desired)
chroma_client = ChromaDBClient(host=CHROMA_HOST, 
                        port=CHROMA_PORT, 
                        auth_credentials=CHROMA_CLIENT_AUTH_CREDENTIALS,
                        auth_provider=CHROMA_SERVER_AUTHN_PROVIDER,
                        auth_token_transport_header=CHROMA_AUTH_TOKEN_TRANSPORT_HEADER)

# Setup AI 
Settings.llm = get_llm(provider=AI_PROVIDER)
Settings.embed_model = get_embed_model(provider=AI_PROVIDER)

app = FastAPI()


@app.post("/v1/rag/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Query("default_collection"),
    doc_type: str = Query("GENERIC", description="Specify the type of document to be processed"),
    loader: str = Query("pymupdf", description="Specify 'smart' to use smart-llm-loader or 'pymupdf' for default")
):
    # Accept only PDF files
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    # Save the uploaded file to a temporary file
    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp_path = tmp.name
            contents = await file.read()
            tmp.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to save file")
    # Process the PDF using the chosen loader type
    try:
        await process_pdf(chroma_client, 
                          tmp_path, 
                          collection_name, 
                          loader_type=loader, 
                          vision_model=VISION_MODEL, 
                          doc_type=doc_type,
                          api_key=OPENAI_API_KEY)
    except Exception as e:
        print(e)
        os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    os.unlink(tmp_path)  # Remove the temporary file
    return {"message": f"File uploaded and processed into collection '{collection_name}' using loader '{loader}'."}

@app.get("/v1/rag/query")
def query_documents(
    q: str = Query(..., description="Query string"),
    doc_type: str = Query(None, description="Specify the type of document to be processed"),
    collection_name: str = Query("default_collection", description="Name of the collection to query"),
    response_mode: str = Query("compact", description="Specify the response mode")
):
    print("querying:", q)
    # Retrieve (or create) the collection and wrap it in a ChromaVectorStore
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an index object with an empty list—this lets the index query the persisted vector store
    index = VectorStoreIndex([], 
                             vector_store=vector_store,
                             storage_context=storage_context)
    
    if doc_type:
        filters = MetadataFilters(filters=[
            ExactMatchFilter(
                    key="doc_type", 
                    value=doc_type,
                ),
            ])
        query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            response_mode=response_mode,
            similarity_top_k=3,
            verbose=True,
            filters=filters
        )
    else:
        query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            response_mode=response_mode,
            similarity_top_k=3,
            verbose=True
        )
    try:
        response = query_engine.query(q)
        metadata = transform_metadata(response.metadata, doc_type = None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
    
    return {
            "question": q, 
            "answer": response.response, 
            "metadata": metadata
        }
    
    
@app.get("/v1/rag/info")
def get_info():
    return {
        "version": "1.0.0",
        "description": "RAG API",
        "supported_response_modes": response_mode_dict
    }


@app.get("/v1/rag/collections")
def list_all_collections():
    try:
        collections = chroma_client.list_collections()  # returns collection objects
        # Convert each collection object to its name (or a dict if needed)
        collection_names = [getattr(coll, "name", str(coll)) for coll in collections]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")
    return {"collections": collection_names}