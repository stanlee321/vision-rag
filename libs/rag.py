import os
from fastapi import UploadFile
from tempfile import gettempdir

from db.chroma import ChromaDBClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from fastapi import HTTPException

from libs.utils import process_pdf, transform_metadata
from libs.data import response_mode_dict


class RagAPI:
    """
    A class for the RAG API.
    """
    def __init__(self, chroma_client: ChromaDBClient, qa_template: PromptTemplate, openai_api_key: str, vision_model: str):
        self.chroma_client = chroma_client
        self.qa_template = qa_template
        self.openai_api_key = openai_api_key
        self.vision_model = vision_model

    async def upload_document(self, file: UploadFile, collection_name: str, doc_type: str, loader: str):
        """
        Upload a document to the RAG API.

        Args:
            file: The file to upload.
            collection_name: The name of the collection to upload the document to.
            doc_type: The type of the document.
            loader: The loader to use to load the document.

        Returns:
            A message indicating that the file has been uploaded and processed.
        """
        print(f"Uploading document to collection: {collection_name}")
        print(f"Document type: {doc_type}")
        print(f"Loader: {loader}")
        print(f"File: {file}")
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        try:
            # Create a directory for uploads if it doesn't exist
            upload_dir = os.path.join(gettempdir(), "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Use the original filename but ensure it's safe
            safe_filename = os.path.basename(file.filename)
            file_path = os.path.join(upload_dir, safe_filename)
            file_path = file_path.split(".")[0] + ".pdf"
            # Write the file with original name
            contents = await file.read()
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            try:
                _, documents_size = await process_pdf(
                    self.chroma_client, file_path, collection_name,
                    loader_type=loader, vision_model=self.vision_model,
                    doc_type=doc_type, api_key=self.openai_api_key
                )
                print("File processed successfully, at file_path: ", file_path)
                print(f"Documents size: {documents_size}")
            finally:
                # Clean up the file after processing
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            return {
                "message": f"File uploaded and processed into collection '{collection_name}' using loader '{loader}'.",
                "status": "success",
                "documents_size": documents_size
            }
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

    def query_documents(self, q: str, doc_type: str, collection_name: str, response_mode: str):
        """
        Query the RAG API for a question.

        Args:
            q: The question to query the RAG API with.
            doc_type: The type of the document to query the RAG API with.
            collection_name: The name of the collection to query the RAG API with.
            response_mode: The response mode to use for the query.

        Returns:
            A message indicating that the query has been processed.
        """
        coll = self.chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=coll)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex([], vector_store=vector_store, storage_context=storage_context)
        if doc_type:
            filters = MetadataFilters(filters=[
                ExactMatchFilter(key="doc_type", value=doc_type)
            ])
            query_engine = index.as_query_engine(
                text_qa_template=self.qa_template,
                response_mode=response_mode,
                similarity_top_k=3,
                verbose=True,
                filters=filters
            )
        else:
            query_engine = index.as_query_engine(
                text_qa_template=self.qa_template,
                response_mode=response_mode,
                similarity_top_k=3,
                verbose=True
            )
        try:
            response = query_engine.query(q)
            print(f"Response from query: {response}")
            if response.metadata:
                metadata = transform_metadata(response.metadata, doc_type=None)
            else:
                metadata = []
        except Exception as e:
            print(f"Query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
        return {"question": q, "answer": response.response, "metadata": metadata}

    def get_info(self):
        """
        Get information about the RAG API.

        Returns:
            A message indicating that the information has been processed.
        """
        return {"version": "1.0.0", "description": "RAG API", "supported_response_modes": response_mode_dict}

    def list_all_collections(self):
        """
        List all collections in the RAG API.

        Returns:
            A message indicating that the collections have been listed.
        """
        try:
            return {
                "collections": self.chroma_client.list_collections()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list collections: {str(e)}")


    def delete_collection(self, collection_name: str):
        """
        Delete a collection from the RAG API.
        """
        try:
            self.chroma_client.delete_collection(collection_name)
            return {"message": f"Collection '{collection_name}' deleted successfully."}
        except Exception as e:
            print(f"Error deleting collection: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")
