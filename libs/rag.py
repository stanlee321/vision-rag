import os
from fastapi import UploadFile
from tempfile import gettempdir

from db.chroma import ChromaDBClient
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.prompts import PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)

from pprint import pprint
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import VectorStoreIndex, download_loader, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

from db.chroma import ChromaDBClient
from smart_llm_loader import SmartLLMLoader

from fastapi import HTTPException

from libs.utils import transform_metadata, get_llm, sanitize_metadata, get_embed_model
from libs.data import response_mode_dict
from anyio import to_thread
from typing import Tuple


class RagAPI:
    """
    A class for the RAG API.
    """
    def __init__(self, chroma_client: ChromaDBClient, qa_template: PromptTemplate, openai_api_key: str, vision_model: str):
        self.chroma_client = chroma_client
        self.qa_template = qa_template
        self.openai_api_key = openai_api_key
        self.vision_model = vision_model
        
        llm_transformations_provider = os.getenv("LLM_TRANSFORMATIONS_PROVIDER", "openai")
        llm_transformations_model = os.getenv("LLM_TRANSFORMATIONS_MODEL", "gpt-4o-mini")

        self.llm_transformations = get_llm(provider=llm_transformations_provider, 
                                           model_name=llm_transformations_model)

        llm_embeddings_provider = os.getenv("LLM_EMBEDDINGS_PROVIDER", "openai")
        llm_embeddings_model = os.getenv("LLM_EMBEDDINGS_MODEL", "text-embedding-3-large")

        self.llm_embedding = get_embed_model(provider=llm_embeddings_provider, llm_embeddings_model = llm_embeddings_model)
        
        llm_query_provider = os.getenv("LLM_QUERY_PROVIDER", "openai")
        llm_query_model = os.getenv("LLM_QUERY_MODEL", "gpt-4o-mini")
        

        self.llm_query = get_llm(provider=llm_query_provider, model_name = llm_query_model)
        
        self.llm_translate_model = os.getenv("LLM_TRANSLATE_MODEL", "gpt-4o-mini")
        
        
    def get_text_splitter(self):

        text_splitter = SentenceSplitter(
            separator=" ", chunk_size=1024, chunk_overlap=128
        )
        return text_splitter
    
    def get_title_extractor(self):
        title_extractor = TitleExtractor(llm=self.llm_transformations, nodes=5)
        return title_extractor
    
    def get_qa_extractor(self):
        qa_extractor = QuestionsAnsweredExtractor(llm=self.llm_transformations, questions=3)
        return qa_extractor

    def get_pipeline(self):
        
        pipeline = IngestionPipeline(
                transformations=[
                    self.get_text_splitter(),
                    self.get_title_extractor(),
                    self.get_qa_extractor()
                ]
            )
        return pipeline
    
    def convert_langchain_to_llama_docs(self, lc_docs, doc_type: str):
        return [
            LlamaDocument(
                text=doc.page_content, 
                metadata=sanitize_metadata(doc.metadata, doc_type))
            for doc in lc_docs
        ]


    async def process_pdf(
        self,
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
            docs = self.convert_langchain_to_llama_docs(docs, doc_type)
            documents_size = len(docs)
        else:
            PyMuPDFReader = download_loader("PyMuPDFReader")
            docs = PyMuPDFReader().load_data(file_path)
            
            # Add doc_type to metadata
            for doc in docs:
                doc.metadata["doc_type"] = doc_type
            
            documents_size = len(docs)
            
        pprint(docs[0].metadata)
        pipeline = self.get_pipeline()
        
        # Run the pipeline
        nodes = pipeline.run(
            documents=docs,
            in_place=True,
            show_progress=True,
        )

        index = VectorStoreIndex(
            nodes, 
            storage_context=storage_context, 
            embed_model=self.llm_embedding
        )

        # Build (or update) the index using the parsed documents
        # index = VectorStoreIndex.from_documents(
        #     docs, 
        #     vector_store=vector_store,
        #     storage_context=storage_context,
        #     transformations=[
        #         SentenceSplitter(chunk_size=1000, chunk_overlap=200)
        #     ],
        #     show_progress=True
        # )
        return index, documents_size
    
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
            # Extract only the filname and the extension
            file_name = os.path.splitext(safe_filename)[0]
            extension = os.path.splitext(safe_filename)[1]
            
            file_path = os.path.join(upload_dir, file_name) + extension
            
            # Write the file with original name
            contents = await file.read()
            with open(file_path, 'wb') as f:
                f.write(contents)
            
            try:
                _, documents_size = await self.process_pdf(
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
        
        # Make sure to use the same embedding model that was used for indexing
        index = VectorStoreIndex(
            [], 
            vector_store=vector_store, 
            storage_context=storage_context,
            embed_model=self.llm_embedding
        )
        
        self.qa_template = self.qa_template.partial_format(question=q)
        
        print(self.qa_template)
        if doc_type:
            filters = MetadataFilters(filters=[
                ExactMatchFilter(key="doc_type", value=doc_type)
            ])
            query_engine = index.as_query_engine(
                llm=self.llm_query,
                text_qa_template=self.qa_template,
                response_mode=response_mode,
                similarity_top_k=3,
                verbose=True,
                filters=filters
            )
        else:
            query_engine = index.as_query_engine(
                llm=self.llm_query,
                text_qa_template=self.qa_template,
                response_mode=response_mode,
                similarity_top_k=3,
                verbose=True
            )
        try:
            response = query_engine.query(q)
            print(f"Response from query: {response}")
            
            if hasattr(response, '__dict__'):
                print("METADATA:")
                import pprint
                pprint.pprint(response.__dict__)
                
            if response.metadata:
                metadata = transform_metadata(response.metadata, doc_type=None)
            else:
                metadata = []
        except Exception as e:
            print(f"Query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
        
        if response.response:
            response.response = self.translate_text(response.response, target_language="Spanish").get("translated")
        
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

    def translate_to_spanish(self, text: str) -> str:
        """
        Translate text from any language to Spanish using OpenAI API.
        
        Args:
            text: The text to translate to Spanish.
            
        Returns:
            The translated text in Spanish.
        """
        return self.translate_text(text, target_language="Spanish")
        
    def translate_text(self, text: str, target_language: str = "Spanish") -> dict:
        """
        Translate text from any language to the specified target language using OpenAI API.
        
        Args:
            text: The text to translate.
            target_language: The target language for translation (default: Spanish).
            
        Returns:
            Dictionary containing the original text and the translated text.
        """
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.openai_api_key)
            
            completion = client.chat.completions.create(
                model=self.llm_translate_model,
                messages=[
                    {"role": "system", "content": f"""
                        You are a professional translator.
                        Translate the following text to {target_language} while maintaining the original meaning, tone, and style.
                        You must answer only with the translated text, without any other text or comments.
                        If the text is already in {target_language}, just return the text, no need to translate it.
                     """},
                    {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
                ]
            )
            
            translated_text = completion.choices[0].message.content
            return {"original": text, "translated": translated_text, "target_language": target_language}
        except Exception as e:
            print(f"Translation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
