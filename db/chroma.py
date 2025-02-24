import chromadb
from chromadb.config import Settings

class ChromaDBClient:
    def __init__(self, host=None, port=None, auth_credentials=None, auth_provider=None, auth_token_transport_header=None):
        self.client = None
        self.host = host
        self.port = port 
        self.auth_credentials = auth_credentials
        self.auth_provider = auth_provider
        self.auth_token_transport_header = auth_token_transport_header
        
        # Setup client
        self.get_or_create_client()
    def get_or_create_collection(self, collection_name):
        return self.client.get_or_create_collection(collection_name)
    
    def list_collections(self):
        collection_names = self.client.list_collections()
        return collection_names
    
    def get_or_create_client(self):
        if self.client is None:
            self.client = chromadb.HttpClient(
                host=self.host,
                port=self.port,
                settings=Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    # chroma_client_auth_provider=self.auth_provider,
                    # chroma_client_auth_credentials=self.auth_credentials,
                    # chroma_auth_token_transport_header=self.auth_token_transport_header,
                )
            )
        
        return self.client
    
    def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name)
