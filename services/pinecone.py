import os
from pinecone import Pinecone, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import Optional

class PineconeService:
    """
    Handles all interactions with the Pinecone vector database using LangChain.
    """
    _vector_store: Optional[PineconeVectorStore] = None

    def __init__(self, api_key: str, environment: str, index_name: str):
        """Initializes the Pinecone client and connects to the index."""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name

        self.pc = Pinecone(api_key=self.api_key)
    
    def create_or_get_index(self):
        """
        Creates the Pinecone index if it does not already exist.
        """
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index: {self.index_name}...")

            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric='cosine',
                spec=PodSpec(environment=self.environment)
            )
            print("Index created.")
        else:
            print(f"Index '{self.index_name}' already exists. Skipping creation.")

    def get_vector_store(self, embeddings: GoogleGenerativeAIEmbeddings) -> PineconeVectorStore:
        """
        Returns a singleton instance of the LangChain PineconeVectorStore.
        
        Args:
            embeddings (GoogleGenerativeAIEmbeddings): The LangChain embeddings model.
        """
        self.create_or_get_index()

        if self._vector_store is None:
            self._vector_store = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=embeddings
            )
        return self._vector_store

