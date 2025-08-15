# rag/services/gemini.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import Optional

class GeminiService:
    """
    Handles all interactions with the Google Generative AI API, providing
    LangChain-compatible models for both text generation (LLM) and embeddings.
    """
    
    _llm_model: Optional[ChatGoogleGenerativeAI] = None
    _embeddings_model: Optional[GoogleGenerativeAIEmbeddings] = None

    def __init__(self, api_key: str):
        """Initializes the service with the Google API key."""
        self.api_key = api_key
    
    def get_llm_model(self) -> ChatGoogleGenerativeAI:
        """
        Returns a singleton instance of the LangChain ChatGoogleGenerativeAI model.
        """
        if self._llm_model is None:
            self._llm_model = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.api_key,
                temperature=0.2,
            )
        return self._llm_model

    def get_embeddings_model(self, output_dimensionality: int) -> GoogleGenerativeAIEmbeddings:
        """
        Returns a singleton instance of the LangChain GoogleGenerativeAIEmbeddings model
        with a specified output dimension.
        
        Args:
            output_dimensionality (int): The desired dimension of the output embeddings.
        """
        if self._embeddings_model is None:
            self._embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=self.api_key,
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=output_dimensionality
            )
        return self._embeddings_model
