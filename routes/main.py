# rag/routes/main.py

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from typing import Annotated
from dotenv import load_dotenv
import os
from interactors.upload_data import upload_pdf_to_pinecone
from interactors.chat import get_rag_response
from services.pinecone import PineconeService
from services.gemini import GeminiService

load_dotenv(dotenv_path="../.env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

router = APIRouter()

def get_pinecone_service() -> PineconeService:
    if not all([PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME]):
        raise HTTPException(
            status_code=500, detail="Pinecone credentials are not configured in the .env file."
        )
    return PineconeService(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name=PINECONE_INDEX_NAME
    )

def get_gemini_service() -> GeminiService:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=500, detail="Google API key is not configured in the .env file."
        )
    return GeminiService(api_key=GOOGLE_API_KEY)

@router.post("/upload-pdf")
async def upload_pdf_route(
    file: Annotated[UploadFile, File()],
    pinecone_service: Annotated[PineconeService, Depends(get_pinecone_service)],
    gemini_service: Annotated[GeminiService, Depends(get_gemini_service)]
):
   
    return await upload_pdf_to_pinecone(file, pinecone_service, gemini_service)

@router.post("/chat")
async def chat_route(
    query: dict,
    pinecone_service: Annotated[PineconeService, Depends(get_pinecone_service)],
    gemini_service: Annotated[GeminiService, Depends(get_gemini_service)]
):
    """
    Endpoint to handle a user's chat query and return a RAG-based response.
    """
    return await get_rag_response(query['query'], pinecone_service, gemini_service)
