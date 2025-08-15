# rag/interactors/upload_data.py

from fastapi import HTTPException, UploadFile
from services.gemini import GeminiService
from services.pinecone import PineconeService
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import shutil
from tempfile import NamedTemporaryFile

async def upload_pdf_to_pinecone(
    file: UploadFile,
    pinecone_service: PineconeService,
    gemini_service: GeminiService,
):
    """
    Handles the entire process of uploading a PDF, splitting it into chunks,
    creating embeddings, and storing them in a Pinecone vector index using LangChain.
    """
    if not file.content_type == "application/pdf":
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a PDF."
        )

    print(f"✅ Starting upload process for file: {file.filename}")

    try:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        print("✅ File saved to temporary path. Loading PDF...")
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        os.remove(tmp_path)
        print(f"✅ PDF loaded and temporary file removed. Found {len(docs)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        documents = text_splitter.split_documents(docs)
        print(f"✅ Documents split into {len(documents)} chunks.")

        embeddings = gemini_service.get_embeddings_model(output_dimensionality=768)
        vector_store = pinecone_service.get_vector_store(embeddings)

        print("🔄 Adding documents to Pinecone vector store...")
        await vector_store.aadd_documents(documents)
        print(f"✅ Successfully added {len(documents)} documents to Pinecone!")

        return {
            "message": "PDF uploaded, processed, and stored successfully!",
            "documents_uploaded": len(documents)
        }
    except Exception as e:
        print(f"An error occurred during document processing: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
