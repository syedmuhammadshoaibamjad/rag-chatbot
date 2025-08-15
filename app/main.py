# rag/app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# from dotenv import load_dotenv


import os
import sys

os.environ["GOOGLE_API_KEY"] = "REDACTED"
os.environ["PINECONE_API_KEY"] = "REDACTED"
os.environ["PINECONE_ENVIRONMENT"] = "us-east-1"
os.environ["PINECONE_INDEX_NAME"] = "ragai"


# load_dotenv(dotenv_path=os.path.join(project_root, ".env"))



current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from routes.main import router

app = FastAPI(
    title="RAG with LangChain and FastAPI",
    description="A backend application for Retrieval-Augmented Generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
