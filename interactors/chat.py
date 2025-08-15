# rag/interactors/chat.py

from services.gemini import GeminiService
from services.pinecone import PineconeService
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import HTTPException
from typing import Dict, Any

async def get_rag_response(
    query: str,
    pinecone_service: PineconeService,
    gemini_service: GeminiService,
) -> Dict[str, Any]:

    try:
        llm = gemini_service.get_llm_model()
        embeddings = gemini_service.get_embeddings_model(output_dimensionality=768)
        
        retriever = pinecone_service.get_vector_store(
            embeddings=embeddings
        ).as_retriever(search_kwargs={"k": 6})
        
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        If the context does not contain the answer, say "I couldn't find an answer to that question in the provided documents."
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """)

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = await retrieval_chain.ainvoke({"input": query})

        return {
            "query": query,
            "response": response["answer"],
            "source_documents": [doc.page_content for doc in response["context"]],
        }

    except Exception as e:
        print(f"An error occurred in chat.py: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
