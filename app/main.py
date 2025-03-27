from fastapi import FastAPI, Query
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from typing import List
import os
import logging
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for debugging)
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# ✅ Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTOR_DB_PATH = "C:/Users/mdjal/OneDrive/Desktop/langchain/vector_db"

# ✅ Initialize the embeddings
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# ✅ Load existing vector database
if os.path.exists(VECTOR_DB_PATH):
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model)
    retriever = vector_db.as_retriever()
    logger.info("Vector database loaded successfully.")
else:
    raise RuntimeError("Vector database not found. Ensure you have persisted it before running FastAPI.")

# ✅ Load the stored Chroma vector database at startup


@app.get("/")
def home():
    return {"message": "Welcome to the AI Agent Retriever API"}

@app.get("/search/")
def search(query: str = Query(..., description="Enter your query text"), k: int = 5):
    """
    Search for the most similar documents in the vector database.
    """
    try:
        print(1)
       
        results = retriever.get_relevant_documents(query, k=5)  # ✅ Using retriever instead of direct vector_db call
        response = [
            {
                "Agent_Name": doc.metadata.get("Agent Name", "N/A"),
                "Agent_UID": doc.metadata.get("Agent UID", "N/A"),
                "Chunk": doc.page_content
            }
            for doc in results
        ]
        return {"query": query, "results": response}
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return {"error": "An internal error occurred. Please try again later."}
