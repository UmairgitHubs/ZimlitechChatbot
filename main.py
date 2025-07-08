# main.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#langchain_google_genai, langchain_community,langchain, langchain_huggingface
# Import configurations from our custom config file
import config

# --- 0. Set up Logging ---
# Configure basic logging to console and a file
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output logs to console
        logging.FileHandler(config.LOG_FILE) # Output logs to a file
    ]
)
logger = logging.getLogger(__name__)

# --- Ensure API Key is Set ---
if not config.GOOGLE_API_KEY:
    logger.critical("GOOGLE_API_KEY environment variable is not set. Please set it before running the application.")
    # In a real production app, you might want to exit here or disable LLM features
    # For now, we'll let it proceed but expect LLM calls to fail.
    # raise ValueError("GOOGLE_API_KEY environment variable not set.")
    # For demonstration, we'll allow it to proceed with a warning, but this is BAD for prod.
    pass # Allowing for local testing if not strictly enforced by environment

# --- 1. Document Loading Function ---
def load_documents_from_folder(folder_path: str) -> list:
    """
    Loads all .docx documents from the specified folder.

    Args:
        folder_path: The path to the folder containing DOCX files.

    Returns:
        A list of loaded LangChain Document objects.
    """
    documents = []
    path = Path(folder_path)
    if not path.is_dir():
        logger.warning(f"Document folder '{folder_path}' not found. No documents will be loaded.")
        return []

    for file_path in path.glob("*.docx"):
        try:
            loader = Docx2txtLoader(str(file_path))
            documents.extend(loader.load())
            logger.info(f"Successfully loaded document: {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading document {file_path.name}: {e}", exc_info=True)
    return documents

# --- 2. Initialize Embeddings and Vector Store ---
# Initialize the embedding model outside the main logic to avoid re-creation
try:
    embedding_model = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    logger.info(f"Successfully initialized embedding model: {config.EMBEDDING_MODEL_NAME}")
except Exception as e:
    logger.critical(f"Failed to initialize embedding model: {e}", exc_info=True)
    # Depending on criticality, you might want to exit here
    exit(1)

vectordb = None
# Attempt to load FAISS index from disk
faiss_index_path_obj = Path(config.FAISS_INDEX_PATH)
if faiss_index_path_obj.exists() and faiss_index_path_obj.is_dir():
    try:
        # allow_dangerous_deserialization=True is needed for loading FAISS indexes
        # that contain custom objects (like LangChain documents or embeddings)
        vectordb = FAISS.load_local(str(faiss_index_path_obj), embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Successfully loaded FAISS index from: {config.FAISS_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Failed to load FAISS index from {config.FAISS_INDEX_PATH}: {e}. Rebuilding index.", exc_info=True)
        vectordb = None # Force rebuild if loading fails
else:
    logger.info(f"FAISS index directory '{config.FAISS_INDEX_PATH}' not found. Will build new index.")

# If FAISS index was not loaded or failed, build it
if vectordb is None:
    documents = load_documents_from_folder(config.DOCS_FOLDER)
    if not documents:
        logger.warning("No .docx files found or loaded successfully. The chatbot will not have a knowledge base.")
        # If no documents, the RAG chain will have no context.
        # Consider raising an error or providing a fallback behavior.
    else:
        try:
            text_splitter = CharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(documents)
            vectordb = FAISS.from_documents(docs, embedding_model)
            # Save the newly built index for future use
            vectordb.save_local(str(faiss_index_path_obj))
            logger.info(f"Successfully built and saved new FAISS index to: {config.FAISS_INDEX_PATH}")
        except Exception as e:
            logger.critical(f"Failed to build and save FAISS index: {e}", exc_info=True)
            # This is a critical error as the RAG system won't function without it.
            vectordb = None # Ensure vectordb is None if build fails


# --- 3. Custom Prompt ---
PROMPT_TEMPLATE_STR = """
You are an AI assistant for ZimliTech named Zimlibot, specializing in providing helpful and accurate information to customers about our range of tech services.

**Your Goal:** To answer customer questions comprehensively, clearly, and concisely, drawing primarily from the provided knowledge base. If the knowledge base does not contain the answer, you should politely state that you cannot find the information and suggest alternative actions.

**Customer Persona:** Customers are typically seeking information about our services, technical support, pricing, onboarding, or troubleshooting. They may range from tech-savvy individuals to those with limited technical understanding.

**Key Instructions:**

1.  **Retrieval First:** Prioritize retrieving relevant information from the provided knowledge base snippets.
2.  **Synthesize and Summarize:** Combine information from multiple retrieved snippets if necessary to form a complete answer. Summarize complex technical details into easily understandable language.
3.  **Direct and Clear:** Provide direct answers to the customer's question. Avoid jargon where possible, or explain it clearly if unavoidable.
5.  **Polite and Professional:** Maintain a helpful, respectful, and professional tone at all times.
6.  **Out-of-Scope Handling:** If a question cannot be answered from the provided knowledge base, state clearly: "I apologize, but I couldn't find specific information regarding that in my current knowledge base. Would you like me to connect you with a support agent, or can I assist you with something else?"
7.  **No Hallucination:** Absolutely do not invent information. Stick strictly to the retrieved content.
8.  **Formatting:** Use bullet points or numbered lists for multi-step instructions or lists of features to enhance readability.
9. The compliments like 'great', 'best', 'excellent', etc. should be answer polietly no need to check for knowledge base. 
**Knowledge Base Snippets (will be dynamically inserted by the RAG system):**
{context}

**Customer Query:**
{question}

**Your Response:**
"""
PROMPT = PromptTemplate(template=PROMPT_TEMPLATE_STR, input_variables=["context", "question"])

# --- 4. Initialize Gemini and RetrievalQA Chain ---
qa_chain = None
if vectordb: # Only initialize QA chain if vector database is available
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL_NAME,
            temperature=config.GEMINI_TEMPERATURE,
            google_api_key=config.GOOGLE_API_KEY # Explicitly pass the key if available
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=False, # We just want the answer
            chain_type="stuff", # "stuff" all retrieved documents into one prompt
            chain_type_kwargs={"prompt": PROMPT}
        )
        logger.info("Successfully initialized RetrievalQA chain.")
    except Exception as e:
        logger.critical(f"Failed to initialize RetrievalQA chain: {e}", exc_info=True)
else:
    logger.warning("Vector database not initialized. RetrievalQA chain will not be available.")

# --- 5. FastAPI Setup ---
app = FastAPI(title=config.APP_TITLE, debug=config.DEBUG_MODE)

# Pydantic model for request body validation
class QueryRequest(BaseModel):
    question: str

@app.post("/ask", response_model=Dict[str, str])
async def ask_question(req: QueryRequest):
    """
    API endpoint to ask a question to the Zimli Tech Chatbot.
    """
    query = req.question.strip()
    logger.info(f"Received query: '{query}'")

    if not query:
        logger.warning("Empty query received.")
        raise HTTPException(status_code=400, detail="Please enter a valid question.")

    if not qa_chain:
        logger.error("QA chain is not initialized. Cannot process query.")
        raise HTTPException(status_code=503, detail="Chatbot service is not fully operational. Please try again later.")

    try:
        # LangChain's invoke method is designed to be async-compatible
        result = await qa_chain.ainvoke({"query": query}) # Use ainvoke for async calls
        response_text = result.get("result", "An answer could not be generated at this time.")
        logger.info(f"Query processed. Response: '{response_text[:100]}...'") # Log first 100 chars
        return {"response": response_text}
    except Exception as e:
        logger.error(f"Error during QA chain invocation for query '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

@app.get("/")
async def serve_webpage():
    """
    Serves the main HTML page for the chatbot.
    """
    html_file_path = Path("index.html")
    if not html_file_path.exists():
        logger.error(f"index.html not found at {html_file_path.absolute()}")
        raise HTTPException(status_code=404, detail="Webpage not found.")
    logger.info("Serving index.html")
    return FileResponse(html_file_path)

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    status = "OK"
    message = "Service is healthy."

    if not config.GOOGLE_API_KEY:
        status = "WARNING"
        message = "API key not set, LLM functionality might be limited."
    if not vectordb:
        status = "CRITICAL"
        message = "Vector database not initialized, RAG functionality is impaired."
    if not qa_chain:
        status = "CRITICAL"
        message = "QA chain not initialized, chatbot is not operational."

    logger.debug(f"Health check status: {status}, Message: {message}")
    return {"status": status, "message": message}

# --- How to Run ---
# To run this application:
# 1. Make sure you have the required packages installed: (run in terminal, make sure you make virtual enviornment)
#    pip install fastapi uvicorn python-multipart langchain-google-genai langchain-community docx2txt langchain-huggingface

# 2. Set your Google API Key as an environment variable:
#    (On Windows: $env:GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY_HERE" in PowerShell or set GOOGLE_API_KEY=... in CMD)

# 3. Create a 'docs' folder in the same directory as main.py and place your .docx files inside.

# 4. Run the application using Uvicorn:
#    uvicorn main2:app --host 127.0.0.1 --port 8000

