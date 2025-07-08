# config.py

import os

# --- General Application Settings ---uvicorn main:app --reload --host 127.0.0. --port 8000
APP_TITLE = "Zimli Tech Chatbot"
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "t")

# --- API Key Configuration ---
# It should be set as an environment variable in your deployment environment.
# Example: $env:GOOGLE_API_KEY="YOUR_ACTUAL_API_KEY"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Document Loading Settings ---
DOCS_FOLDER = "docs"  # Folder containing your .docx knowledge base files

# --- Text Splitting Settings ---
CHUNK_SIZE = 600
CHUNK_OVERLAP = 70

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- FAISS Vector Store Settings ---
FAISS_INDEX_PATH = "faiss_index_data" # Directory to save/load the FAISS index

# --- Gemini LLM Settings ---
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite" #or "gemini-2.0-flash"
GEMINI_TEMPERATURE = 0.5

# --- Logging Settings ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper() # INFO, DEBUG, WARNING, ERROR, CRITICAL
LOG_FILE = "app.log"
