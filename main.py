import os
import json
from typing import List, Dict, Tuple
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen3.8-27b")
DATA_DIR = "data"
CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "company_docs"

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

# --- VECTOR STORE LOGIC ---
class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks: List[Dict]):
        if not chunks: return 0
        documents = [c['text'] for c in chunks]
        metadatas = [{"source": c['source']} for c in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=documents, ids=ids, metadatas=metadatas)
        return len(chunks)

    def query(self, query_text: str, n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        return results['documents'][0], results['metadatas'][0]

vector_store = VectorStore()

# --- DOCUMENT PROCESSOR ---
def process_documents():
    all_chunks = []
    if not os.path.exists(DATA_DIR):
        return all_chunks

    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.txt')]
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        source_name = filename.replace('.txt', '').replace('_', ' ').title()
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            raw_sections = text.split("============================")
            for section in raw_sections:
                section = section.strip()
                if section:
                    all_chunks.append({"text": section, "source": source_name})
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return all_chunks

# --- RAG AGENT LOGIC ---
class RAGAgent:
    def __init__(self):
        self.client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
        self.model = MODEL_NAME
        self.tools = [{
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search company internal documents for policies, products, or support.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"]
                }
            }
        }]

    def search_docs(self, query: str) -> str:
        chunks, _ = vector_store.query(query)
        return "\n\n".join(chunks)

    def chat(self, user_query: str, max_steps: int = 3):
        messages = [
            {"role": "system", "content": "You are a helpful company assistance. Use search_docs tool for internal info. If not found, say so."},
            {"role": "user", "content": user_query}
        ]
        steps_taken = []
        for step in range(max_steps):
            response = self.client.chat.completions.create(model=self.model, messages=messages, tools=self.tools)
            choice = response.choices[0]
            if choice.finish_reason == "stop":
                return {"answer": choice.message.content, "steps": steps_taken, "status": "completed"}
            if choice.message.tool_calls:
                messages.append(choice.message)
                for tool_call in choice.message.tool_calls:
                    args = json.loads(tool_call.function.arguments)
                    query = args.get("query")
                    result = self.search_docs(query)
                    steps_taken.append({"tool": "search_docs", "query": query, "result_preview": result[:200]})
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
        return {"answer": "Limit reached", "steps": steps_taken, "status": "limit_reached"}

rag_agent = RAGAgent()

# --- FASTAPI APP ---
app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    return rag_agent.chat(request.message)

@app.post("/ingest")
async def ingest():
    chunks = process_documents()
    count = vector_store.add_documents(chunks)
    return {"status": "success", "chunks_indexed": count}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
