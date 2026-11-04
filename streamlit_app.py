import streamlit as st
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

# 1. Configuration & Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file. Please add it to proceed.")
    st.stop()

client_groq = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# Initialize ChromaDB
# Use PersistentClient to ensure data stays on disk between app restarts
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Use get_or_create to avoid "Collection already exists" error
collection = chroma_client.get_or_create_collection(name="company_docs")

# 2. Helper Functions (from Notebook)
def chunk_document(text, source_name):
    raw_sections = text.split("============================")
    chunks = []
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue
        chunks.append({"text": section, "source": source_name})
    return chunks

def ingest_docs():
    data_files = {
        "data/company_hr_policy.txt": "HR Policy",
        "data/engineering_standards.txt": "Engineering",
        "data/onboarding_guide.txt": "Onboarding",
        "data/product_knowledge_base.txt": "Product",
        "data/security_policy.txt": "Security",
    }

    all_chunks = []
    for path, source in data_files.items():
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                all_chunks.extend(chunk_document(text, source))
        except FileNotFoundError:
            st.warning(f"File not found: {path}")

    if not all_chunks:
        return 0

    documents = [c['text'] for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"source": c["source"]} for c in all_chunks]

    # Clear existing to avoid duplicates if re-ingesting
    try:
        # ChromaDB doesn't have a simple 'clear', we recreate or update
        # For simplicity in a demo app, we just add.
        # In production, you'd manage IDs carefully.
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
    except Exception as e:
        # If IDs already exist, we might need to handle it
        st.error(f"Ingestion error: {e}")
        return 0

    return len(all_chunks)

def search_docs(query: str) -> str:
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    chunks = results['documents'][0]
    return "\n\n".join(chunks)

# Tool definition for the Agent
rag_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the company internal documents (HR Policy, Product, Security, onboarding, engineering) to find the relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            }
        }
    }
]

available_tools = {"search_docs": search_docs}

def rag_agent(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful company assistant with access to internal documents "
                "(HR, Product, Engineering, Onboarding, Security). "
                "Use the search_docs tool to find answers. If the documents don't contain "
                "the answer, say so clearly. Base your answers strictly on retrieved documents."
            )
        },
        {"role": "user", "content": question}
    ]

    logs = []
    max_steps = 3
    for _ in range(max_steps):
        response = client_groq.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=messages,
            tools=rag_tools,
        )
        choice = response.choices[0]

        if choice.finish_reason == "stop":
            return choice.message.content, logs

        if choice.message.tool_calls:
            messages.append(choice.message)
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                logs.append(f"🔍 Searching for: {args['query']}...")
                result = available_tools[func_name](**args)
                logs.append(f"📄 Found {len(result)} characters of context.")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

    return "Could not answer within step limit.", logs

# 3. Streamlit UI
st.set_page_config(page_title="Hirec Solutions AI", layout="wide", page_icon="🏢")

# Custom CSS for a better look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏢 Hirec Solutions AI Assistant")
st.markdown("Welcome to the internal company knowledge base. Ask me anything about HR, Engineering, or Security policies!")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ System Control")

    if st.button("🔄 Sync Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents..."):
            count = ingest_docs()
            if count > 0:
                st.success(f"✅ Successfully indexed {count} chunks!")
            else:
                st.error("❌ Failed to index documents. Check if 'data/' folder exists.")

    st.divider()
    show_logs = st.checkbox("🔍 Show Agent Thought Process", value=True)

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "logs" in message and show_logs and message["role"] == "assistant":
            with st.expander("Agent Logs"):
                for log in message["logs"]:
                    st.write(log)

# Chat input
if prompt := st.chat_input("Ask about company policies..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching internal docs..."):
            answer, logs = rag_agent(prompt)
            st.markdown(answer)
            if show_logs:
                with st.expander("Agent Logs"):
                    for log in logs:
                        st.write(log)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "logs": logs
            })
