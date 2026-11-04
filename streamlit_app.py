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
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_docs")

# 2. Helper Functions
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

    try:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)
    except Exception as e:
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
st.set_page_config(page_title="Hirec Solutions AI", layout="centered", page_icon="🏢")

# --- DARK MODE / HIGH CONTRAST THEME ---
st.markdown("""
    <style>
    /* Main App Background - Deep Grey/Dark */
    .stApp {
        background-color: #121212 !important;
        color: #FFFFFF !important;
    }

    /* Force all text to be white/light grey for maximum contrast */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div {
        color: #E0E0E0 !important;
    }

    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: #1E1E1E !important;
        border: 1px solid #333333 !important;
        border-radius: 15px !important;
        color: #FFFFFF !important;
    }

    /* Distinguish User and Assistant bubbles */
    [data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #2A2A2A !important;
        border-left: 5px solid #4A90E2 !important;
    }

    [data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1E1E1E !important;
        border-left: 5px solid #50C878 !important;
    }

    /* Header Styling */
    .main-title {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        text-align: center !important;
        margin-bottom: 10px !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    .sub-title {
        font-size: 18px !important;
        color: #B0B0B0 !important;
        text-align: center !important;
        margin-bottom: 30px !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #181818 !important;
        border-right: 1px solid #333333 !important;
    }

    /* Sidebar Text */
    [data-testid="stSidebar"] .stMarkdown p {
        color: #CCCCCC !important;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #333333 !important;
        color: white !important;
        border: 1px solid #444444 !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        background-color: #4A90E2 !important;
        border-color: #4A90E2 !important;
        color: white !important;
    }

    /* Input box styling */
    .stChatInputContainer {
        background-color: #121212 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Page Header
st.markdown('<p class="main-title">🏢 Hirec Solutions AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Enterprise Knowledge Assistant • Dark Edition</p>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("### ⚙️ System Control")

    if st.button("🔄 Sync Knowledge Base", use_container_width=True):
        with st.spinner("Indexing documents..."):
            count = ingest_docs()
            if count > 0:
                st.success(f"✅ Indexed {count} chunks!")
            else:
                st.error("❌ Indexing failed.")

    st.divider()
    show_logs = st.checkbox("🔍 Show Agent Thoughts", value=False)

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("💡 **Tip**: If you don't see answers, click 'Sync Knowledge Base' first.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "logs" in message and show_logs and message["role"] == "assistant":
            with st.expander("View Step-by-Step Search"):
                for log in message["logs"]:
                    st.write(log)

# Chat input
if prompt := st.chat_input("Ask about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching internal docs..."):
            answer, logs = rag_agent(prompt)
            st.markdown(answer)
            if show_logs:
                with st.expander("View Step-by-Step Search"):
                    for log in logs:
                        st.write(log)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "logs": logs
            })
