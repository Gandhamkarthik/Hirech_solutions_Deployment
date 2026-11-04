# Hirec Solutions RAG AI Assistant

An enterprise-grade RAG (Retrieval-Augmented Generation) chatbot that provides answers based on internal company documents.

## 🚀 Architecture
- **Backend**: FastAPI + Groq LLM (qwen/qwen3.8-27b) + ChromaDB.
- **Frontend**: Streamlit Chat Interface.
- **Logic**: Tool-calling agent loop allowing the AI to decide when to retrieve internal data.

## 🛠️ Setup

### 1. Environment Configuration
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
MODEL_NAME=qwen/qwen3.8-27b
```

### 2. Installation & Running (Local)

**Backend:**
```bash
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
streamlit run streamlit_app.py
```

### 3. Running with Docker
```bash
docker build -t rag-app .
docker run -p 8000:8000 -p 8501:8501 rag-app
```

## 📂 Project Structure
- `data/`: Source documents (.txt).
- `main.py`: Integrated Backend, Vector Store, and RAG Agent.
- `streamlit_app.py`: Frontend user interface.
- `requirements.txt`: Project dependencies.

## 🔍 Key Features
- **Custom Chunking**: Preserves semantic sections using `====` separators.
- **Agentic RAG**: Uses tool-calling to retrieve information only when necessary.
- **Retrieval Logs**: Toggleable logs in the UI to see exactly what the AI searched for.

## 🚀 Deployment Instructions

### Create a new repository on the command line
```bash
echo "# Hirech_solutions_completedeplyment" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Gandhamkarthik/Hirech_solutions_completedeplyment.git
git push -u origin main
```

### Push an existing repository from the command line
```bash
git remote add origin https://github.com/Gandhamkarthik/Hirech_solutions_completedeplyment.git
git branch -M main
git push -u origin main
```
