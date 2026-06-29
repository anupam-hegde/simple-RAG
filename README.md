# 🤖 AI-Powered RAG Document Assistant

An intelligent document assistant that answers questions from uploaded documents using **Retrieval-Augmented Generation (RAG)**. Built with FastAPI, Streamlit, ChromaDB, and OpenAI.

---

## ✨ Features

### Core
- **📤 Document Upload** — Upload PDF, TXT, and Markdown files via API or UI
- **📄 Text Extraction** — Automatic text extraction using `pypdf` (PDF) and raw I/O (TXT/MD)
- **✂️ Smart Chunking** — `RecursiveCharacterTextSplitter` with 1000-char chunks and 200-char overlap
- **🧮 Embedding Generation** — `all-MiniLM-L6-v2` (via Hugging Face) for free, local semantic vector encoding
- **🗄️ ChromaDB Integration** — Persistent vector store with cosine similarity search
- **💬 Q&A with Source References** — Answers cite exact filenames and page numbers
- **📜 Chat History API** — Full CRUD for conversation sessions with JSON persistence

### Bonus
- **📁 Multi-document Support** — Upload and query across multiple documents simultaneously
- **🐳 Dockerization** — Full `docker-compose` setup for one-command deployment
- **🔑 API Key Authentication** — Protect all endpoints with `X-API-Key` header
- **📡 Streaming Responses** — Real-time token-by-token SSE streaming in the chat UI

---

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system diagrams and the RAG pipeline flow.

```
User → Streamlit UI → FastAPI Backend → ChromaDB + OpenAI → Streamed Answer
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- OpenAI API key

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd RAG
```

### 2. Configure Environment

```bash
# backend/.env
OPENAI_API_KEY=sk-your-key-here
API_KEY=your-secret-api-key   # optional — leave empty to disable auth
```

### 3. Run Locally

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

### 4. Run with Docker

```bash
docker-compose up --build
```

| Service  | URL |
|----------|-----|
| Frontend | http://localhost:8501 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| API Docs (ReDoc) | http://localhost:8000/redoc |

---

## 📡 API Reference

### Documents

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a document (PDF/TXT/MD) for ingestion |

### Chat

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Ask a question (JSON response) |
| `POST` | `/api/chat/stream` | Ask a question (SSE streaming response) |

### Chat History

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/chat/history` | List all chat sessions |
| `POST` | `/api/chat/history` | Create a new chat session |
| `GET` | `/api/chat/history/{session_id}` | Get full conversation |
| `DELETE` | `/api/chat/history/{session_id}` | Delete a session |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |

### Authentication

All endpoints (except `/api/health` and `/docs`) require an `X-API-Key` header when `API_KEY` is configured in the backend `.env` file.

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/api/chat/history
```

### Example: Upload a Document

```bash
curl -X POST http://localhost:8000/api/upload \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf"
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'
```

**Response:**
```json
{
  "answer": "The document discusses...",
  "sources": [
    {
      "text": "Relevant chunk text...",
      "filename": "document.pdf",
      "page": 3
    }
  ],
  "session_id": null
}
```

### Example: Streaming

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document"}'
```

---

## 🧪 Testing

```bash
# Health check
curl http://localhost:8000/api/health

# Upload
curl -X POST http://localhost:8000/api/upload -F "file=@test.pdf"

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?"}'

# Chat History
curl http://localhost:8000/api/chat/history
```

---

## 📂 Project Structure

```
RAG/
├── docker-compose.yml
├── ARCHITECTURE.md
├── README.md
├── backend/
│   ├── Dockerfile
│   ├── .env
│   ├── requirements.txt
│   └── app/
│       ├── main.py               # FastAPI app + endpoints
│       ├── core/
│       │   ├── config.py         # Settings
│       │   └── auth.py           # API key auth
│       └── services/
│           ├── rag_service.py    # RAG pipeline
│           └── chat_history_service.py
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py                    # Streamlit UI
```

---

## 🎥 Screen Recording

> 📹 [Screen Recording Link — TODO: Add your recording link here]

---

## 📄 License

MIT
