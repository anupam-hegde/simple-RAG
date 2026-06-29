# 🏗️ Architecture

## System Architecture

```mermaid
graph TB
    subgraph "Frontend (Streamlit)"
        UI["🖥️ Streamlit UI<br/>Chat + File Upload"]
    end

    subgraph "Backend (FastAPI)"
        API["🔀 FastAPI Router"]
        AUTH["🔑 API Key Auth"]
        UPLOAD["📤 Upload Handler"]
        CHAT["💬 Chat Handler"]
        STREAM["📡 SSE Stream Handler"]
        HISTORY["📜 History Service"]
    end

    subgraph "RAG Pipeline"
        EXTRACT["📄 Text Extraction<br/>(pypdf / raw read)"]
        CHUNK["✂️ Chunking<br/>(RecursiveCharacterTextSplitter)"]
        EMBED["🧮 Embedding<br/>(all-MiniLM-L6-v2)"]
        RETRIEVE["🔍 Retrieval<br/>(Cosine Similarity)"]
        GENERATE["🤖 Generation<br/>(Groq - Llama 3.3)"]
    end

    subgraph "Storage"
        CHROMA["🗄️ ChromaDB<br/>(Vector Store)"]
        JSON["💾 JSON File<br/>(Chat History)"]
    end

    subgraph "External"
        GROQ["☁️ Groq API"]
        HF["☁️ HuggingFace Local"]
    end

    UI --> |"HTTP/SSE"| API
    API --> AUTH
    AUTH --> UPLOAD
    AUTH --> CHAT
    AUTH --> STREAM
    AUTH --> HISTORY

    UPLOAD --> EXTRACT
    EXTRACT --> CHUNK
    CHUNK --> EMBED
    EMBED --> |"Store vectors"| CHROMA

    CHAT --> RETRIEVE
    STREAM --> RETRIEVE
    RETRIEVE --> |"Query vectors"| CHROMA
    RETRIEVE --> GENERATE
    GENERATE --> |"Answer"| API

    EMBED --> HF
    GENERATE --> GROQ
    HISTORY --> JSON
```

## RAG Pipeline Flow

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Streamlit Frontend
    participant BE as FastAPI Backend
    participant RAG as RAG Service
    participant DB as ChromaDB
    participant AI as Groq API

    Note over U, AI: 📤 Document Ingestion Flow (Progressive SSE)
    U->>FE: Upload PDF/TXT/MD
    FE->>BE: POST /api/upload/progress
    BE->>BE: Validate file type
    BE->>RAG: ingest_document_with_progress()
    RAG->>RAG: Extract text (pypdf / raw read)
    RAG-->>FE: SSE: extracting progress
    RAG->>RAG: Chunk text (1000 chars, 200 overlap)
    RAG-->>FE: SSE: chunking progress
    loop Each Chunk
        RAG->>RAG: Generate local embedding
        RAG->>DB: Store chunk + metadata
        RAG-->>FE: SSE: embedding progress
    end
    BE-->>FE: SSE: done


    Note over U, AI: 💬 Question Answering Flow (Streaming)
    U->>FE: Ask question
    FE->>BE: POST /api/chat/stream
    BE->>RAG: answer_query_stream()
    RAG->>RAG: Embed query locally
    RAG->>DB: Similarity search (top-k=4)
    DB-->>RAG: Relevant chunks + metadata
    RAG->>AI: Generate answer (stream=True)
    loop Token by token
        AI-->>RAG: Token
        RAG-->>BE: SSE data event
        BE-->>FE: SSE token
        FE->>FE: Render token
    end
    RAG-->>BE: SSE sources event
    BE-->>FE: Source references
    FE->>U: Display answer + sources
```

## Directory Structure

```
RAG/
├── docker-compose.yml            # Container orchestration
├── ARCHITECTURE.md               # This file
├── README.md                     # Project documentation
├── .gitignore
│
├── backend/
│   ├── Dockerfile
│   ├── .env                      # GROQ_API_KEY, API_KEY
│   ├── requirements.txt
│   └── app/
│       ├── __init__.py
│       ├── main.py               # FastAPI app + all endpoints
│       ├── core/
│       │   ├── config.py         # Pydantic settings
│       │   └── auth.py           # API key dependency
│       ├── services/
│       │   ├── rag_service.py    # RAG pipeline (ingest + query + stream)
│       │   └── chat_history_service.py  # Session persistence
│       ├── api/
│       │   └── routes/           # (extensible)
│       └── models/               # (extensible)
│
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py                    # Streamlit UI
│
└── data/                         # Git-ignored runtime data
    ├── chroma_db/                # Vector store persistence
    ├── uploads/                  # Temporary file uploads
    └── chat_history.json         # Session history
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | Streamlit | Interactive chat UI |
| Backend | FastAPI + Uvicorn | Async REST API |
| LLM | Groq (Llama 3.3 70B) | Answer generation |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace) | Semantic vector encoding (Local) |
| Vector DB | ChromaDB (PersistentClient) | Similarity search |
| Text Extraction | pypdf | PDF parsing |
| Chunking | LangChain RecursiveCharacterTextSplitter | Document segmentation |
| Auth | API Key (X-API-Key header) | Endpoint protection |
| Containerization | Docker + Docker Compose | Deployment |
