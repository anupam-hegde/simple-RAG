"""
FastAPI application — RAG Backend.

Endpoints:
  POST /api/upload              Upload a document (PDF / TXT / MD)
  POST /api/chat                Ask a question (JSON response)
  POST /api/chat/stream         Ask a question (SSE streaming response)
  GET  /api/chat/history        List all chat sessions
  POST /api/chat/history        Create a new chat session
  GET  /api/chat/history/{id}   Get a full chat session
  DELETE /api/chat/history/{id} Delete a chat session
  GET  /api/health              Health check
"""

import asyncio
import logging
import traceback
from dataclasses import asdict
from io import BytesIO

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.app.core.auth import require_api_key
from backend.app.core.config import get_settings
from backend.app.services.chat_history_service import get_chat_history_service
from backend.app.services.rag_service import (
    DocumentIngestionError,
    QueryResponse as RAGQueryResponse,
    RAGService,
)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Document Assistant API",
    version="1.0.0",
    description=(
        "AI-powered Document Assistant capable of answering questions "
        "from uploaded documents using Retrieval-Augmented Generation (RAG)."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Services ─────────────────────────────────────────────────────────────────

rag_service = RAGService()

# ── Pydantic Schemas ─────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question text")
    session_id: str | None = Field(None, description="Optional chat session ID for history tracking")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or only whitespace.")
        return v


class SourceReference(BaseModel):
    text: str
    filename: str
    page: int


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceReference]
    session_id: str | None = None


class UploadResponse(BaseModel):
    status: str
    filename: str
    message: str


class SessionCreate(BaseModel):
    title: str = "New Chat"


class ErrorResponse(BaseModel):
    detail: str



# ── Upload Endpoint ──────────────────────────────────────────────────────────


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    tags=["Documents"],
    summary="Upload a document for ingestion",
)
async def upload_document(
    file: UploadFile = File(...),
    _api_key: str = Depends(require_api_key),
):
    """
    Uploads a document file (PDF, TXT, MD) and processes it synchronously.
    Triggers text extraction → chunking → embedding → ChromaDB storage.
    Returns only after the document is fully ingested.
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    if not file.filename or not any(
        file.filename.lower().endswith(ext) for ext in allowed_extensions
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only {', '.join(allowed_extensions)} files are allowed.",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded.",
            )

        file_stream = BytesIO(contents)
        num_chunks = await rag_service.ingest_document_stream(file.filename, file_stream)
        logger.info(f"Document '{file.filename}' ingested successfully: {num_chunks} chunks")

        return UploadResponse(
            status="success",
            filename=file.filename,
            message=f"File '{file.filename}' processed successfully. {num_chunks} chunks indexed.",
        )

    except DocumentIngestionError as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document ingestion failed: {str(e)}",
        ) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e


# ── Upload Progress (SSE) Endpoint ───────────────────────────────────────────


@app.post(
    "/api/upload/progress",
    tags=["Documents"],
    summary="Upload a document and stream ingestion progress (SSE)",
)
async def upload_document_progress(
    file: UploadFile = File(...),
    _api_key: str = Depends(require_api_key),
):
    """
    Uploads a document and streams real-time ingestion progress as SSE events.
    Each event is a JSON object with keys: stage, message, pct, (current, total).
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    if not file.filename or not any(
        file.filename.lower().endswith(ext) for ext in allowed_extensions
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only {', '.join(allowed_extensions)} files are allowed.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded.",
        )

    file_stream = BytesIO(contents)

    return StreamingResponse(
        rag_service.ingest_document_with_progress(file.filename, file_stream),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chat Endpoint ────────────────────────────────────────────────────────────


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Ask a question about uploaded documents",
)
async def chat(
    request: ChatRequest,
    _api_key: str = Depends(require_api_key),
):
    """
    Accepts a question and returns a generated answer with source references.
    Optionally persists messages to a chat session when session_id is provided.
    """
    try:
        history_service = get_chat_history_service()

        # Auto-create session if session_id provided but doesn't exist
        session_id = request.session_id
        if session_id and not history_service.get_session(session_id):
            history_service.create_session()

        # Persist the user message
        if session_id:
            history_service.add_message(session_id, "user", request.question)

        query_response: RAGQueryResponse = await rag_service.answer_query(
            request.question
        )

        sources = [
            SourceReference(text=ctx.text, filename=ctx.filename, page=ctx.page)
            for ctx in query_response.sources
        ]

        # Persist the assistant message
        if session_id:
            history_service.add_message(
                session_id,
                "assistant",
                query_response.answer,
                [s.model_dump() for s in sources],
            )

        return ChatResponse(
            answer=query_response.answer,
            sources=sources,
            session_id=session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat query failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat query.",
        ) from e


# ── Chat Streaming Endpoint ─────────────────────────────────────────────────


@app.post(
    "/api/chat/stream",
    tags=["Chat"],
    summary="Ask a question with streaming response (SSE)",
)
async def chat_stream(
    request: ChatRequest,
    _api_key: str = Depends(require_api_key),
):
    """
    Accepts a question and streams the generated answer token-by-token
    using Server-Sent Events (SSE). Source references are sent as a
    separate 'sources' event at the end of the stream.
    """
    return StreamingResponse(
        rag_service.answer_query_stream(request.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Chat History Endpoints ───────────────────────────────────────────────────


@app.get(
    "/api/chat/history",
    tags=["Chat History"],
    summary="List all chat sessions",
)
async def list_sessions(_api_key: str = Depends(require_api_key)):
    """Returns lightweight summaries of all chat sessions, newest first."""
    return get_chat_history_service().list_sessions()


@app.post(
    "/api/chat/history",
    status_code=status.HTTP_201_CREATED,
    tags=["Chat History"],
    summary="Create a new chat session",
)
async def create_session(
    body: SessionCreate,
    _api_key: str = Depends(require_api_key),
):
    """Creates a new empty chat session and returns its metadata."""
    session = get_chat_history_service().create_session(body.title)
    return {"id": session.id, "title": session.title, "created_at": session.created_at}


@app.get(
    "/api/chat/history/{session_id}",
    tags=["Chat History"],
    summary="Get full chat session",
)
async def get_session(
    session_id: str,
    _api_key: str = Depends(require_api_key),
):
    """Returns the full conversation for a given session ID."""
    session = get_chat_history_service().get_session(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return asdict(session)


@app.delete(
    "/api/chat/history/{session_id}",
    status_code=status.HTTP_200_OK,
    tags=["Chat History"],
    summary="Delete a chat session",
)
async def delete_session(
    session_id: str,
    _api_key: str = Depends(require_api_key),
):
    """Deletes a chat session and its messages."""
    deleted = get_chat_history_service().delete_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return {"status": "deleted", "session_id": session_id}


# ── Health Check ─────────────────────────────────────────────────────────────


@app.get("/api/health", status_code=status.HTTP_200_OK, tags=["System"])
async def health_check():
    """Simple health check to verify the API is running."""
    return {"status": "healthy", "service": "RAG-Document-Assistant"}


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
