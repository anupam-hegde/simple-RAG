import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI
from pypdf import PdfReader

from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    source: str
    page: int


class DocumentIngestionError(Exception):
    pass


class RetrievalError(Exception):
    pass


@dataclass
class RetrievedContext:
    text: str
    filename: str
    page: int


@dataclass
class QueryResponse:
    answer: str
    sources: list[RetrievedContext]


class ChromaService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        settings = get_settings()
        self.embedding_model = settings.EMBEDDING_MODEL
        self.collection_name = "document_collection"
        self.persist_directory = settings.CHROMA_PERSIST_DIR
        self._client = None
        self._collection = None
        self._initialized = True

    @property
    def client(self) -> chromadb.PersistentClient:
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def embedding_function(self) -> HuggingFaceEmbeddings:
        return HuggingFaceEmbeddings(
            model_name=self.embedding_model,
        )

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
            self._collection = None
        except Exception as e:
            logger.warning(f"Collection not deleted: {e}")


class RAGService:
    def __init__(self):
        self.settings = get_settings()
        self._chroma = ChromaService()
        self._text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.CHUNK_SIZE,
            chunk_overlap=self.settings.CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )

    async def ingest_document(self, file_path: str | Path) -> int:
        file_path = Path(file_path)
        if not file_path.exists():
            raise DocumentIngestionError(f"File not found: {file_path}")
            
        allowed_extensions = {".pdf", ".txt", ".md"}
        if file_path.suffix.lower() not in allowed_extensions:
            raise DocumentIngestionError(f"Unsupported file type: {file_path.name}")

        try:
            chunks, metadata = await self._extract_and_chunk(file_path)
            await self._embed_and_store(chunks, metadata)
            return len(chunks)
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Ingestion failed for {file_path.name}: {e}")
            raise DocumentIngestionError(f"Failed to ingest document: {e}") from e

    async def ingest_document_stream(
        self,
        filename: str,
        file_stream: BinaryIO,
    ) -> int:
        try:
            temp_path = Path(self.settings.UPLOAD_DIR) / filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("wb") as f:
                f.write(file_stream.read())
            return await self.ingest_document(temp_path)
        except DocumentIngestionError:
            raise
        except Exception as e:
            logger.error(f"Stream ingestion failed for {filename}: {e}")
            raise DocumentIngestionError(f"Failed to ingest document from stream: {e}") from e
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    async def answer_query(self, query: str, top_k: int = 4) -> QueryResponse:
        contexts = await self._retrieve_contexts(query, top_k)
        if not contexts:
            return QueryResponse(
                answer="I could not find any relevant context to answer your question.",
                sources=[],
            )

        return await self._generate_answer(query, contexts)

    async def _retrieve_contexts(
        self, query: str, top_k: int
    ) -> list[RetrievedContext]:
        chroma_collection = self._chroma.collection
        embedding_fn = self._chroma.embedding_function

        try:
            query_embedding = embedding_fn.embed_query(query)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise RetrievalError("Failed to embed query vector.") from e

        try:
            results = chroma_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise RetrievalError("Vector database query failed.") from e

        contexts: list[RetrievedContext] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        for doc, meta in zip(documents, metadatas):
            if doc is not None:
                contexts.append(
                    RetrievedContext(
                        text=doc,
                        filename=meta.get("source", "Unknown"),
                        page=meta.get("page", 0),
                    )
                )

        return contexts

    async def _generate_answer(
        self, query: str, contexts: list[RetrievedContext]
    ) -> QueryResponse:
        settings = get_settings()
        system_prompt = (
            "You are a precise AI Document Assistant. Answer the user's question using ONLY "
            "the provided context blocks. If the context does not contain the answer, state clearly "
            "that you cannot find it. For every factual claim you make, you MUST explicitly cite "
            "the source file name and page number at the end of the sentence or paragraph based on "
            "the context metadata."
        )

        context_text = "\n\n".join(
            f"[Source: {ctx.filename}, Page: {ctx.page}]\n{ctx.text}"
            for ctx in contexts
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {query}",
            },
        ]

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        try:
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.0,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise RetrievalError("LLM answer generation failed.") from e

        return QueryResponse(answer=answer, sources=contexts)

    async def answer_query_stream(
        self, query: str, top_k: int = 4
    ) -> AsyncGenerator[str, None]:
        """Stream the LLM answer token-by-token as Server-Sent Events."""
        contexts = await self._retrieve_contexts(query, top_k)

        if not contexts:
            yield "data: I could not find any relevant context to answer your question.\n\n"
            yield "event: sources\ndata: []\n\n"
            yield "event: done\ndata: [DONE]\n\n"
            return

        settings = get_settings()
        system_prompt = (
            "You are a precise AI Document Assistant. Answer the user's question using ONLY "
            "the provided context blocks. If the context does not contain the answer, state clearly "
            "that you cannot find it. For every factual claim you make, you MUST explicitly cite "
            "the source file name and page number at the end of the sentence or paragraph based on "
            "the context metadata."
        )
        context_text = "\n\n".join(
            f"[Source: {ctx.filename}, Page: {ctx.page}]\n{ctx.text}"
            for ctx in contexts
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
        ]

        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        try:
            stream = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=messages,
                temperature=0.0,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield f"data: {delta.content}\n\n"
        except Exception as e:
            logger.error(f"LLM streaming failed: {e}")
            yield f"data: [Error: LLM generation failed]\n\n"

        # Send source references as a JSON event
        import json
        sources_payload = json.dumps([
            {"text": ctx.text, "filename": ctx.filename, "page": ctx.page}
            for ctx in contexts
        ])
        yield f"event: sources\ndata: {sources_payload}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    async def _extract_and_chunk(
        self,
        file_path: Path,
    ) -> tuple[list[str], list[ChunkMetadata]]:
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            return await self._extract_and_chunk_pdf(file_path)
        elif ext in [".txt", ".md"]:
            return await self._extract_and_chunk_text(file_path)
        else:
            raise DocumentIngestionError(f"Unsupported file type: {ext}")

    async def _extract_and_chunk_pdf(
        self,
        file_path: Path,
    ) -> tuple[list[str], list[ChunkMetadata]]:
        try:
            reader = PdfReader(str(file_path))
        except Exception as e:
            raise DocumentIngestionError(f"Could not read PDF: {e}") from e

        chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        source = file_path.name

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num}: {e}")
                continue

            if not text or not text.strip():
                logger.warning(f"Empty page {page_num} in {source}")
                continue

            page_chunks = self._text_splitter.split_text(text)
            for chunk in page_chunks:
                if chunk.strip():
                    chunks.append(chunk)
                    chunk_metadata.append(ChunkMetadata(source=source, page=page_num))

        if not chunks:
            raise DocumentIngestionError(f"No extractable text found in {source}")

        return chunks, chunk_metadata

    async def _extract_and_chunk_text(
        self,
        file_path: Path,
    ) -> tuple[list[str], list[ChunkMetadata]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            raise DocumentIngestionError(f"Could not read text file: {e}") from e

        if not text or not text.strip():
            raise DocumentIngestionError(f"No extractable text found in {file_path.name}")

        chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        source = file_path.name
        
        page_chunks = self._text_splitter.split_text(text)
        for chunk in page_chunks:
            if chunk.strip():
                chunks.append(chunk)
                chunk_metadata.append(ChunkMetadata(source=source, page=1))

        if not chunks:
            raise DocumentIngestionError(f"No extractable text found in {source}")

        return chunks, chunk_metadata

    async def _embed_and_store(
        self,
        chunks: list[str],
        metadata: list[ChunkMetadata],
    ) -> None:
        if len(chunks) != len(metadata):
            raise ValueError("Chunks and metadata length mismatch")

        chroma_collection = self._chroma.collection
        embedding_fn = self._chroma.embedding_function

        for idx, (chunk, meta) in enumerate(zip(chunks, metadata)):
            try:
                embedding = embedding_fn.embed_query(chunk)
            except Exception as e:
                logger.error(f"Embedding failed for chunk {idx}: {e}")
                raise DocumentIngestionError("Embedding generation failed: {e}") from e

            doc_id = f"{meta.source}_page{meta.page}_chunk{idx}"
            chroma_collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": meta.source, "page": meta.page}],
            )

        logger.info(f"Stored {len(chunks)} chunks in ChromaDB")


async def get_rag_service() -> RAGService:
    return RAGService()