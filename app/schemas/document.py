"""Pydantic schemas for document ingestion, storage, and RAG search."""

from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading and indexing a document."""

    id: str = Field(..., description="Unique document UUID")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current lifecycle status (PENDING, PARSING, INDEXED, FAILED)")
    chunks_count: int = Field(default=0, description="Number of chunks extracted and indexed (0 while processing in background)")
    r2_key: str = Field(..., description="Object key in Cloudflare R2")
    message: str = Field(..., description="Status summary message")


class DocumentItemResponse(BaseModel):
    """Representation of an uploaded document record."""

    id: str = Field(..., description="Document UUID")
    file_name: str = Field(..., description="Original file name")
    mime_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current status")
    created_at: datetime = Field(..., description="Upload timestamp")
    download_url: str | None = Field(default=None, description="Presigned download URL")


class DocumentListResponse(BaseModel):
    """List of documents uploaded by a user."""

    total: int = Field(..., description="Total document count")
    documents: list[DocumentItemResponse] = Field(..., description="List of document records")


class DeleteDocumentResponse(BaseModel):
    """Response returned after deleting a document."""

    id: str = Field(..., description="Document ID")
    deleted: bool = Field(..., description="Whether deletion was successful")
    chunks_removed: int = Field(..., description="Number of vector & FTS chunks purged")
    message: str = Field(..., description="Status description")


class RAGSearchRequest(BaseModel):
    """Request payload to query the hybrid RAG search engine."""

    query: str = Field(..., min_length=1, description="Question or search query text")
    preset: str = Field(default="balanced", description="Precision vs recall preset (high_precision, balanced, high_recall)")
    top_k: int | None = Field(default=None, ge=1, le=30, description="Optional top-k override")
    user_id: str | None = Field(default=None, description="Optional user filter to scope retrieval")


class RAGSearchChunkResponse(BaseModel):
    """Individual retrieved and fused context chunk."""

    content: str = Field(..., description="Chunk text content")
    source: str = Field(default="unknown", description="Originating document file name")
    chunk_index: int | None = Field(default=None, description="Index within parent document")
    rrf_score: float | None = Field(default=None, description="Reciprocal Rank Fusion score")
    dense_rank: int | None = Field(default=None, description="Dense vector similarity rank")
    sparse_rank: int | None = Field(default=None, description="Sparse PostgreSQL FTS rank")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk header hierarchy and metadata")


class RAGSearchResponse(BaseModel):
    """Response containing hybrid search results fused via RRF."""

    query: str = Field(..., description="Original query text")
    preset: str = Field(..., description="Applied RAG preset profile")
    total_results: int = Field(..., description="Number of fused results returned")
    results: list[RAGSearchChunkResponse] = Field(..., description="Ranked list of context chunks")


class DocumentDetailResponse(BaseModel):
    """Comprehensive details for an individual document in the knowledge base."""

    id: str = Field(..., description="Document UUID")
    file_name: str = Field(..., description="Original file name")
    mime_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    formatted_size: str = Field(..., description="Human-readable size string")
    status: str = Field(..., description="Lifecycle status: PENDING, PARSING, INDEXED, FAILED")
    error_message: str | None = Field(default=None, description="Ingestion error if failed")
    indexing_config: dict[str, Any] = Field(default_factory=dict)
    provider_status: dict[str, Any] = Field(default_factory=dict)
    chunks_count: int = Field(..., description="Number of chunks currently indexed")
    r2_key: str = Field(..., description="Object key in Cloudflare R2")
    download_url: str | None = Field(default=None, description="Presigned download URL")
    created_at: datetime = Field(..., description="Upload timestamp")
    updated_at: datetime | None = Field(default=None, description="Last updated timestamp")


class DocumentChunkItem(BaseModel):
    """Granular chunk item for inspection in knowledge base."""

    id: str = Field(..., description="Chunk UUID")
    chunk_index: int = Field(..., description="Zero-indexed chunk sequence number")
    token_count: int = Field(..., description="Token count (tiktoken cl100k_base)")
    section_header: str | None = Field(default=None, description="Heading section path (e.g. 'Overview > Arch')")
    content: str = Field(..., description="Full chunk text")
    preview: str = Field(..., description="Short text preview")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary")


class DocumentChunksListResponse(BaseModel):
    """List of chunks for an individual document."""

    document_id: str = Field(..., description="Parent document UUID")
    file_name: str = Field(..., description="Parent document file name")
    total_chunks: int = Field(..., description="Total chunks in document")
    chunks: list[DocumentChunkItem] = Field(..., description="List of chunk items")


class ReindexDocumentRequest(BaseModel):
    """Request payload to re-index an existing document from R2 storage."""

    preset: Literal["balanced", "high_precision", "high_recall"] = Field(
        default="balanced",
        description="Chunking preset (high_precision, balanced, high_recall)",
    )
    index_graph: bool | None = Field(default=None, description="Override optional graph indexing; otherwise preserve the document setting")


class ReindexDocumentResponse(BaseModel):
    """Response returned upon re-indexing a document."""

    document_id: str = Field(..., description="Document UUID")
    file_name: str = Field(..., description="Document file name")
    previous_chunks: int = Field(default=0, description="Number of previous chunks before re-indexing")
    new_chunks: int = Field(default=0, description="Number of new chunks indexed (0 while processing in background)")
    preset: str = Field(..., description="Applied RAG preset profile")
    status: str = Field(..., description="Document status")
    message: str = Field(..., description="Status summary")


class KnowledgeBaseStatsResponse(BaseModel):
    """Aggregated metrics across user's Knowledge Base."""

    total_documents: int = Field(..., description="Total documents in knowledge base")
    total_chunks: int = Field(..., description="Total indexed text chunks")
    total_storage_bytes: int = Field(..., description="Combined file size in bytes")
    formatted_storage: str = Field(..., description="Formatted storage size (e.g. 5.42 MB)")
    status_breakdown: dict[str, int] = Field(..., description="Count of documents per status (INDEXED, FAILED, etc.)")
    active_preset_default: str = Field(default="balanced", description="Default RAG retrieval preset")
