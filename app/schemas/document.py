"""Pydantic schemas for document ingestion, storage, and RAG search."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    """Response returned upon uploading and indexing a document."""

    id: str = Field(..., description="Unique document UUID")
    file_name: str = Field(..., description="Original file name")
    file_size: int = Field(..., description="File size in bytes")
    status: str = Field(..., description="Current lifecycle status (PENDING, INDEXED, FAILED)")
    chunks_count: int = Field(..., description="Number of chunks extracted and indexed")
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
