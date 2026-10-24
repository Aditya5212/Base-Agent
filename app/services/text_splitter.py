"""Structure-aware text splitter service using LangChain splitters and tiktoken.

Combines MarkdownHeaderTextSplitter (to preserve header hierarchy) and
RecursiveCharacterTextSplitter.from_tiktoken_encoder (for precise token-bounded chunking).
"""

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import tiktoken

from app.core.rag_config import RAGConfig, get_rag_config

logger = logging.getLogger(__name__)

# Standard Markdown header tags to capture into chunk metadata
DEFAULT_HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
]

# Cached tiktoken encoder
_tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Calculates exact token length for a given text using OpenAI cl100k_base encoding."""
    if not text:
        return 0
    return len(_tokenizer.encode(text, disallowed_special=()))


def split_markdown_document(
    document: Document,
    config: RAGConfig | None = None,
    headers_to_split_on: Sequence[tuple[str, str]] | None = None,
) -> list[Document]:
    """Splits a Markdown Document into token-bounded, structure-aware chunks.

    Args:
        document: Input Document containing Markdown page_content.
        config: RAGConfig specifying chunk_size and chunk_overlap (in tokens).
        headers_to_split_on: Header prefixes and metadata field names.

    Returns:
        List of chunk Documents with header hierarchy metadata and exact token counts.
    """
    rag_cfg = config or get_rag_config()
    headers = list(headers_to_split_on or DEFAULT_HEADERS_TO_SPLIT_ON)

    # Step 1: Split along markdown header boundaries
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    )
    section_docs = header_splitter.split_text(document.page_content)

    # If the document has no headers, fallback to a single document with the original text
    if not section_docs:
        section_docs = [Document(page_content=document.page_content, metadata={})]

    # Step 2: Sub-split sections into token-bounded chunks
    token_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=rag_cfg.chunk_size,
        chunk_overlap=rag_cfg.chunk_overlap,
        add_start_index=True,
    )
    raw_chunks = token_splitter.split_documents(section_docs)

    # Step 3: Enrich chunks with parent document metadata and chunk index
    final_chunks: list[Document] = []
    parent_metadata = dict(document.metadata)

    for idx, chunk in enumerate(raw_chunks):
        merged_metadata = dict(parent_metadata)
        # Merge header hierarchy captured by MarkdownHeaderTextSplitter
        merged_metadata.update(chunk.metadata)
        merged_metadata["chunk_index"] = idx
        merged_metadata["token_count"] = count_tokens(chunk.page_content)

        final_chunks.append(
            Document(
                page_content=chunk.page_content,
                metadata=merged_metadata,
            )
        )

    logger.debug(
        "Split document into %d chunks (chunk_size=%d, overlap=%d)",
        len(final_chunks),
        rag_cfg.chunk_size,
        rag_cfg.chunk_overlap,
    )
    return final_chunks
