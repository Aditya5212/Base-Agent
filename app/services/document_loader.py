"""Document loader service using Microsoft MarkItDown.

Converts documents of various formats (PDF, DOCX, PPTX, XLSX, HTML, TXT, MD) into
clean, semantic Markdown suitable for structure-aware chunking and retrieval.
"""

import io
import logging
from pathlib import Path
from typing import BinaryIO

from langchain_core.documents import Document
from markitdown import MarkItDown
from markitdown._stream_info import StreamInfo

logger = logging.getLogger(__name__)

# Singleton MarkItDown converter instance
_converter = MarkItDown()


def convert_to_markdown(
    source: str | Path | bytes | BinaryIO,
    filename: str | None = None,
) -> str:
    """Converts a document source to clean Markdown text.

    Args:
        source: Local filepath string, Path object, raw bytes, or readable BinaryIO stream.
        filename: Optional filename hint to assist format/extension detection.

    Returns:
        Clean, structured Markdown text.
    """
    try:
        if isinstance(source, (str, Path)):
            result = _converter.convert(str(source))
            return result.text_content

        if isinstance(source, bytes):
            ext = Path(filename).suffix if filename else None
            stream = io.BytesIO(source)
            info = StreamInfo(extension=ext) if ext else None
            result = _converter.convert(stream, stream_info=info)
            return result.text_content

        # Assume BinaryIO
        ext = Path(filename).suffix if filename else None
        info = StreamInfo(extension=ext) if ext else None
        result = _converter.convert(source, stream_info=info)
        return result.text_content

    except Exception as exc:
        logger.error("Failed to convert document '%s' to markdown: %s", filename or "unknown", exc, exc_info=True)
        raise ValueError(f"Could not convert document '{filename or 'unknown'}' to Markdown: {exc}") from exc


def load_document(
    source: str | Path | bytes | BinaryIO,
    filename: str,
    extra_metadata: dict | None = None,
) -> Document:
    """Loads a document, converts it to Markdown, and returns a LangChain Document.

    Args:
        source: File path, Path object, raw bytes, or binary stream.
        filename: Name of the file being loaded.
        extra_metadata: Additional key-value pairs (e.g. user_id, document_id) to attach.

    Returns:
        langchain_core.documents.Document with markdown content and metadata.
    """
    markdown_content = convert_to_markdown(source, filename=filename)
    metadata = {
        "source": filename,
        "filename": filename,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return Document(page_content=markdown_content, metadata=metadata)
