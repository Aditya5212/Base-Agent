"""Unit tests for document loader service using MarkItDown."""

import io
from app.services.document_loader import convert_to_markdown, load_document


def test_convert_markdown_plain_text():
    sample_text = b"Hello, this is a plain text file for testing."
    result = convert_to_markdown(sample_text, filename="sample.txt")
    assert "Hello, this is a plain text file" in result


def test_convert_markdown_html():
    html_content = b"<html><body><h1>Title</h1><p>Paragraph text.</p></body></html>"
    result = convert_to_markdown(html_content, filename="test.html")
    assert "Title" in result
    assert "Paragraph text." in result


def test_load_document_metadata():
    content = b"# Document Header\n\nContent line with details."
    doc = load_document(
        source=content,
        filename="guide.md",
        extra_metadata={"user_id": "test_user_1", "version": "1.0"},
    )
    assert "Document Header" in doc.page_content
    assert doc.metadata["source"] == "guide.md"
    assert doc.metadata["filename"] == "guide.md"
    assert doc.metadata["user_id"] == "test_user_1"
    assert doc.metadata["version"] == "1.0"
