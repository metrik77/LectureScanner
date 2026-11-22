from io import BytesIO

from docx import Document

from lecture_scanner.exports import build_docx


def test_docx_export_keeps_slide_sections_and_text() -> None:
    payload = build_docx([("slide-01.jpeg", "Лекция\nLecture"), ("slide-02.jpeg", "Термин")])
    document = Document(BytesIO(payload))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    assert "slide-01.jpeg" in paragraphs
    assert "Лекция" in paragraphs
    assert "Lecture" in paragraphs
    assert "slide-02.jpeg" in paragraphs
    assert "Термин" in paragraphs

