"""Document formats for the reviewed lecture text."""

from __future__ import annotations

from io import BytesIO
from collections.abc import Iterable

from docx import Document

from .organizer import Topic


def build_docx(slides: Iterable[tuple[str, str]]) -> bytes:
    document = Document()
    document.add_heading("Lecture Scanner — конспект лекции", level=0)
    for filename, text in slides:
        document.add_heading(filename, level=1)
        lines = text.splitlines() or [""]
        for line in lines:
            document.add_paragraph(line)
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def build_organized_docx(topics: Iterable[Topic]) -> bytes:
    document = Document()
    document.add_heading("Lecture Scanner — структурированный конспект", level=0)
    for topic in topics:
        document.add_heading(topic.title, level=1)
        for slide in topic.slides:
            document.add_heading(f"Источник: {slide.filename}", level=2)
            for line in slide.text.splitlines() or [""]:
                document.add_paragraph(line)
    output = BytesIO()
    document.save(output)
    return output.getvalue()

