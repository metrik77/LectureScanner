from io import BytesIO

from docx import Document

from lecture_scanner.exports import build_organized_docx
from lecture_scanner.organizer import SourceSlide, organize, to_markdown, to_text


def test_organizer_preserves_order_text_and_sources_without_inventing_content() -> None:
    slides = [
        SourceSlide("slide-01.jpeg", "Лексикология изучает значение слова."),
        SourceSlide("slide-02.jpeg", "Значение слова рассматривается в лексикологии."),
        SourceSlide("slide-03.jpeg", "Фонетика изучает звуки речи."),
    ]
    topics = organize(slides)

    assert [slide.filename for topic in topics for slide in topic.slides] == [slide.filename for slide in slides]
    markdown, text = to_markdown(topics), to_text(topics)
    for slide in slides:
        assert slide.filename in markdown and slide.text in markdown
        assert slide.filename in text and slide.text in text
    assert "выдуманный" not in markdown


def test_organized_docx_keeps_source_references_and_text() -> None:
    topics = organize([SourceSlide("slide-01.jpeg", "Текст лекции."), SourceSlide("slide-02.jpeg", "Другой текст лекции.")])
    document = Document(BytesIO(build_organized_docx(topics)))
    paragraphs = [paragraph.text for paragraph in document.paragraphs]
    assert "Источник: slide-01.jpeg" in paragraphs
    assert "Текст лекции." in paragraphs
    assert "Источник: slide-02.jpeg" in paragraphs
