from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from main_screen_detector import ScreenCandidate
from lecture_scanner.processing import process_photo


@dataclass(frozen=True)
class FakeLine:
    text: str
    confidence: float
    polygon: list[list[int]] | None = None


@dataclass(frozen=True)
class FakeDocument:
    source_path: Path
    lines: tuple[FakeLine, ...]
    original_text: str
    average_confidence: float | None
    manual_review_required: bool = False


def make_photo(path: Path) -> None:
    image = np.full((80, 160, 3), 235, dtype=np.uint8)
    cv2.putText(image, "Slide text", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    assert cv2.imwrite(str(path), image)


def test_missing_screen_uses_whole_photo_and_saves_raw_ocr(tmp_path: Path) -> None:
    source = tmp_path / "lecture.jpeg"
    make_photo(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()

    class Recognizer:
        def __init__(self) -> None:
            self.paths: list[Path] = []

        def recognize(self, path: str | Path) -> FakeDocument:
            image_path = Path(path)
            self.paths.append(image_path)
            lines = (
                FakeLine("Заголовок слайда", 0.98),
                FakeLine("Основной текст лекции", 0.97),
            )
            return FakeDocument(image_path, lines, "Заголовок слайда\nОсновной текст лекции", 0.975)

    recognizer = Recognizer()
    outcome = process_photo(
        source,
        source_sha256="a" * 64,
        results_dir=tmp_path / "results",
        recognizer=recognizer,
        screen_detector=lambda image: [],
    )

    assert outcome.primary.method == "full_image"
    assert outcome.quality_status == "usable"
    assert recognizer.paths == [source.resolve()]
    assert outcome.primary.report_path.is_file()
    assert outcome.manifest_path.is_file()
    assert "Главный экран не обнаружен" in outcome.warnings[0]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_short_full_photo_ocr_triggers_upscaled_alternative(tmp_path: Path) -> None:
    source = tmp_path / "lecture.jpeg"
    make_photo(source)

    class Recognizer:
        def recognize(self, path: str | Path) -> FakeDocument:
            image_path = Path(path)
            if "upscaled" not in image_path.name:
                return FakeDocument(image_path, (), "", None)
            lines = (FakeLine("Достаточно длинная первая строка", 0.91), FakeLine("И вторая строка текста", 0.90))
            return FakeDocument(image_path, lines, "Достаточно длинная первая строка\nИ вторая строка текста", 0.905)

    outcome = process_photo(
        source,
        source_sha256="b" * 64,
        results_dir=tmp_path / "results",
        recognizer=Recognizer(),
        screen_detector=lambda image: [],
    )

    assert [variant.method for variant in outcome.variants] == ["full_image", "full_image_upscaled_2x"]
    assert outcome.primary.method == "full_image_upscaled_2x"
    assert outcome.quality_status == "usable"
    assert any("слишком мало текста" in warning for warning in outcome.warnings)


def test_low_quality_screen_ocr_falls_back_to_whole_photo(tmp_path: Path) -> None:
    source = tmp_path / "lecture.jpeg"
    make_photo(source)
    candidate = ScreenCandidate(
        corners=((0.0, 0.0), (159.0, 0.0), (159.0, 79.0), (0.0, 79.0)),
        area_px=12_561.0,
        area_ratio=0.98,
        rectified_width=159,
        rectified_height=79,
        aspect_ratio=2.01,
        detection_method="test",
    )

    class Recognizer:
        def recognize(self, path: str | Path) -> FakeDocument:
            image_path = Path(path)
            if "main-screen" in image_path.name:
                return FakeDocument(image_path, (FakeLine("Title", 0.99),), "Title", 0.99)
            lines = (FakeLine("Полный кадр содержит текст", 0.97), FakeLine("Во второй строке", 0.96))
            return FakeDocument(image_path, lines, "Полный кадр содержит текст\nВо второй строке", 0.965)

    outcome = process_photo(
        source,
        source_sha256="c" * 64,
        results_dir=tmp_path / "results",
        recognizer=Recognizer(),
        screen_detector=lambda image: [candidate],
    )

    assert [variant.method for variant in outcome.variants] == [
        "main_screen",
        "full_image_after_low_screen",
    ]
    assert outcome.primary.method == "full_image_after_low_screen"
    assert outcome.quality_status == "usable"

