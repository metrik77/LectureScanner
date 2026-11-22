from pathlib import Path

import cv2
import numpy as np
import hashlib

from ocr_quality import (
    RegionQC,
    analyze_region,
    crop_and_upscale,
    retry_region,
    set_review_decision,
)


def test_low_confidence_is_flagged_and_text_is_not_changed():
    result = analyze_region(1, "сомнительный текст", 0.61, [[2, 2], [90, 2], [90, 22], [2, 22]])

    assert result.review_status == "pending"
    assert result.flags
    assert result.original_text == "сомнительный текст"
    assert result.corrected_text is None


def test_high_confidence_suspicious_word_is_flagged():
    result = analyze_region(1, "нет одкого", 0.966, [[1, 1], [100, 1], [100, 20], [1, 20]])

    assert result.original_confidence > 0.95
    assert "Подозрительная последовательность букв" in result.flags
    assert result.review_status == "pending"


def test_mixed_latin_cyrillic_is_flagged():
    result = analyze_region(1, "тeст", 0.99, [[0, 0], [80, 0], [80, 20], [0, 20]])

    assert "Смешение латиницы и кириллицы в слове" in result.flags


def test_crop_uses_roi_and_upscales():
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    crop = crop_and_upscale(image, [[10, 20], [50, 20], [50, 40], [10, 40]], 2.0)

    assert crop.shape[1] > 80
    assert crop.shape[0] > 40
    assert crop.shape[0] < 80


def test_retry_compares_only_the_selected_roi_and_review_is_explicit(tmp_path: Path):
    image_path = tmp_path / "source.png"
    image = np.full((60, 140, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    original_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()

    class FakeRecognizer:
        def __init__(self):
            self.paths = []

        def recognize(self, image_path):
            self.paths.append(Path(image_path))
            return type(
                "Document",
                (),
                {"lines": [type("Line", (), {"text": "одного", "confidence": 0.98})()]},
            )()

    recognizer = FakeRecognizer()
    region = RegionQC(
        region_id=1,
        original_text="одкого",
        retry_text=None,
        original_confidence=0.966,
        retry_confidence=None,
        coordinates=[[10, 15], [90, 15], [90, 35], [10, 35]],
        review_status="pending",
        corrected_text=None,
        flags=["Подозрительная последовательность букв"],
    )
    retried = retry_region(image_path, region, recognizer, 2.0)

    assert retried.original_text == "одкого"
    assert retried.retry_text == "одного"
    assert retried.original_confidence == 0.966
    assert retried.retry_confidence == 0.98
    assert recognizer.paths and not recognizer.paths[0].exists()
    assert retried.corrected_text is None
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == original_hash

    set_review_decision(retried, accept_retry=True)
    assert retried.original_text == "одкого"
    assert retried.corrected_text == "одного"
    assert retried.review_status == "accepted_retry"


def test_keep_original_never_overwrites_original():
    result = RegionQC(
        region_id=1,
        original_text="одкого",
        retry_text="одного",
        original_confidence=0.966,
        retry_confidence=0.98,
        coordinates=None,
        review_status="pending",
        corrected_text=None,
        flags=["Подозрительная последовательность букв"],
    )

    set_review_decision(result, accept_retry=False)
    assert result.original_text == "одкого"
    assert result.corrected_text is None
    assert result.review_status == "kept_original"
