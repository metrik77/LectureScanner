from pathlib import Path

import cv2
import numpy as np

from main_screen_detector import _fallback_color_candidates, detect_screen_result, detect_screens

ROOT = Path(__file__).resolve().parents[1]


def test_generated_bilingual_slide_keeps_primary_candidate():
    image = cv2.imread(str(ROOT / "fixtures" / "bilingual_slide_sample.png"))
    assert image is not None
    candidates = detect_screens(image)
    assert candidates
    assert candidates[0].selection_algorithm == "primary"


def test_fallback_rejects_large_region_not_clipped_by_frame():
    contour = np.array([[[100, 100]], [[900, 100]], [[900, 700]], [[100, 700]]], dtype=np.int32)
    candidates = _fallback_color_candidates(
        [contour], image_area=1_000_000, width=1000, height=1000,
        min_area_ratio=0.025, min_side_px=100, aspect_range=(1.2, 2.3),
    )
    assert candidates == []


def test_missing_screen_returns_needs_review_status():
    image = np.full((300, 500, 3), 128, dtype=np.uint8)
    result = detect_screen_result(image)
    assert result.status == "needs_review"
    assert result.candidates == ()
