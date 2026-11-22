"""Detect the largest presentation screen and rectify it with a homography."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ScreenCandidate:
    corners: tuple[tuple[float, float], ...]  # TL, TR, BR, BL in source pixels
    area_px: float
    area_ratio: float
    rectified_width: int
    rectified_height: int
    aspect_ratio: float
    detection_method: str
    selection_algorithm: str = "primary"
    rectification_mode: str = "perspective"
    boundary_clipped: bool = False
    crop_rect: tuple[int, int, int, int] | None = None  # x0, y0, x1, y1
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DetectionResult:
    status: str
    candidates: tuple[ScreenCandidate, ...]
    warnings: tuple[str, ...]


def _order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    ordered = np.roll(ordered, -start, axis=0)
    # In image coordinates the expected order TL, TR, BR, BL has positive area.
    if cv2.contourArea(ordered.reshape(-1, 1, 2)) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def _angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first = a - b
    second = c - b
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom == 0:
        return 0.0
    cosine = float(np.dot(first, second) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _candidate_from_contour(
    contour: np.ndarray,
    image_area: float,
    min_area_ratio: float,
    min_side_px: float,
    aspect_range: tuple[float, float],
    detection_method: str,
    approximation_ratio: float = 0.02,
) -> ScreenCandidate | None:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return None
    polygon = cv2.approxPolyDP(contour, approximation_ratio * perimeter, True)
    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
        return None

    corners = _order_corners(polygon.reshape(4, 2))
    area_px = abs(float(cv2.contourArea(corners.reshape(-1, 1, 2))))
    area_ratio = area_px / image_area
    if area_ratio < min_area_ratio:
        return None

    edges = [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]
    width = int(round(max(edges[0], edges[2])))
    height = int(round(max(edges[1], edges[3])))
    if min(width, height) < min_side_px or height == 0:
        return None
    aspect_ratio = width / height
    if not aspect_range[0] <= aspect_ratio <= aspect_range[1]:
        return None

    angles = [
        _angle_degrees(corners[(index - 1) % 4], corners[index], corners[(index + 1) % 4])
        for index in range(4)
    ]
    if min(angles) < 35 or max(angles) > 145:
        return None

    return ScreenCandidate(
        corners=tuple((float(point[0]), float(point[1])) for point in corners),
        area_px=area_px,
        area_ratio=area_ratio,
        rectified_width=width,
        rectified_height=height,
        aspect_ratio=aspect_ratio,
        detection_method=detection_method,
    )


def _frame_contacts(
    corners: tuple[tuple[float, float], ...], width: int, height: int, tolerance: int = 4
) -> tuple[str, ...]:
    """Return image borders touched by the candidate's visible quadrilateral."""
    points = np.asarray(corners, dtype=np.float32)
    xs, ys = points[:, 0], points[:, 1]
    contacts: list[str] = []
    if float(xs.min()) <= tolerance:
        contacts.append("left")
    if float(xs.max()) >= width - 1 - tolerance:
        contacts.append("right")
    if float(ys.min()) <= tolerance:
        contacts.append("top")
    if float(ys.max()) >= height - 1 - tolerance:
        contacts.append("bottom")
    return tuple(contacts)


def _fallback_color_candidates(
    color_contours: Sequence[np.ndarray],
    image_area: float,
    width: int,
    height: int,
    min_area_ratio: float,
    min_side_px: float,
    aspect_range: tuple[float, float],
) -> list[ScreenCandidate]:
    """Recover a slide which fills the frame and has clipped screen borders.

    This runs only after the primary detector has found no candidate. A fallback
    candidate must therefore be a dominant chromatic region touching at least
    two image borders; ordinary four-sided room objects do not qualify.
    """
    recovered: list[ScreenCandidate] = []
    for contour in color_contours:
        contour_area_ratio = abs(float(cv2.contourArea(contour))) / image_area
        if contour_area_ratio < 0.35:
            continue
        for epsilon in (0.025, 0.03, 0.04, 0.05):
            base = _candidate_from_contour(
                contour,
                image_area,
                min_area_ratio,
                min_side_px,
                aspect_range,
                "fallback_color_surface",
                approximation_ratio=epsilon,
            )
            if base is None:
                continue
            contacts = _frame_contacts(base.corners, width, height)
            # The fallback is intended for a slide that is visibly clipped by
            # the camera frame. This guard is what keeps a large coloured wall,
            # poster, or random quadrilateral from being accepted on geometry.
            if len(contacts) < 2 or base.area_ratio < 0.35:
                continue
            points = np.asarray(base.corners, dtype=np.float32)
            x0 = max(0, int(math.floor(float(points[:, 0].min()))))
            y0 = max(0, int(math.floor(float(points[:, 1].min()))))
            x1 = min(width, int(math.ceil(float(points[:, 0].max()))) + 1)
            y1 = min(height, int(math.ceil(float(points[:, 1].max()))) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            recovered.append(
                ScreenCandidate(
                    corners=base.corners,
                    area_px=base.area_px,
                    area_ratio=base.area_ratio,
                    rectified_width=x1 - x0,
                    rectified_height=y1 - y0,
                    aspect_ratio=(x1 - x0) / (y1 - y0),
                    detection_method=base.detection_method,
                    selection_algorithm="fallback",
                    rectification_mode="crop",
                    boundary_clipped=True,
                    crop_rect=(x0, y0, x1, y1),
                    warnings=(
                        "Границы экрана частично обрезаны краями фотографии; "
                        "использовано кадрирование без перспективного преобразования.",
                        f"Fallback: крупный цветовой контур ({contour_area_ratio:.1%}) "
                        f"упрощён с epsilon={epsilon:.3f}; касания границ: {', '.join(contacts)}.",
                    ),
                )
            )
            break
    return _deduplicate(recovered)


def _deduplicate(candidates: Sequence[ScreenCandidate]) -> list[ScreenCandidate]:
    """Collapse nested inner/outer frame contours for the same physical screen."""
    kept: list[ScreenCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.area_px, reverse=True):
        polygon = np.asarray(candidate.corners, dtype=np.float32)
        duplicate = False
        for existing in kept:
            other = np.asarray(existing.corners, dtype=np.float32)
            intersection, _ = cv2.intersectConvexConvex(polygon, other)
            smaller_area = min(candidate.area_px, existing.area_px)
            if smaller_area and intersection / smaller_area >= 0.88:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def detect_screens(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.025,
    min_side_px: float = 100,
    aspect_range: tuple[float, float] = (1.2, 2.3),
) -> list[ScreenCandidate]:
    """Return plausible screen quadrilaterals sorted by area, largest first."""
    if image is None or image.size == 0:
        raise ValueError("Пустое изображение")
    height, width = image.shape[:2]
    image_area = float(width * height)
    # A chromatic screen surface is a useful boundary cue in lecture photos:
    # the screen often differs in hue from neutral walls and ceilings. Avoid
    # closing this mask aggressively, since that can join the screen to the room.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    blue_surface = (image[:, :, 0].astype(np.int16) > image[:, :, 2].astype(np.int16) + 30) & (value > 145)
    color_mask = (((saturation > 45) & (value > 120)) | blue_surface).astype(np.uint8) * 255
    color_contours, _ = cv2.findContours(
        color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    color_candidates = [
        candidate
        for contour in color_contours
        if (
            candidate := _candidate_from_contour(
                contour,
                image_area,
                min_area_ratio,
                min_side_px,
                aspect_range,
                "color_surface",
            )
        )
        is not None
    ]
    color_candidates = _deduplicate(color_candidates)
    if color_candidates and max(c.area_ratio for c in color_candidates) >= 0.08:
        return color_candidates

    # Fallback for neutral/white screens with weak color separation. Raw edge
    # contours are used without dilation so room edges do not merge as easily.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edge_contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    edge_candidates = [
        candidate
        for contour in edge_contours
        if (
            candidate := _candidate_from_contour(
                contour,
                image_area,
                min_area_ratio,
                min_side_px,
                aspect_range,
                "canny_edges",
            )
        )
        is not None
    ]
    edge_candidates = _deduplicate(edge_candidates)
    if edge_candidates:
        return edge_candidates

    # Keep primary thresholds and its output unchanged. This route is reached
    # only after both primary colour and Canny passes found no valid candidate.
    return _fallback_color_candidates(
        color_contours,
        image_area,
        width,
        height,
        min_area_ratio,
        min_side_px,
        aspect_range,
    )


def detect_screen_result(
    image: np.ndarray,
    *,
    min_area_ratio: float = 0.025,
    min_side_px: float = 100,
    aspect_range: tuple[float, float] = (1.2, 2.3),
) -> DetectionResult:
    """Return a reviewable state when neither detector confirms a screen."""
    candidates = detect_screens(
        image,
        min_area_ratio=min_area_ratio,
        min_side_px=min_side_px,
        aspect_range=aspect_range,
    )
    reliable = tuple(candidate for candidate in candidates if candidate.area_ratio >= 0.08)
    if reliable:
        return DetectionResult("detected", reliable, ())
    return DetectionResult(
        "needs_review",
        tuple(candidates),
        (
            "Основной и резервный алгоритмы не подтвердили область экрана; "
            "требуется ручная проверка или OCR полной фотографии.",
        ),
    )


def rectify_screen(
    image: np.ndarray,
    candidate: ScreenCandidate,
    *,
    min_area_ratio: float = 0.08,
) -> np.ndarray:
    """Perspective-warp only the selected quadrilateral; reject weak detections."""
    if candidate.area_ratio < min_area_ratio:
        raise ValueError(
            f"Надёжный экран не найден: площадь кандидата "
            f"{candidate.area_ratio:.1%} меньше порога {min_area_ratio:.1%}"
        )
    if candidate.rectification_mode == "crop":
        if candidate.crop_rect is None:
            raise ValueError("Для кадрирования отсутствуют координаты прямоугольника")
        x0, y0, x1, y1 = candidate.crop_rect
        if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
            raise ValueError("Координаты кадрирования выходят за пределы изображения")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Некорректный прямоугольник кадрирования")
        return image[y0:y1, x0:x1].copy()

    source = np.asarray(candidate.corners, dtype=np.float32)
    destination = np.asarray(
        [
            [0, 0],
            [candidate.rectified_width - 1, 0],
            [candidate.rectified_width - 1, candidate.rectified_height - 1],
            [0, candidate.rectified_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        transform,
        (candidate.rectified_width, candidate.rectified_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Исходная фотография")
    parser.add_argument("--output", type=Path, required=True, help="Путь для выровненного экрана")
    parser.add_argument("--metadata", type=Path, help="JSON с углами и площадями")
    args = parser.parse_args()

    started_at = time.perf_counter()
    source_path = args.image.expanduser().resolve()
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        parser.error(f"Не удалось прочитать изображение: {source_path}")

    candidates = detect_screens(image)
    reliable = [candidate for candidate in candidates if candidate.area_ratio >= 0.08]
    if not reliable:
        parser.error("Экран нельзя надёжно обнаружить по контурам и геометрии")
    selected = max(reliable, key=lambda candidate: candidate.area_px)

    rectified = rectify_screen(image, selected)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), rectified):
        parser.error(f"Не удалось сохранить выровненный экран: {output_path}")
    elapsed_seconds = time.perf_counter() - started_at

    metadata = {
        "source_path": str(source_path),
        "source_size_px": [int(image.shape[1]), int(image.shape[0])],
        "selected_screen": asdict(selected),
        "candidates": [asdict(candidate) for candidate in candidates],
        "ignored_smaller_screens": max(0, len(candidates) - 1),
        "rectified_path": str(output_path),
        "rectified_size_px": [int(rectified.shape[1]), int(rectified.shape[0])],
        "selection_rule": (
            f"{selected.selection_algorithm}: largest reliable quadrilateral "
            "by polygon area"
        ),
        "processing_time_seconds": elapsed_seconds,
    }
    if args.metadata:
        metadata_path = args.metadata.expanduser().resolve()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"Найдено экранов-кандидатов: {len(candidates)}")
    print(f"Выбран экран площадью {selected.area_px:.0f} px² ({selected.area_ratio:.1%})")
    print(f"Формат после выравнивания: {selected.rectified_width}x{selected.rectified_height}")
    print(f"Время обнаружения и выравнивания: {elapsed_seconds:.3f} с")
    print("Углы TL, TR, BR, BL:")
    for name, point in zip(("TL", "TR", "BR", "BL"), selected.corners):
        print(f"  {name}: ({point[0]:.1f}, {point[1]:.1f})")
    print(f"Сохранено: {output_path}")
    if args.metadata:
        print(f"Координаты и кандидаты сохранены: {args.metadata.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
