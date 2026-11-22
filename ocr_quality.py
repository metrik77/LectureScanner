"""Quality checks and targeted retry/review for saved PaddleOCR regions."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2


DEFAULT_CONFIDENCE_THRESHOLD = 0.92
DEFAULT_UPSCALE_FACTOR = 2.0
_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]+")
_SUSPICIOUS_CYRILLIC_PAIRS = ("зп", "пз")


class RegionRecognizer(Protocol):
    def recognize(self, image_path: str | Path) -> Any: ...


@dataclass
class RegionQC:
    region_id: int
    original_text: str
    retry_text: str | None
    original_confidence: float
    retry_confidence: float | None
    coordinates: list[list[int]] | None
    review_status: str
    corrected_text: str | None
    flags: list[str]
    retry_error: str | None = None
    retry_processing_seconds: float | None = None


def _has_mixed_script(word: str) -> bool:
    scripts: set[str] = set()
    for char in word:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        if "CYRILLIC" in name:
            scripts.add("CYRILLIC")
        elif "LATIN" in name:
            scripts.add("LATIN")
    return "CYRILLIC" in scripts and "LATIN" in scripts


def _has_mixed_script_token(text: str) -> bool:
    return any(_has_mixed_script(word) for word in _WORD_RE.findall(text))


def _has_unusual_word_pattern(text: str) -> bool:
    for word in _WORD_RE.findall(text.casefold()):
        if _has_mixed_script(word):
            return True
        if any(pair in word for pair in _SUSPICIOUS_CYRILLIC_PAIRS):
            return True
        # "дк" is uncommon inside ordinary Russian words. Exempt common prefix
        # constructions (подк..., надк...), and leave the decision to review.
        if "дк" in word and not word.startswith(("подк", "надк")):
            return True
        if re.search(r"(.)\1{2,}", word):
            return True
    return False


def analyze_region(
    region_id: int,
    text: str,
    confidence: float,
    coordinates: list[list[int]] | None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> RegionQC:
    """Flag a region by confidence and text-shape heuristics, without editing it."""
    flags: list[str] = []
    if confidence < confidence_threshold:
        flags.append(f"Низкая уверенность: {confidence:.3f} < {confidence_threshold:.2f}")
    if _has_mixed_script_token(text):
        flags.append("Смешение латиницы и кириллицы в слове")
    if _has_unusual_word_pattern(text):
        flags.append("Подозрительная последовательность букв")
    if text.count("*") or re.search(r"[^\w\s.,:;!?()«»„“”'’—–-]", text, re.UNICODE):
        flags.append("Необычный символ или пунктуация")

    return RegionQC(
        region_id=region_id,
        original_text=text,
        retry_text=None,
        original_confidence=float(confidence),
        retry_confidence=None,
        coordinates=coordinates,
        review_status="pending" if flags else "not_flagged",
        corrected_text=None,
        flags=flags,
    )


def crop_and_upscale(
    image: Any,
    coordinates: list[list[int]],
    upscale_factor: float = DEFAULT_UPSCALE_FACTOR,
    padding_ratio: float = 0.08,
) -> Any:
    """Crop the OCR polygon's bounding region and enlarge it for a targeted retry."""
    if image is None or image.size == 0:
        raise ValueError("Пустое изображение")
    points = [(int(point[0]), int(point[1])) for point in coordinates]
    if len(points) < 4:
        raise ValueError("Для ROI нужны координаты четырёх углов")
    height, width = image.shape[:2]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = max(2, int((x_max - x_min) * padding_ratio))
    pad_y = max(2, int((y_max - y_min) * padding_ratio))
    x_min, x_max = max(0, x_min - pad_x), min(width, x_max + pad_x)
    y_min, y_max = max(0, y_min - pad_y), min(height, y_max + pad_y)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Координаты ROI выходят за изображение")
    crop = image[y_min:y_max, x_min:x_max]
    if upscale_factor <= 0:
        raise ValueError("Масштаб увеличения должен быть положительным")
    if upscale_factor != 1.0:
        crop = cv2.resize(
            crop,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def _document_lines(document: Any) -> list[tuple[str, float]]:
    if hasattr(document, "lines"):
        return [(line.text, float(line.confidence)) for line in document.lines]
    return []


def retry_region(
    source_image: str | Path,
    region: RegionQC,
    recognizer: RegionRecognizer,
    upscale_factor: float = DEFAULT_UPSCALE_FACTOR,
) -> RegionQC:
    """Run OCR on a temporary enlarged ROI; the source image is never written."""
    if not region.flags or not region.coordinates:
        return region
    started = time.perf_counter()
    source_path = Path(source_image).expanduser().resolve()
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {source_path}")
    roi = crop_and_upscale(image, region.coordinates, upscale_factor)

    try:
        with tempfile.TemporaryDirectory(prefix="lecture_scanner_roi_") as folder:
            roi_path = Path(folder) / "region.png"
            if not cv2.imwrite(str(roi_path), roi):
                raise OSError("Не удалось сохранить временный ROI")
            retry_document = recognizer.recognize(roi_path)
        lines = _document_lines(retry_document)
        region.retry_text = " ".join(text for text, _ in lines).strip()
        region.retry_confidence = (
            sum(score for _, score in lines) / len(lines) if lines else None
        )
        region.review_status = "pending" if region.retry_text else "retry_empty"
    except Exception as error:
        region.retry_error = f"{type(error).__name__}: {error}"
        region.review_status = "retry_failed"
    region.retry_processing_seconds = time.perf_counter() - started
    return region


def set_review_decision(region: RegionQC, accept_retry: bool) -> RegionQC:
    """Record an explicit human decision without changing original OCR text."""
    if accept_retry and region.retry_text is not None:
        region.corrected_text = region.retry_text
        region.review_status = "accepted_retry"
    else:
        region.corrected_text = None
        region.review_status = "kept_original"
    return region


def analyze_report(
    report: dict[str, Any], confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> list[RegionQC]:
    results = []
    for index, line in enumerate(report.get("lines", []), start=1):
        results.append(
            analyze_region(
                index,
                str(line.get("text", "")),
                float(line.get("confidence", 0.0)),
                line.get("polygon"),
                confidence_threshold,
            )
        )
    return results


def _interactive_review(regions: list[RegionQC]) -> None:
    for region in regions:
        if region.review_status != "pending" or region.retry_text is None:
            continue
        print(f"\nОбласть #{region.region_id}; координаты: {region.coordinates}")
        print(f"OCR [{region.original_confidence:.3f}]: {region.original_text}")
        retry_conf = (
            f"{region.retry_confidence:.3f}"
            if region.retry_confidence is not None
            else "нет оценки"
        )
        print(f"Повтор [{retry_conf}]: {region.retry_text}")
        while True:
            answer = input("Принять повторный вариант? [y/N]: ").strip().casefold()
            if answer in {"y", "yes", "д", "да"}:
                set_review_decision(region, True)
                break
            if answer in {"", "n", "no", "н", "нет"}:
                set_review_decision(region, False)
                break


def _save_report(
    path: Path,
    source_report: dict[str, Any],
    regions: list[RegionQC],
    elapsed_seconds: float,
    confidence_threshold: float,
    upscale_factor: float,
) -> None:
    payload = {
        "source_path": source_report.get("source_path"),
        "source_sha256": source_report.get("source_sha256"),
        "quality_policy": {
            "confidence_threshold": confidence_threshold,
            "upscale_factor": upscale_factor,
            "notes": "Confidence является сигналом для проверки, не гарантией точности.",
        },
        "regions": [asdict(region) for region in regions],
        "flagged_region_count": sum(bool(region.flags) for region in regions),
        "retry_count": sum(region.retry_text is not None for region in regions),
        "elapsed_seconds": elapsed_seconds,
    }
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_quality_control(
    report_path: str | Path,
    image_path: str | Path | None = None,
    output_path: str | Path | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    upscale_factor: float = DEFAULT_UPSCALE_FACTOR,
    review: bool = False,
    recognizer_factory: Callable[[], RegionRecognizer] | None = None,
) -> tuple[list[RegionQC], float, Path]:
    """Load a saved OCR report, retry only flagged regions and save a sidecar."""
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    source_image = Path(image_path or report["source_path"]).expanduser().resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {source_image}")

    started = time.perf_counter()
    regions = analyze_report(report, confidence_threshold)
    flagged = [region for region in regions if region.flags and region.coordinates]
    if flagged:
        if recognizer_factory is None:
            from ocr_prototype import PaddleOCREngine

            recognizer_factory = PaddleOCREngine
        recognizer = recognizer_factory()
        for region in flagged:
            retry_region(source_image, region, recognizer, upscale_factor)
    if review:
        _interactive_review(regions)

    elapsed = time.perf_counter() - started
    target = (
        Path(output_path)
        if output_path
        else report_file.with_name(f"{report_file.stem}.quality.json")
    )
    _save_report(target, report, regions, elapsed, confidence_threshold, upscale_factor)
    return regions, elapsed, target.expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ocr_report", type=Path, help="Сохранённый результат ocr_prototype.py")
    parser.add_argument("--image", type=Path, help="Исходное изображение, если путь в JSON устарел")
    parser.add_argument("--output", type=Path, help="Отдельный JSON для QC-результатов")
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--scale", type=float, default=DEFAULT_UPSCALE_FACTOR)
    parser.add_argument("--review", action="store_true", help="Провести интерактивную проверку в консоли")
    args = parser.parse_args()

    try:
        regions, elapsed, target = run_quality_control(
            args.ocr_report,
            image_path=args.image,
            output_path=args.output,
            confidence_threshold=args.threshold,
            upscale_factor=args.scale,
            review=args.review,
        )
    except Exception as error:
        parser.error(f"Контроль качества не выполнен: {type(error).__name__}: {error}")

    for region in regions:
        if not region.flags:
            continue
        print(f"\nОбласть #{region.region_id} ({region.coordinates})")
        print(f"  Исходное [{region.original_confidence:.3f}]: {region.original_text}")
        print(f"  Флаги: {'; '.join(region.flags)}")
        if region.retry_text is not None:
            retry_conf = region.retry_confidence or 0.0
            print(f"  Повтор [{retry_conf:.3f}]: {region.retry_text}")
        elif region.retry_error:
            print(f"  Ошибка retry: {region.retry_error}")
    print(f"Проверено областей: {len(regions)}; помечено: {sum(bool(x.flags) for x in regions)}")
    print(f"Время QC и повторного OCR: {elapsed:.3f} с")
    print(f"QC-результат: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
