"""Per-photo OCR fallbacks that preserve every attempted variant."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import cv2

from main_screen_detector import ScreenCandidate, detect_screens, rectify_screen


MIN_TEXT_CHARACTERS = 30
MIN_TEXT_LINES = 2
MIN_AVERAGE_CONFIDENCE = 0.55


class Recognizer(Protocol):
    def recognize(self, image_path: str | Path) -> Any: ...


@dataclass(frozen=True)
class OCRVariant:
    method: str
    image_path: Path
    report_path: Path
    original_text: str
    line_count: int
    average_confidence: float | None
    manual_review_required: bool
    quality_usable: bool
    quality_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProcessingOutcome:
    source_path: Path
    primary: OCRVariant
    variants: tuple[OCRVariant, ...]
    warnings: tuple[str, ...]
    screen_candidate: ScreenCandidate | None
    manifest_path: Path

    @property
    def quality_status(self) -> str:
        return "usable" if self.primary.quality_usable else "needs_retry"


def _assess_quality(document: Any) -> tuple[bool, tuple[str, ...]]:
    lines = tuple(getattr(document, "lines", ()) or ())
    text = str(getattr(document, "original_text", "") or "").strip()
    average = getattr(document, "average_confidence", None)
    reasons: list[str] = []
    if len(lines) < MIN_TEXT_LINES:
        reasons.append(f"Найдено строк: {len(lines)}; минимум: {MIN_TEXT_LINES}")
    if len(text) < MIN_TEXT_CHARACTERS:
        reasons.append(
            f"Распознано символов: {len(text)}; минимум: {MIN_TEXT_CHARACTERS}"
        )
    if average is None:
        reasons.append("OCR не вернул среднюю confidence")
    elif float(average) < MIN_AVERAGE_CONFIDENCE:
        reasons.append(
            f"Средняя confidence {float(average):.3f} ниже {MIN_AVERAGE_CONFIDENCE:.2f}"
        )
    return not reasons, tuple(reasons)


def _write_document_report(
    document: Any,
    report_path: Path,
    *,
    source_photo: Path,
    source_sha256: str,
    method: str,
) -> None:
    payload = asdict(document)
    payload.update(
        {
            "source_path": str(getattr(document, "source_path", report_path)),
            "original_photo_path": str(source_photo),
            "source_photo_sha256": source_sha256,
            "processing_method": method,
            "model_name": "eslav_PP-OCRv5_mobile_rec",
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _run_variant(
    recognizer: Recognizer,
    image_path: Path,
    report_path: Path,
    *,
    method: str,
    source_photo: Path,
    source_sha256: str,
) -> OCRVariant:
    document = recognizer.recognize(image_path)
    _write_document_report(
        document,
        report_path,
        source_photo=source_photo,
        source_sha256=source_sha256,
        method=method,
    )
    usable, reasons = _assess_quality(document)
    return OCRVariant(
        method=method,
        image_path=image_path,
        report_path=report_path,
        original_text=str(getattr(document, "original_text", "") or ""),
        line_count=len(tuple(getattr(document, "lines", ()) or ())),
        average_confidence=(
            float(getattr(document, "average_confidence"))
            if getattr(document, "average_confidence", None) is not None
            else None
        ),
        manual_review_required=bool(
            getattr(document, "manual_review_required", False)
        ),
        quality_usable=usable,
        quality_reasons=reasons,
    )


def _select_primary(variants: list[OCRVariant]) -> OCRVariant:
    usable = [variant for variant in variants if variant.quality_usable]
    candidates = usable or variants
    # A screen crop has priority only after it passes the baseline quality gate.
    return max(
        candidates,
        key=lambda variant: (
            variant.quality_usable,
            variant.method == "main_screen",
            variant.line_count,
            len(variant.original_text),
            variant.average_confidence or 0.0,
        ),
    )


def process_photo(
    source_path: str | Path,
    *,
    source_sha256: str,
    results_dir: str | Path,
    recognizer: Recognizer,
    screen_detector: Callable[[Any], list[ScreenCandidate]] = detect_screens,
    screen_rectifier: Callable[[Any, ScreenCandidate], Any] = rectify_screen,
) -> ProcessingOutcome:
    """OCR a photo even if screen detection produces no reliable candidate.

    Derived images and reports are stored beneath ``results_dir``. The source
    photo is only read. Every attempted OCR variant has a separate immutable
    report, and the manifest records why a particular variant was selected.
    """
    source = Path(source_path).expanduser().resolve()
    output = Path(results_dir).expanduser().resolve()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV не смог прочитать фотографию: {source}")

    output.mkdir(parents=True, exist_ok=True)
    derived_dir = output / "derived"
    warnings: list[str] = []
    variants: list[OCRVariant] = []
    candidates = screen_detector(image)
    reliable = [candidate for candidate in candidates if candidate.area_ratio >= 0.08]
    selected = max(reliable, key=lambda candidate: candidate.area_px) if reliable else None

    if selected is not None:
        rectified_path = derived_dir / f"{source_sha256}.main-screen.png"
        rectified_path.parent.mkdir(parents=True, exist_ok=True)
        rectified = screen_rectifier(image, selected)
        if not cv2.imwrite(str(rectified_path), rectified):
            raise OSError("Не удалось сохранить выровненный главный экран")
        variants.append(
            _run_variant(
                recognizer,
                rectified_path,
                output / f"{source_sha256}.main-screen.ocr.json",
                method="main_screen",
                source_photo=source,
                source_sha256=source_sha256,
            )
        )
    else:
        if candidates:
            warnings.append(
                "Экран-кандидат найден, но его площадь меньше порога надёжности; "
                "сохранены OCR предполагаемой области и всей фотографии."
            )
            uncertain = max(candidates, key=lambda candidate: candidate.area_px)
            rectified_path = derived_dir / f"{source_sha256}.uncertain-screen.png"
            rectified_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                rectified = screen_rectifier(image, uncertain, min_area_ratio=0.025)
                if not cv2.imwrite(str(rectified_path), rectified):
                    raise OSError("Не удалось сохранить предполагаемую область экрана")
                variants.append(
                    _run_variant(
                        recognizer,
                        rectified_path,
                        output / f"{source_sha256}.uncertain-screen.ocr.json",
                        method="uncertain_screen",
                        source_photo=source,
                        source_sha256=source_sha256,
                    )
                )
            except Exception as error:
                warnings.append(
                    "Предполагаемая область экрана не обработана: "
                    f"{type(error).__name__}: {error}"
                )
        else:
            warnings.append(
                "Главный экран не обнаружен; выполнен OCR всей фотографии без обрезки."
            )

        variants.append(
            _run_variant(
                recognizer,
                source,
                output / f"{source_sha256}.full-photo.ocr.json",
                method="full_image",
                source_photo=source,
                source_sha256=source_sha256,
            )
        )

    if selected is not None and not any(variant.quality_usable for variant in variants):
        warnings.append(
            "OCR главного экрана содержит слишком мало текста; выполнен OCR всей фотографии без обрезки."
        )
        variants.append(
            _run_variant(
                recognizer,
                source,
                output / f"{source_sha256}.full-photo-after-screen.ocr.json",
                method="full_image_after_low_screen",
                source_photo=source,
                source_sha256=source_sha256,
            )
        )

    if not any(variant.quality_usable for variant in variants):
        full_path = derived_dir / f"{source_sha256}.full-photo-upscaled.png"
        full_path.parent.mkdir(parents=True, exist_ok=True)
        upscaled = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        if not cv2.imwrite(str(full_path), upscaled):
            raise OSError("Не удалось сохранить увеличенную альтернативу")
        warnings.append("Основной OCR содержит слишком мало текста; выполнен OCR увеличенной фотографии.")
        variants.append(
            _run_variant(
                recognizer,
                full_path,
                output / f"{source_sha256}.full-photo-upscaled.ocr.json",
                method="full_image_upscaled_2x",
                source_photo=source,
                source_sha256=source_sha256,
            )
        )

    primary = _select_primary(variants)
    if not primary.quality_usable:
        warnings.append(
            "Ни один OCR-вариант не прошёл минимальную проверку качества; "
            "фотография сохранена для повторной обработки."
        )

    manifest_path = output / f"{source_sha256}.attempts.json"
    manifest = {
        "schema_version": 1,
        "source_photo_path": str(source),
        "source_photo_sha256": source_sha256,
        "screen_detection": {
            "candidate_count": len(candidates),
            "selected_candidate": asdict(selected) if selected is not None else None,
        },
        "warnings": warnings,
        "quality_status": "usable" if primary.quality_usable else "needs_retry",
        "primary_method": primary.method,
        "primary_report_path": str(primary.report_path),
        "variants": [
            {
                "method": variant.method,
                "image_path": str(variant.image_path),
                "report_path": str(variant.report_path),
                "line_count": variant.line_count,
                "text_characters": len(variant.original_text),
                "average_confidence": variant.average_confidence,
                "manual_review_required": variant.manual_review_required,
                "quality_usable": variant.quality_usable,
                "quality_reasons": list(variant.quality_reasons),
            }
            for variant in variants
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return ProcessingOutcome(
        source_path=source,
        primary=primary,
        variants=tuple(variants),
        warnings=tuple(warnings),
        screen_candidate=selected,
        manifest_path=manifest_path,
    )
