"""Small local PaddleOCR smoke-test module for a single image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep PaddleX model/cache writes inside this prototype workspace.
_PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault(
    "PADDLE_PDX_CACHE_HOME", str(_PROJECT_DIR / ".cache" / "paddlex")
)

import cv2
import paddle
from paddleocr import PaddleOCR


MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"
PIPELINE_VERSION = "prototype-v1"
REVIEW_AVERAGE_THRESHOLD = 0.85
REVIEW_LINE_THRESHOLD = 0.95


@dataclass(frozen=True)
class OCRLine:
    text: str
    confidence: float
    polygon: list[list[int]] | None


@dataclass(frozen=True)
class OCRDocument:
    source_path: Path
    image_size: tuple[int, int]
    source_sha256: str
    lines: tuple[OCRLine, ...]
    original_text: str
    corrected_text: str | None
    average_confidence: float | None
    manual_review_required: bool
    review_reasons: tuple[str, ...]
    cache_hit: bool = False


class PaddleOCREngine:
    """Load the multilingual PP-OCRv5 recognizer once and process one image."""

    def __init__(self) -> None:
        self._pipeline = PaddleOCR(
            text_recognition_model_name=MODEL_NAME,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    def recognize(self, image_path: str | Path) -> OCRDocument:
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Изображение не найдено: {source}")

        # Exercise the OpenCV wheel and reject files it cannot decode before OCR.
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV не смог прочитать изображение: {source}")

        predictions = self._pipeline.predict(str(source))
        if not predictions:
            return OCRDocument(
                source_path=source,
                image_size=(image.shape[1], image.shape[0]),
                source_sha256=_sha256(source),
                lines=(),
                original_text="",
                corrected_text=None,
                average_confidence=None,
                manual_review_required=True,
                review_reasons=("OCR не вернул результат",),
            )

        payload = predictions[0].json
        result: dict[str, Any] = payload.get("res", payload)
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polygons = result.get("rec_polys", [])

        lines: list[OCRLine] = []
        for index, text in enumerate(texts):
            normalized_text = str(text).strip()
            if not normalized_text:
                continue
            confidence = float(scores[index]) if index < len(scores) else 0.0
            polygon = None
            if index < len(polygons):
                polygon = [
                    [int(point[0]), int(point[1])] for point in polygons[index]
                ]
            lines.append(
                OCRLine(
                    text=normalized_text,
                    confidence=confidence,
                    polygon=polygon,
                )
            )

        average, reasons = _review_metrics(lines)

        return OCRDocument(
            source_path=source,
            image_size=(image.shape[1], image.shape[0]),
            source_sha256=_sha256(source),
            lines=tuple(lines),
            original_text="\n".join(line.text for line in lines),
            corrected_text=None,
            average_confidence=average,
            manual_review_required=bool(reasons),
            review_reasons=reasons,
        )


def _review_metrics(
    lines: list[OCRLine] | tuple[OCRLine, ...],
) -> tuple[float | None, tuple[str, ...]]:
    confidences = [line.confidence for line in lines]
    average = sum(confidences) / len(confidences) if confidences else None
    reasons: list[str] = []
    if not lines:
        reasons.append("Текст не обнаружен")
    if average is not None and average < REVIEW_AVERAGE_THRESHOLD:
        reasons.append(f"Средняя уверенность ниже {REVIEW_AVERAGE_THRESHOLD:.2f}")
    low_confidence_count = sum(
        confidence < REVIEW_LINE_THRESHOLD for confidence in confidences
    )
    if low_confidence_count:
        reasons.append(
            f"Строк с уверенностью ниже {REVIEW_LINE_THRESHOLD:.2f}: "
            f"{low_confidence_count}"
        )
    return average, tuple(reasons)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(source_sha256: str) -> Path:
    identity = f"{source_sha256}:{MODEL_NAME}:{PIPELINE_VERSION}".encode("utf-8")
    key = hashlib.sha256(identity).hexdigest()
    cache_dir = Path(os.environ["PADDLE_PDX_CACHE_HOME"]) / "ocr_results"
    return cache_dir / f"{key}.json"


def _read_cache(source: Path, source_sha256: str) -> OCRDocument | None:
    cache_file = _cache_path(source_sha256)
    if not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("source_sha256") != source_sha256
            or payload.get("model_name") != MODEL_NAME
            or payload.get("pipeline_version") != PIPELINE_VERSION
        ):
            return None
        lines = tuple(
            OCRLine(
                text=item["text"],
                confidence=float(item["confidence"]),
                polygon=item.get("polygon"),
            )
            for item in payload["lines"]
        )
        average, reasons = _review_metrics(lines)
        return OCRDocument(
            source_path=source,
            image_size=tuple(payload["image_size"]),
            source_sha256=source_sha256,
            lines=lines,
            original_text=payload["original_text"],
            corrected_text=payload.get("corrected_text"),
            average_confidence=average,
            manual_review_required=bool(reasons),
            review_reasons=reasons,
            cache_hit=True,
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(document: OCRDocument) -> None:
    cache_file = _cache_path(document.source_sha256)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "recognized_at_utc": datetime.now(timezone.utc).isoformat(),
        **asdict(document),
        "source_path": str(document.source_path),
    }
    temporary = cache_file.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(cache_file)


def recognize_once(image_path: str | Path, force: bool = False) -> OCRDocument:
    source = Path(image_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Изображение не найдено: {source}")
    source_sha256 = _sha256(source)
    cached = _read_cache(source, source_sha256)
    if cached is not None and not force:
        # Check the image still decodes even when its content hash has a cache hit.
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"OpenCV не смог прочитать изображение: {source}")
        cached = replace(
            cached,
            source_path=source,
            image_size=(image.shape[1], image.shape[0]),
        )
        _write_cache(cached)
        return cached

    document = PaddleOCREngine().recognize(source)
    if cached is not None and cached.corrected_text is not None:
        document = replace(document, corrected_text=cached.corrected_text)
    _write_cache(document)
    return document


def print_document(document: OCRDocument) -> None:
    print(f"Изображение: {document.source_path}")
    print(f"Размер: {document.image_size[0]}x{document.image_size[1]}")
    print(f"OCR-модель: {MODEL_NAME}")
    print(f"Источник результата: {'кэш' if document.cache_hit else 'новое распознавание'}")
    print(f"SHA-256 файла: {document.source_sha256}")
    print("Распознанный текст:")
    for line in document.lines:
        if line.polygon:
            xs = [point[0] for point in line.polygon]
            ys = [point[1] for point in line.polygon]
            region = f"x={min(xs)}..{max(xs)}, y={min(ys)}..{max(ys)}"
        else:
            region = "координаты недоступны"
        print(f"  [{line.confidence:.3f}] ({region}) {line.text}")
    print("Оригинальный OCR-текст:")
    print(document.original_text or "<пусто>")
    print(f"Исправленный текст: {document.corrected_text!r}")
    confidence = (
        f"{document.average_confidence:.3f}"
        if document.average_confidence is not None
        else "нет данных"
    )
    print(f"Оценка уверенности (среднее OCR): {confidence}")
    review_status = "да" if document.manual_review_required else "нет"
    print(f"Требуется ручная проверка: {review_status}")
    if document.review_reasons:
        for reason in document.review_reasons:
            print(f"  Причина: {reason}")


def _save_manual_correction(document: OCRDocument, corrected_text: str) -> OCRDocument:
    updated = replace(document, corrected_text=corrected_text)
    _write_cache(updated)
    return updated


def _write_report(path: Path, document: OCRDocument, elapsed_seconds: float) -> None:
    report_path = path.expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(document)
    payload["source_path"] = str(document.source_path)
    payload["model_name"] = MODEL_NAME
    payload["processing_time_seconds"] = elapsed_seconds
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Путь к одному изображению")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Повторить OCR, даже если изображение и модель уже есть в кэше",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Открыть консольную проверку и сохранить исправленный текст отдельно",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Сохранить полный OCR-результат, координаты и время в JSON",
    )
    args = parser.parse_args()

    print(f"Python: {platform.python_version()} ({platform.machine()})")
    print(f"PaddlePaddle: {paddle.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"PaddleOCR: {__import__('paddleocr').__version__}")
    started_at = time.perf_counter()
    try:
        document = recognize_once(args.image, force=args.force)
    except Exception as error:  # show a useful failure in a prototype CLI
        print(f"Ошибка OCR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    elapsed_seconds = time.perf_counter() - started_at
    print_document(document)
    print(f"Время обработки: {elapsed_seconds:.3f} с")
    if args.result_json:
        _write_report(args.result_json, document, elapsed_seconds)
        print(f"Полный результат сохранён: {args.result_json.expanduser().resolve()}")
    if args.review:
        print("\nПроверка строк: Enter оставляет распознанный вариант.")
        reviewed_lines: list[str] = []
        changed = False
        for index, line in enumerate(document.lines, start=1):
            corrected_line = input(f"[{index}] {line.text}\n  Исправление: ")
            if corrected_line:
                reviewed_lines.append(corrected_line)
                changed = True
            else:
                reviewed_lines.append(line.text)
        if changed:
            document = _save_manual_correction(
                document, "\n".join(reviewed_lines)
            )
            print("Исправленный текст сохранён отдельно в локальном JSON-кэше.")
        else:
            print("Исправлений нет.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
