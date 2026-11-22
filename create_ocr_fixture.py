"""Generate a small photo-like Russian/English slide for the OCR smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial.ttf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "fixtures"
        / "bilingual_slide_sample.png",
    )
    output = parser.parse_args().output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"Не найден шрифт с кириллицей: {FONT_PATH}")

    slide = Image.new("RGB", (1280, 720), "#fbfaf6")
    draw = ImageDraw.Draw(slide)
    title_font = ImageFont.truetype(str(FONT_PATH), 56)
    body_font = ImageFont.truetype(str(FONT_PATH), 39)

    draw.text((90, 75), "Лекция: обработка изображений", fill="#163c65", font=title_font)
    draw.rounded_rectangle((92, 160, 1188, 168), radius=4, fill="#4c8fc7")
    draw.text((95, 235), "Русский текст распознаётся локально.", fill="#18232d", font=body_font)
    draw.text((95, 330), "English text is processed locally.", fill="#18232d", font=body_font)
    draw.text(
        (95, 425),
        "PaddleOCR handles both languages in one model.",
        fill="#18232d",
        font=body_font,
    )
    draw.text((95, 600), "Тестовый слайд / Smoke test", fill="#4b5966", font=body_font)

    slide_array = np.asarray(slide)
    height, width = slide_array.shape[:2]
    source_points = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    target_points = np.float32([[235, 105], [1370, 145], [1450, 865], [145, 825]])
    transform = cv2.getPerspectiveTransform(source_points, target_points)

    # Neutral wall and soft shadow make the generated fixture photo-like while
    # keeping every test string known and legible.
    background = np.full((1000, 1600, 3), (94, 99, 103), dtype=np.uint8)
    background[:, :, 0] = np.clip(
        background[:, :, 0].astype(np.int16)
        + np.linspace(8, -10, 1600, dtype=np.int16)[None, :],
        0,
        255,
    ).astype(np.uint8)
    warped = cv2.warpPerspective(
        slide_array,
        transform,
        (1600, 1000),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        transform,
        (1600, 1000),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    photo = (
        background.astype(np.float32) * (1.0 - alpha)
        + warped.astype(np.float32) * alpha
    ).astype(np.uint8)
    photo = cv2.GaussianBlur(photo, (3, 3), 0.35)
    if not cv2.imwrite(str(output), cv2.cvtColor(photo, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Не удалось записать изображение: {output}")
    print(f"Создан синтетический тестовый кадр: {output}")
    print("Это не фотография с камеры; кадр имитирует перспективу слайда.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
