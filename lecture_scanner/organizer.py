"""Local, evidence-preserving lecture organizer without external AI services."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable


_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_STOP = {"это", "как", "для", "или", "что", "the", "and", "with", "from", "при", "это", "также", "более"}


@dataclass(frozen=True)
class SourceSlide:
    filename: str
    text: str


@dataclass(frozen=True)
class Topic:
    title: str
    slides: tuple[SourceSlide, ...]


def _terms(text: str) -> set[str]:
    return {word.casefold() for word in _WORD.findall(text) if word.casefold() not in _STOP}


def _title(slide: SourceSlide) -> str:
    first = next((line.strip() for line in slide.text.splitlines() if line.strip()), "")
    return first[:120] or slide.filename


def organize(slides: Iterable[SourceSlide]) -> tuple[Topic, ...]:
    """Keep slide order, group adjacent related material, retain every slide.

    Titles are extracted from OCR text only. No summaries, claims, or text are
    generated, so each exported paragraph remains traceable to its photo.
    """
    groups: list[list[SourceSlide]] = []
    group_terms: list[set[str]] = []
    for slide in slides:
        terms = _terms(slide.text)
        if groups and terms and len(terms & group_terms[-1]) >= 2:
            groups[-1].append(slide)
            group_terms[-1].update(terms)
        else:
            groups.append([slide])
            group_terms.append(set(terms))
    return tuple(Topic(_title(group[0]), tuple(group)) for group in groups)


def to_markdown(topics: Iterable[Topic]) -> str:
    sections = ["# Конспект лекции"]
    for index, topic in enumerate(topics, start=1):
        sections.append(f"\n## {index}. {topic.title}")
        for slide in topic.slides:
            sections.append(f"\n### Источник: {Path(slide.filename).name}\n\n{slide.text.strip()}")
    return "\n".join(sections).strip() + "\n"


def to_text(topics: Iterable[Topic]) -> str:
    blocks: list[str] = []
    for index, topic in enumerate(topics, start=1):
        blocks.append(f"{index}. {topic.title}")
        for slide in topic.slides:
            blocks.append(f"[{Path(slide.filename).name}]\n{slide.text.strip()}")
    return "\n\n".join(blocks).strip() + "\n"
