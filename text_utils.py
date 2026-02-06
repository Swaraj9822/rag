import re
from typing import List


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _normalize_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _split_paragraphs(text: str) -> List[str]:
    normalized = _normalize_lines(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", normalized) if p.strip()]
    return paragraphs


def _split_oversized_unit(unit: str, chunk_size: int) -> List[str]:
    if len(unit) <= chunk_size:
        return [unit]

    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(unit) if s.strip()]
    if len(sentences) <= 1:
        return [unit[i : i + chunk_size] for i in range(0, len(unit), chunk_size)]

    packed: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            packed.append(current)
            current = sentence
        else:
            packed.extend([sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)])
            current = ""

    if current:
        packed.append(current)

    return packed


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    units: List[str] = []
    for paragraph in paragraphs:
        units.extend(_split_oversized_unit(paragraph, chunk_size))

    chunks: List[str] = []
    current = ""

    for unit in units:
        separator = "\n\n" if current else ""
        candidate = f"{current}{separator}{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if overlap and chunks:
            tail = chunks[-1][-overlap:].strip()
            if tail and len(tail) + 2 + len(unit) <= chunk_size:
                current = f"{tail}\n\n{unit}"
            else:
                current = unit
        else:
            current = unit

    if current:
        chunks.append(current)

    return chunks
