from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Split on sentence-ending punctuation followed by whitespace,
        # keeping the punctuation attached to the preceding sentence.
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        chunks: list[str] = []
        n = self.max_sentences_per_chunk
        for i in range(0, len(sentences), n):
            group = sentences[i : i + n]
            chunks.append(" ".join(group))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        seps = self.separators if self.separators else [""]
        return self._split(text, seps)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        # Base case: text fits in one chunk
        if len(current_text) <= self.chunk_size:
            return [current_text] if current_text else []

        # No separators left — hard cut by character
        if not remaining_separators:
            results = []
            for i in range(0, len(current_text), self.chunk_size):
                piece = current_text[i : i + self.chunk_size]
                if piece:
                    results.append(piece)
            return results

        sep = remaining_separators[0]
        rest = remaining_separators[1:]

        # Empty string separator — hard cut fallback
        if sep == "":
            return self._split(current_text, rest) if rest else [
                current_text[i : i + self.chunk_size]
                for i in range(0, len(current_text), self.chunk_size)
                if current_text[i : i + self.chunk_size]
            ]

        parts = current_text.split(sep)
        if len(parts) == 1:
            # Separator not found — try next one
            return self._split(current_text, rest)

        chunks: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(part) <= self.chunk_size:
                chunks.append(part)
            else:
                # Part too large — recurse with remaining separators
                chunks.extend(self._split(part, rest))
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    norm_a = math.sqrt(_dot(vec_a, vec_a))
    norm_b = math.sqrt(_dot(vec_b, vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (norm_a * norm_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0).chunk(text),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3).chunk(text),
            "recursive": RecursiveChunker(chunk_size=chunk_size).chunk(text),
        }

        result = {}
        for name, chunks in strategies.items():
            lengths = [len(c) for c in chunks]
            result[name] = {
                "count": len(chunks),
                "avg_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
                "chunks": chunks,
            }
        return result
