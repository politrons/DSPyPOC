from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class RagChunk:
    chunk_id: str
    text: str
    source: str


class TfidfRagRetriever:
    """Small in-process TF-IDF retriever for RAG PoCs.

    This retriever avoids external vector databases. It is suitable for small
    to medium knowledge bases in a proof-of-concept setting.
    """

    def __init__(self, chunks: list[RagChunk]) -> None:
        if not chunks:
            raise ValueError("At least one chunk is required")
        self._chunks = chunks

        self._doc_tokens: list[list[str]] = [self._tokenize(c.text) for c in chunks]
        self._doc_tf: list[Counter[str]] = [Counter(tokens) for tokens in self._doc_tokens]
        self._doc_len: list[int] = [max(1, len(tokens)) for tokens in self._doc_tokens]

        doc_freq: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            doc_freq.update(set(tokens))

        total_docs = len(chunks)
        self._idf: dict[str, float] = {
            term: math.log((1 + total_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()
        }

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "TfidfRagRetriever":
        rows: list[RagChunk] = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            rows.append(
                RagChunk(
                    chunk_id=str(item.get("chunk_id", f"chunk-{len(rows)+1}")),
                    source=str(item.get("source", "knowledge_base")),
                    text=text,
                )
            )

        if not rows:
            raise ValueError(f"No chunks were found in {path}")
        return cls(rows)

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> list[dict[str, str | float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        query_tf = Counter(query_tokens)
        query_len = max(1, len(query_tokens))

        scored: list[tuple[float, int]] = []
        for idx, tf in enumerate(self._doc_tf):
            score = 0.0
            doc_len = self._doc_len[idx]

            for token, q_count in query_tf.items():
                idf = self._idf.get(token)
                if idf is None:
                    continue
                doc_count = tf.get(token, 0)
                if doc_count == 0:
                    continue

                q_weight = (q_count / query_len) * idf
                d_weight = (doc_count / doc_len) * idf
                score += q_weight * d_weight

            if score >= min_score:
                scored.append((score, idx))

        scored.sort(key=lambda item: item[0], reverse=True)

        out: list[dict[str, str | float]] = []
        for score, idx in scored[: max(0, top_k)]:
            chunk = self._chunks[idx]
            out.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": score,
                }
            )
        return out

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [tok.lower() for tok in _TOKEN_RE.findall(text)]
