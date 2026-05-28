"""Embedding generation and FAISS vector index for clinical guidelines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class DocumentEmbedder:
    """Encodes clinical text into dense embeddings using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode a batch of documents, returning L2-normalized embeddings."""
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query, returning an L2-normalized embedding."""
        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0].astype(np.float32)


class FAISSIndex:
    """Flat inner-product FAISS index for cosine-similarity search with
    L2-normalized embeddings."""

    def __init__(self) -> None:
        self.index: faiss.IndexFlatIP | None = None
        self._doc_ids: list[str] = []
        self._dim: int | None = None

    @property
    def dim(self) -> int | None:
        return self._dim

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal

    def build(self, embeddings: np.ndarray, doc_ids: list[str]) -> None:
        if len(embeddings) == 0:
            return
        self._dim = embeddings.shape[1]
        self._doc_ids = list(doc_ids)
        self.index = faiss.IndexFlatIP(self._dim)
        self.index.add(embeddings.astype(np.float32))

    def search(
        self, query_embedding: np.ndarray, k: int = 10
    ) -> tuple[list[list[str]], list[list[float]]]:
        if self.index is None or self.index.ntotal == 0:
            return [], []

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query, k)

        result_ids: list[list[str]] = []
        result_scores: list[list[float]] = []
        for row_idx, row_scores in zip(indices, scores):
            result_ids.append([self._doc_ids[i] for i in row_idx if i >= 0])
            result_scores.append([float(s) for s, i in zip(row_scores, row_idx) if i >= 0])

        return result_ids, result_scores

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index_path = str(path.with_suffix(".index"))
        meta_path = str(path.with_suffix(".json"))

        if self.index is not None:
            faiss.write_index(self.index, index_path)

        with open(meta_path, "w") as f:
            json.dump({"doc_ids": self._doc_ids, "dim": self._dim}, f)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        index_path = str(path.with_suffix(".index"))
        meta_path = str(path.with_suffix(".json"))

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        self.index = faiss.read_index(index_path)
        self._dim = self.index.d

        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self._doc_ids = meta.get("doc_ids", [])
