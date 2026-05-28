"""Hybrid retriever combining BM25 lexical search with dense embedding similarity."""

from __future__ import annotations

import re
from typing import Union

import numpy as np

from drug_dose.rag.embeddings import DocumentEmbedder, FAISSIndex

try:
    from rank_bm25 import BM25Okapi
    _HAS_RANK_BM25 = True
except ImportError:
    _HAS_RANK_BM25 = False


_DRUG_NAMES = {
    "metformin", "lisinopril", "amlodipine", "hydrochlorothiazide",
    "atorvastatin", "simvastatin", "warfarin", "apixaban", "rivaroxaban",
    "dabigatran", "clopidogrel", "aspirin", "ibuprofen", "naproxen",
    "acetaminophen", "paracetamol", "morphine", "oxycodone", "fentanyl",
    "gabapentin", "pregabalin", "insulin", "glipizide", "glyburide",
    "levothyroxine", "prednisone", "methotrexate", "cyclophosphamide",
    "doxorubicin", "cisplatin", "carboplatin", "paclitaxel",
    "fluorouracil", "capecitabine", "tamoxifen", "anastrozole",
    "enalapril", "losartan", "valsartan", "carvedilol", "metoprolol",
    "atenolol", "propranolol", "digoxin", "amiodarone", "furosemide",
    "spironolactone", "omeprazole", "pantoprazole", "ondansetron",
}

_CLINICAL_SYNONYMS: dict[str, list[str]] = {
    "high blood pressure": ["hypertension", "elevated BP", "HTN"],
    "hypertension": ["high blood pressure", "elevated BP", "HTN"],
    "diabetes": ["diabetes mellitus", "DM", "type 2 diabetes", "T2DM"],
    "kidney disease": ["renal impairment", "CKD", "chronic kidney disease", "nephropathy"],
    "heart failure": ["CHF", "congestive heart failure", "cardiac failure"],
    "atrial fibrillation": ["AFib", "AF", "arrhythmia"],
    "cancer": ["malignancy", "neoplasm", "tumor"],
    "pain": ["analgesia", "nociception"],
    "infection": ["sepsis", "bacterial", "antimicrobial"],
    "seizure": ["epilepsy", "convulsion"],
    "depression": ["MDD", "major depressive disorder"],
    "asthma": ["COPD", "bronchospasm"],
    "liver disease": ["hepatic impairment", "cirrhosis", "hepatitis"],
    "elderly": ["geriatric", "older adults", "aged"],
    "pregnancy": ["gestation", "prenatal", "obstetric"],
    "pediatric": ["children", "neonates", "infants"],
    "renal impairment": ["kidney disease", "CKD", "renal dysfunction", "reduced GFR"],
    "obesity": ["overweight", "high BMI"],
    "contraindicated": ["avoid", "do not use", "not recommended"],
    "adverse reaction": ["ADR", "side effect", "toxicity"],
}


class _BM25Scratch:
    """BM25 from scratch. Uses the canonical Okapi BM25 formula."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._corpus: list[list[str]] = []
        self._doc_ids: list[str] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._doc_len: list[int] = []
        self._N: int = 0

    def index(self, tokenized_corpus: list[list[str]], doc_ids: list[str]) -> None:
        self._corpus = tokenized_corpus
        self._doc_ids = list(doc_ids)
        self._N = len(tokenized_corpus)
        self._doc_len = [len(doc) for doc in tokenized_corpus]
        self._avgdl = np.mean(self._doc_len) if self._doc_len else 0.0

        df: dict[str, int] = {}
        for doc in tokenized_corpus:
            seen = set()
            for term in doc:
                if term not in seen:
                    df[term] = df.get(term, 0) + 1
                    seen.add(term)

        self._idf = {}
        for term, doc_freq in df.items():
            self._idf[term] = np.log((self._N - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def get_scores(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self._N, dtype=np.float64)
        for term in query_tokens:
            term_idf = self._idf.get(term, 0.0)
            if term_idf == 0.0:
                continue
            for i, doc in enumerate(self._corpus):
                tf = doc.count(term)
                if tf == 0:
                    continue
                dl = self._doc_len[i]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / (self._avgdl + 1e-8))
                scores[i] += term_idf * numerator / denominator
        return scores

    def get_top_n(self, query_tokens: list[str], n: int = 10) -> tuple[list[str], list[float]]:
        scores = self.get_scores(query_tokens)
        if self._N == 0 or n <= 0:
            return [], []
        n = min(n, self._N)
        top_indices = np.argpartition(scores, -n)[-n:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._doc_ids[i] for i in top_indices], [float(scores[i]) for i in top_indices]


class _BM25Wrapper:
    """Unified BM25 interface that tries rank_bm25 first, falls back to scratch."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._scratch = _BM25Scratch(k1=k1, b=b)
        self._bm25: BM25Okapi | None = None
        self._doc_ids: list[str] = []
        self._corpus: list[list[str]] = []
        self._tokenizer = _simple_tokenize

    def index(self, texts: list[str], doc_ids: list[str]) -> None:
        corpus = [self._tokenizer(t) for t in texts]
        self._corpus = corpus
        self._doc_ids = list(doc_ids)

        if _HAS_RANK_BM25:
            self._bm25 = BM25Okapi(corpus)
        else:
            self._scratch.index(corpus, doc_ids)

    def get_top_n(self, query: str, n: int = 10) -> tuple[list[str], list[float]]:
        tokens = self._tokenizer(query)
        if _HAS_RANK_BM25 and self._bm25 is not None:
            scores = self._bm25.get_scores(tokens)
            if len(scores) == 0:
                return [], []
            n = min(n, len(scores))
            top_indices = np.argpartition(scores, -n)[-n:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            return (
                [self._doc_ids[i] for i in top_indices],
                [float(scores[i]) for i in top_indices],
            )

        return self._scratch.get_top_n(tokens, n=n)


def _simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def _minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [0.5] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


class HybridRetriever:
    """Hybrid retrieval combining BM25 keyword search with dense embeddings.

    Per the paper: ψ(r, D) uses BM25 for lexical matching alongside cosine
    distance in a dense vector space as a hybrid model incorporating both
    lexical and semantic aspects.
    """

    def __init__(
        self,
        embedder: DocumentEmbedder,
        faiss_index: FAISSIndex,
        documents: list,
        bm25_weight: float = 0.3,
    ) -> None:
        self.embedder = embedder
        self.faiss_index = faiss_index
        self._documents = documents
        self.bm25_weight = bm25_weight
        self._doc_by_id: dict[str, dict] = {}
        self._bm25 = _BM25Wrapper()
        self._indexed = False

    def index_documents(self, texts: list[str], doc_ids: list[str]) -> None:
        embeddings = self.embedder.embed_documents(texts)
        self.faiss_index.build(embeddings, doc_ids)
        self._bm25.index(texts, doc_ids)

        for i, doc_id in enumerate(doc_ids):
            doc_dict = None
            if i < len(self._documents) and hasattr(self._documents[i], "to_dict"):
                doc_dict = self._documents[i].to_dict()
            elif i < len(self._documents) and isinstance(self._documents[i], dict):
                doc_dict = dict(self._documents[i])
            elif i < len(self._documents):
                doc_dict = {
                    "doc_id": doc_id,
                    "title": getattr(self._documents[i], "title", ""),
                    "content": getattr(self._documents[i], "content", ""),
                    "source": getattr(self._documents[i], "source", ""),
                    "category": getattr(self._documents[i], "category", ""),
                    "cohort": getattr(self._documents[i], "cohort", ""),
                }
            if doc_dict:
                self._doc_by_id[doc_id] = doc_dict

        self._indexed = True

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        if not self._indexed:
            raise RuntimeError("Must call index_documents() before retrieve()")

        query_emb = self.embedder.embed_query(query)

        dense_k = min(k * 2, len(self.faiss_index))
        dense_ids, dense_scores = self.faiss_index.search(query_emb, k=dense_k)
        dense_ids_flat = dense_ids[0] if dense_ids else []
        dense_scores_flat = dense_scores[0] if dense_scores else []

        bm25_ids, bm25_scores = self._bm25.get_top_n(query, n=dense_k)

        norm_dense = _minmax_normalize(dense_scores_flat)
        norm_bm25 = _minmax_normalize(bm25_scores)

        fused: dict[str, dict] = {}

        for idx, doc_id in enumerate(dense_ids_flat):
            score = (1.0 - self.bm25_weight) * norm_dense[idx] if idx < len(norm_dense) else 0.0
            fused.setdefault(doc_id, {})["dense_score"] = dense_scores_flat[idx] if idx < len(dense_scores_flat) else 0.0
            fused.setdefault(doc_id, {})["bm25_score"] = 0.0
            fused.setdefault(doc_id, {})["fused_score"] = score

        for idx, doc_id in enumerate(bm25_ids):
            score = self.bm25_weight * norm_bm25[idx] if idx < len(norm_bm25) else 0.0
            entry = fused.setdefault(doc_id, {})
            entry["bm25_score"] = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            entry.setdefault("dense_score", 0.0)
            entry["fused_score"] = entry.get("fused_score", 0.0) + score

        ranked = sorted(fused.items(), key=lambda x: x[1]["fused_score"], reverse=True)
        ranked = ranked[:k]

        results: list[dict] = []
        for rank_idx, (doc_id, scores) in enumerate(ranked):
            doc_info = self._doc_by_id.get(doc_id, {})
            results.append({
                "doc_id": doc_id,
                "title": doc_info.get("title", ""),
                "content": doc_info.get("content", ""),
                "source": doc_info.get("source", ""),
                "category": doc_info.get("category", ""),
                "cohort": doc_info.get("cohort", ""),
                "bm25_score": scores.get("bm25_score", 0.0),
                "dense_score": scores.get("dense_score", 0.0),
                "fused_score": scores.get("fused_score", 0.0),
                "rank": rank_idx + 1,
            })

        return results


def transform_query(query: str, feedback: str | None = None) -> str:
    """Query transformation φ(qu, f): tokenization, normalization, synonym
    expansion, drug-name detection, and feedback incorporation."""

    result_parts = [query]

    query_lower = query.lower()

    for pattern, synonyms in _CLINICAL_SYNONYMS.items():
        if pattern in query_lower:
            for syn in synonyms:
                if syn.lower() not in query_lower:
                    result_parts.append(syn)

    drug_hits: set[str] = set()
    for drug in _DRUG_NAMES:
        pat = re.compile(rf"\b{drug}\b", re.IGNORECASE)
        if pat.search(query):
            drug_hits.add(drug.lower())
    if drug_hits:
        result_parts.append("drugs involved: " + ", ".join(sorted(drug_hits)))

    lab_value_pattern = re.compile(
        r"\b(GFR|creatinine|HbA1c|ALT|AST|WBC|platelet|BP|blood pressure|sodium|potassium|INR|PTT|TSH|T4|hemoglobin)\b",
        re.IGNORECASE,
    )
    lab_hits = set(m.group(1).lower() for m in lab_value_pattern.finditer(query))
    if lab_hits:
        result_parts.append("lab values: " + ", ".join(sorted(lab_hits)))

    if feedback:
        result_parts.append(f"relevant feedback: {feedback.strip()}")

    return " ".join(result_parts)
