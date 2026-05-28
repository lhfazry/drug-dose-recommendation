"""Mock clinical guideline document store providing ~60 synthetic clinical documents
for retrieval-augmented generation in a drug dosage recommendation pipeline.

Documents span five cohorts (hypertension, diabetes_mellitus, oncology,
renal_impairment, general) and four categories (guideline, case_study,
drug_info, evidence), each containing 200-500 words of realistic clinical
guideline text with specific drug names, dosages, lab values, and references
to actual guidelines (AHA/ACC, ADA, NCCN, KDIGO).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ClinicalDocument:
    """A single mock clinical guideline document for retrieval."""

    doc_id: str
    title: str
    content: str
    source: str
    category: str
    cohort: str
    drug_names: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentStore:
    """In-memory store for ClinicalDocument objects with retrieval methods."""

    def __init__(self, documents: list[ClinicalDocument]) -> None:
        self._documents: dict[str, ClinicalDocument] = {
            doc.doc_id: doc for doc in documents
        }
        self._by_cohort: dict[str, list[ClinicalDocument]] = {}
        self._by_category: dict[str, list[ClinicalDocument]] = {}
        for doc in documents:
            self._by_cohort.setdefault(doc.cohort, []).append(doc)
            self._by_category.setdefault(doc.category, []).append(doc)

    def __len__(self) -> int:
        return len(self._documents)

    def get_by_id(self, doc_id: str) -> ClinicalDocument:
        return self._documents[doc_id]

    def get_by_cohort(self, cohort: str) -> list[ClinicalDocument]:
        return self._by_cohort.get(cohort, [])

    def get_by_category(self, category: str) -> list[ClinicalDocument]:
        return self._by_category.get(category, [])

    def search_by_keyword(self, keyword: str) -> list[ClinicalDocument]:
        kw = keyword.lower()
        return [
            doc
            for doc in self._documents.values()
            if kw in doc.content.lower() or kw in doc.title.lower()
        ]

    def get_all_documents(self) -> list[ClinicalDocument]:
        return list(self._documents.values())

    def get_document_texts(self) -> list[str]:
        return [doc.content for doc in self._documents.values()]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "content": doc.content,
                "source": doc.source,
                "category": doc.category,
                "cohort": doc.cohort,
                "drug_names": doc.drug_names,
                "metadata": doc.metadata,
            }
            for doc in self._documents.values()
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dicts(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DocumentStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        docs = [ClinicalDocument(**item) for item in data]
        return cls(docs)


_JSON_PATH = Path(__file__).parent / "data" / "clinical_documents.json"


def _load_documents_from_json() -> list[ClinicalDocument]:
    data = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    return [ClinicalDocument(**item) for item in data]


def _htn_documents() -> list[ClinicalDocument]:
    return [d for d in _load_documents_from_json() if d.cohort == "hypertension"]


def _dm_documents() -> list[ClinicalDocument]:
    return [d for d in _load_documents_from_json() if d.cohort == "diabetes_mellitus"]


def _onc_documents() -> list[ClinicalDocument]:
    return [d for d in _load_documents_from_json() if d.cohort == "oncology"]


def _ren_documents() -> list[ClinicalDocument]:
    return [d for d in _load_documents_from_json() if d.cohort == "renal_impairment"]


def _gen_documents() -> list[ClinicalDocument]:
    return [d for d in _load_documents_from_json() if d.cohort == "general"]


def build_default_store() -> DocumentStore:
    """Create a DocumentStore populated with 60 mock clinical guideline documents.

    Returns a DocumentStore containing 15 hypertension, 15 diabetes mellitus,
    15 oncology, 10 renal impairment, and 5 general clinical documents spanning
    guideline summaries, case studies, drug information, and evidence reviews.
    """
    return DocumentStore(_load_documents_from_json())
