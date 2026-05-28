"""RAG pipeline orchestrating query transformation, retrieval, and knowledge extraction."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from drug_dose.rag.retriever import HybridRetriever, transform_query


_DRUG_LIST_RE = re.compile(
    r"\b(Metformin|Lisinopril|Amlodipine|Enalapril|Losartan|Valsartan|"
    r"Carvedilol|Metoprolol|Atenolol|Propranolol|Hydrochlorothiazide|"
    r"Warfarin|Apixaban|Rivaroxaban|Clopidogrel|Aspirin|"
    r"Ibuprofen|Naproxen|Acetaminophen|Paracetamol|Morphine|"
    r"Oxycodone|Fentanyl|Gabapentin|Pregabalin|Insulin|"
    r"Glipizide|Glyburide|Levothyroxine|Prednisone|Methotrexate|"
    r"Cyclophosphamide|Doxorubicin|Cisplatin|Carboplatin|Paclitaxel|"
    r"Fluorouracil|Capecitabine|Tamoxifen|Anastrozole|"
    r"Digoxin|Amiodarone|Furosemide|Spironolactone|Omeprazole|"
    r"Pantoprazole|Ondansetron|Atorvastatin|Simvastatin|"
    r"Empagliflozin|Dapagliflozin|Semaglutide|Liraglutide|Sitagliptin|"
    r"Labetalol|Nifedipine|Clonidine|Captopril|Chlorthalidone|"
    r"Pembrolizumab|Nivolumab|Imatinib|Filgrastim|Gentamicin|Vancomycin)\b",
    re.IGNORECASE,
)

_DOSAGE_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:mg|mcg|g|mg/kg|IU|units?)(?:/[k]?g)?)",
    re.IGNORECASE,
)

_CI_KEYWORDS = re.compile(
    r"\b(contraindicated|avoid|not recommended|do not (?:use|administer))\b",
    re.IGNORECASE,
)

_GUIDELINE_KEYWORDS = re.compile(
    r"\b(guidelines?\s+(?:recommend|suggest)|recommend(?:ed|s)?\s+(?:starting\s+)?dose|should\s+be\s+(?:dosed|titrated|started|reduced|adjusted))",
    re.IGNORECASE,
)

_MAX_SENTENCE = 160


def _extract_key_facts(documents: list[dict]) -> list[str]:
    facts: list[str] = []
    seen: set[str] = set()

    for doc in documents:
        doc_id = doc.get("doc_id", "unknown")
        content = doc.get("content", "")
        title = doc.get("title", "")
        combined = f"{title}. {content}"

        for m in _DRUG_LIST_RE.finditer(combined):
            start = max(0, m.start() - 30)
            end = min(len(combined), m.end() + 60)
            snippet = combined[start:end].replace("\n", " ").strip()
            if len(snippet) > _MAX_SENTENCE:
                snippet = snippet[:_MAX_SENTENCE] + "..."
            fact = f"[{doc_id}] Drug: {m.group(1)} — \"{snippet}\""
            _add_unique(facts, seen, fact)

        for m in _DOSAGE_RE.finditer(combined):
            fact = f"[{doc_id}] Dosage: {m.group(1).strip()}"
            _add_unique(facts, seen, fact)

        for m in _CI_KEYWORDS.finditer(combined):
            start = max(0, m.start() - 20)
            end = min(len(combined), m.end() + 120)
            snippet = combined[start:end].replace("\n", " ").strip()
            if len(snippet) > _MAX_SENTENCE:
                snippet = snippet[:_MAX_SENTENCE] + "..."
            fact = f"[{doc_id}] Warning: \"{snippet}\""
            _add_unique(facts, seen, fact)

        for m in _GUIDELINE_KEYWORDS.finditer(combined):
            start = m.start()
            end = min(len(combined), m.end() + 120)
            snippet = combined[start:end].replace("\n", " ").strip()
            if len(snippet) > _MAX_SENTENCE:
                snippet = snippet[:_MAX_SENTENCE] + "..."
            fact = f"[{doc_id}] Guideline: \"{snippet}\""
            _add_unique(facts, seen, fact)

    return facts


def _add_unique(facts: list[str], seen: set[str], fact: str) -> None:
    if fact not in seen:
        facts.append(fact)
        seen.add(fact)


def _extract_drug_mentions(documents: list[dict]) -> list[str]:
    drugs: set[str] = set()
    for doc in documents:
        content = doc.get("content", "")
        for m in _DRUG_LIST_RE.finditer(content):
            drugs.add(m.group(1).lower())
    return sorted(drugs)


def _extract_cohorts(documents: list[dict]) -> list[str]:
    cohorts: set[str] = set()
    for doc in documents:
        cohort = doc.get("cohort", "")
        if cohort:
            cohorts.add(cohort)
        category = doc.get("category", "")
        if category:
            cohorts.add(category)
    return sorted(cohorts)


class RAGPipeline:
    """End-to-end RAG pipeline implementing Algorithm 3 from the paper.

    Flow: query → φ(transform) → ψ(retrieve) → Search Output + Knowledge-Base Output
    """

    def __init__(self, retriever: HybridRetriever, document_store: Any) -> None:
        self.retriever = retriever
        self.document_store = document_store

    def process_query(
        self,
        query: str,
        feedback: str | None = None,
        top_k: int = 10,
    ) -> dict:
        transformed = transform_query(query, feedback)
        retrieved_docs = self.retriever.retrieve(transformed, k=top_k)

        search_output: list[dict] = []
        for doc in retrieved_docs:
            content = doc.get("content", "")
            search_output.append({
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "snippet": content[:200] if len(content) > 200 else content,
                "relevance": doc.get("fused_score", 0.0),
                "source": doc.get("source", ""),
            })

        top3_content = "\n\n".join(
            d.get("content", "")[:600] for d in retrieved_docs[:3]
        )

        key_facts = _extract_key_facts(retrieved_docs)
        drug_mentions = _extract_drug_mentions(retrieved_docs)
        relevant_cohorts = _extract_cohorts(retrieved_docs)

        evidence_levels: dict[str, str] = {}
        for doc in retrieved_docs:
            doc_id = doc.get("doc_id", "")
            content = doc.get("content", "")
            if "systematic review" in content.lower() or "meta-analysis" in content.lower():
                evidence_levels[doc_id] = "Level A"
            elif "RCT" in content or "randomized" in content.lower():
                evidence_levels[doc_id] = "Level A"
            elif "guideline" in content.lower():
                evidence_levels[doc_id] = "Level B"
            elif "expert" in content.lower() or "consensus" in content.lower():
                evidence_levels[doc_id] = "Level C"
            else:
                evidence_levels[doc_id] = "Level D"

        return {
            "query": query,
            "transformed_query": transformed,
            "search_output": search_output,
            "knowledge_base_output": {
                "summary": top3_content,
                "key_facts": key_facts[:15],
                "drug_mentions": drug_mentions,
                "relevant_cohorts": relevant_cohorts,
                "evidence_levels": evidence_levels,
            },
            "raw_documents": retrieved_docs,
        }
