"""RAG pipeline for drug dosage recommendation — document store, retrieval, and generation."""

try:
    from drug_dose.rag.document_store import (
        ClinicalDocument,
        DocumentStore,
        build_default_store,
    )
except ImportError:
    ClinicalDocument = None  # type: ignore[assignment]
    DocumentStore = None  # type: ignore[assignment]

    def build_default_store():
        raise NotImplementedError("document_store module not yet available")


from drug_dose.rag.embeddings import DocumentEmbedder, FAISSIndex
from drug_dose.rag.retriever import HybridRetriever, transform_query
from drug_dose.rag.rag_pipeline import RAGPipeline

__all__ = [
    "ClinicalDocument",
    "DocumentStore",
    "build_default_store",
    "DocumentEmbedder",
    "FAISSIndex",
    "HybridRetriever",
    "transform_query",
    "RAGPipeline",
]
