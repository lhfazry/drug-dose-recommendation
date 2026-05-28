import sys

sys.path.insert(0, "src")

from drug_dose.rag import (
    DocumentEmbedder,
    DocumentStore,
    FAISSIndex,
    HybridRetriever,
    RAGPipeline,
    build_default_store,
    transform_query,
)


def main():
    print("=" * 60)
    print("PHASE 2: RAG Pipeline Demo")
    print("=" * 60)

    store = build_default_store()
    print(f"\n[1] Document Store: {len(store)} clinical documents")
    for cohort in ["hypertension", "diabetes_mellitus", "oncology", "renal_impairment", "general"]:
        count = len(store.get_by_cohort(cohort))
        print(f"    {cohort:25s} {count:3d} docs")

    embedder = DocumentEmbedder("all-MiniLM-L6-v2")
    print(f"\n[2] Embedding Model: {embedder.dim}-dimensional")

    docs_text = store.get_document_texts()
    doc_ids = [d.doc_id for d in store.get_all_documents()]
    embeddings = embedder.embed_documents(docs_text)
    index = FAISSIndex()
    index.build(embeddings, doc_ids)
    print(f"    FAISS index built: {len(index)} vectors")

    retriever = HybridRetriever(embedder, index, store.to_dicts())
    retriever.index_documents(docs_text, doc_ids)
    print(f"    Hybrid retriever ready (BM25 + dense)")

    pipeline = RAGPipeline(retriever, store)

    test_cases = [
        (
            "lisinopril starting dose for hypertension with CKD stage 3 and type 2 diabetes",
            "patient has GFR 38 and HbA1c 7.8%",
        ),
        (
            "metformin dosing in renal impairment",
            "patient has CKD stage 4, GFR 22",
        ),
        (
            "carboplatin dosing for lung cancer patient with reduced kidney function",
            "patient age 70, GFR 48, NSCLC stage IV",
        ),
    ]

    print(f"\n[3] Processing {len(test_cases)} clinical queries\n")

    for i, (query, feedback) in enumerate(test_cases, 1):
        transformed = transform_query(query, feedback)
        result = pipeline.process_query(query, feedback=feedback, top_k=5)

        print(f"  ┌─ Case {i}")
        print(f"  │ Query:    {query}")
        print(f"  │ Feedback: {feedback}")

        so = result["search_output"]
        kb = result["knowledge_base_output"]

        print(f"  ├─ Search Output ({len(so)} results):")
        for r in so[:3]:
            print(f"  │   #{r['doc_id']} [{r['relevance']:.3f}] {r['title'][:55]}")
            print(f"  │      Source: {r['source']}")

        print(f"  ├─ Knowledge-Base Output:")
        print(f"  │   Drugs:     {kb['drug_mentions']}")
        print(f"  │   Cohorts:   {kb['relevant_cohorts']}")
        print(f"  │   Key Facts ({len(kb['key_facts'])} total):")
        for fact in kb["key_facts"][:4]:
            print(f"  │     • {fact[:90]}")
        print(f"  └─")

    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE — Retrieval pipeline verified.")
    print("Output streams ready: Search Output + Knowledge-Base Output")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
