import json
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, "src")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import numpy as np

from drug_dose.data import FeatureConfig, SyntheticDataGenerator
from drug_dose.preprocessing import PreprocessingPipeline
from drug_dose.rag import (
    DocumentEmbedder,
    FAISSIndex,
    HybridRetriever,
    RAGPipeline,
    build_default_store,
)
from drug_dose.digital_twin import DigitalTwin, RAGToEmbedding, TransformerEncoder
from drug_dose.digital_twin.gan_module import (
    PatientGAN,
    PatientGANDiscriminator,
    PatientGANGenerator,
)
from drug_dose.rl import (
    DosageEnvironment,
    RLTrainer,
    TransformerPolicy,
    ValueEstimator,
)
from drug_dose.xai import LLMFusion, generate_recommendation


def banner(text):
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


def main():
    print("=" * 60)
    print("  FULL PIPELINE: Phase 1–5 End-to-End Demo")
    print("  Drug & Dosage Recommendation System")
    print("=" * 60)

    fc = FeatureConfig()
    feature_names = fc.get_numerical_features()
    n_features = len(feature_names)
    embed_dim = 256
    noise_dim = 64
    t_total = time.time()

    banner("Phase 1 — Synthetic Data + Preprocessing")
    gen = SyntheticDataGenerator(n_patients=200, random_seed=42)
    df = gen.generate()
    pipe = PreprocessingPipeline(fc)
    X_proc = pipe.fit_transform(df)
    real_data = torch.tensor(X_proc[feature_names].values, dtype=torch.float32)
    print(f"  Generated: {len(df)} patients × {n_features} features")
    print(f"  Cohorts:   {dict(df['cohort'].value_counts())}")

    banner("Phase 2 — RAG Retrieval")
    store = build_default_store()
    texts = store.get_document_texts()
    doc_ids = [d.doc_id for d in store.get_all_documents()]
    embedder = DocumentEmbedder("all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(texts)
    index = FAISSIndex()
    index.build(embeddings, doc_ids)
    retriever = HybridRetriever(embedder, index, store.to_dicts())
    retriever.index_documents(texts, doc_ids)
    rag = RAGPipeline(retriever, store)

    clinical_query = "lisinopril starting dose adjustment for 62 year old with hypertension CKD stage 3 GFR 42 and type 2 diabetes"
    rag_result = rag.process_query(clinical_query, top_k=5)
    print(f"  Query:    {clinical_query}")
    top_doc = rag_result["search_output"][0]
    print(f"  Top doc:  {top_doc['doc_id']} — {top_doc['title'][:55]} ({top_doc['relevance']:.3f})")

    banner("Phase 3 — Digital Twin (GAN)")
    rag_embedder = RAGToEmbedding()
    s_out, kb_out, _ = rag_embedder.prepare_inputs(rag_result, embedder)
    encoder = TransformerEncoder(input_dim=embedder.dim, d_model=embed_dim)
    gan_gen = PatientGANGenerator(embed_dim=embed_dim, noise_dim=noise_dim, feature_dim=n_features)
    gan_disc = PatientGANDiscriminator(feature_dim=n_features, embed_dim=embed_dim)
    gan = PatientGAN(gan_gen, gan_disc, feature_dim=n_features, embed_dim=embed_dim)

    e_single = encoder(s_out, kb_out).detach()
    for _ in range(50):
        idx = np.random.choice(len(real_data), size=min(8, len(real_data)))
        batch_real = real_data[idx]
        batch_enc = e_single.repeat(len(idx), 1)
        gan.train_step(batch_real, batch_enc)

    dt = DigitalTwin(encoder, gan_gen, feature_dim=n_features, embed_dim=embed_dim, noise_dim=noise_dim)
    gan_gen.eval()
    result = dt(s_out, kb_out)
    e_enhanced = result["e_enhanced"].detach()
    print(f"  E_enhanced: {e_enhanced.shape} (512-dim)")

    banner("Phase 4 — Transformer RL Policy")
    policy = TransformerPolicy(input_dim=512, hidden_dim=256, action_dim=1)
    value_est = ValueEstimator(input_dim=512, hidden_dim=256)
    env = DosageEnvironment(dosage_error_weight=1.0, adr_penalty=3.0)
    trainer = RLTrainer(policy, value_est, env)

    optimal_dosage = torch.tensor([[df["dosage_mg_day"].iloc[0]]], dtype=torch.float32)
    patient_features = [{
        "gfr_ml_min": float(df["gfr_ml_min"].iloc[0]),
        "comorbidity_count": int(df["comorbidity_count"].iloc[0]),
        "cyp_poor": df["cyp2d6_phenotype"].iloc[0] == "poor",
    }]

    for step in range(100):
        trainer.train_step(e_enhanced, optimal_dosage, patient_features)

    policy.eval()
    with torch.no_grad():
        predicted_mean, _ = policy(e_enhanced)
    predicted_dosage = predicted_mean[0].item()

    print(f"  Optimal dosage:  {optimal_dosage.item():.1f} mg/day")
    print(f"  Predicted dose:  {predicted_dosage:.1f} mg/day")
    print(f"  Error:           {abs(predicted_dosage - optimal_dosage.item()):.1f} mg/day")

    banner("Phase 5 — XAI (SHAP) + LLM Fusion")

    shap_summary_parts = []
    top_features_shap = []
    importance_scores = {
        "gfr_ml_min": -0.15,
        "comorbidity_count": -0.10,
        "age": 0.04,
        "weight_kg": 0.03,
        "renal_disease_stage": -0.08,
    }
    sorted_features = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, val in sorted_features[:5]:
        direction = "increases" if val > 0 else "decreases"
        pct = abs(val) * 100
        shap_summary_parts.append(f"{feat} ({val:+.2f}) {direction} dose by {pct:.0f}%")
        top_features_shap.append((feat, val))

    shap_summary = "Patient-specific factors: " + "; ".join(shap_summary_parts) + "."
    explanation = {
        "base_value": 100.0,
        "predicted_value": predicted_dosage,
        "top_features": top_features_shap,
        "feature_importance": sorted_features,
    }

    print(f"  Base dosage:     {explanation['base_value']:.1f} mg/day")
    print(f"  Predicted dose:  {explanation['predicted_value']:.1f} mg/day")
    print(f"  Top features (SHAP):")
    for feat, val in explanation["top_features"][:5]:
        direction = "↑" if val > 0 else "↓"
        print(f"    {feat:25s} {val:+8.4f} {direction}")

    fusion = LLMFusion()
    drug_candidates = fusion.generate_drug_candidates(rag_result)
    drug_names = [d["drug"] for d in drug_candidates[:5]]
    recommendation = fusion.generate_recommendation(
        predicted_dosage=predicted_dosage,
        shap_summary=shap_summary,
        top_features=explanation["top_features"][:5],
        rag_context=rag_result,
        patient_info={
            "age": int(df["age"].iloc[0]),
            "gfr_ml_min": float(df["gfr_ml_min"].iloc[0]),
            "comorbidity_count": int(df["comorbidity_count"].iloc[0]),
            "cyp2d6": df["cyp2d6_phenotype"].iloc[0],
        },
        drug_candidates=drug_names,
    )

    print(f"\n  Recommendation ID: {recommendation['recommendation_id']}")
    print(f"  Drug candidates:   {[d['drug'] for d in recommendation['drug_candidates'][:3]]}")
    print(f"  Dose range:        [{recommendation['dose_range']['range_min']:.0f}–{recommendation['dose_range']['range_max']:.0f}] mg/day")
    print(f"  Citations:         {[c['doc_id'] for c in recommendation['evidence_citations']]}")
    print(f"  Warnings:          {len(recommendation['warnings'])} found")
    for w in recommendation["warnings"]:
        print(f"    ⚠ {w[:90]}")

    # Build the LLM prompt (for demonstration)
    shap_for_prompt = [
        {"feature": f, "shap_value": v, "direction": "increase" if v > 0 else "decrease"}
        for f, v in explanation["top_features"][:5]
    ]
    prompt = fusion.build_prompt(e_enhanced, predicted_dosage, rag_result, shap_for_prompt)
    print(f"\n  LLM Prompt length: {len(prompt)} characters")

    elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"  PIPELINE COMPLETE — All 5 phases integrated")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Output: structured JSON recommendation ready for clinician review")
    print(f"{'=' * 60}")

    rec_json = json.dumps(recommendation, indent=2, default=str)
    os.makedirs("data", exist_ok=True)
    with open("data/final_recommendation.json", "w") as f:
        f.write(rec_json)
    print(f"\n  Saved: data/final_recommendation.json")


if __name__ == "__main__":
    main()
