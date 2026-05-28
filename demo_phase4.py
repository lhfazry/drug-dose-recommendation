import sys, os, time

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


def main():
    print("=" * 60)
    print("PHASE 4: Transformer-Based RL Policy Demo")
    print("=" * 60)

    fc = FeatureConfig()
    num_features = len(fc.get_numerical_features())
    embed_dim = 256
    noise_dim = 64

    print(f"\n[1] Setup: Data + RAG + DigitalTwin (Phases 1-3)")
    gen = SyntheticDataGenerator(n_patients=200, random_seed=42)
    df = gen.generate()
    pipe = PreprocessingPipeline(fc)
    X_processed = pipe.fit_transform(df)
    real_data = torch.tensor(
        X_processed[fc.get_numerical_features()].values, dtype=torch.float32
    )

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

    queries = [
        "lisinopril starting dose for hypertension with CKD and diabetes",
        "metformin dosing in renal impairment GFR 30",
        "carboplatin AUC dosing for lung cancer reduced kidney function",
        "ACE inhibitor dose adjustment in CKD stage 3",
        "beta blocker dosing in elderly hypertensive patient",
    ]
    rag_results = [rag.process_query(q, top_k=3) for q in queries]
    rag_embedder = RAGToEmbedding()
    s_list, kb_list = [], []
    for r in rag_results:
        s_t, kb_t, _ = rag_embedder.prepare_inputs(r, embedder)
        s_list.append(s_t)
        kb_list.append(kb_t)

    s_out = torch.cat(s_list, dim=0)
    kb_out = torch.cat(kb_list, dim=0)

    encoder = TransformerEncoder(input_dim=embedder.dim, d_model=embed_dim)
    gan_gen = PatientGANGenerator(embed_dim=embed_dim, noise_dim=noise_dim, feature_dim=num_features)
    dt = DigitalTwin(encoder, gan_gen, feature_dim=num_features, embed_dim=embed_dim, noise_dim=noise_dim)
    result = dt(s_out, kb_out)
    e_enhanced = result["e_enhanced"].detach()
    print(f"    E_enhanced: {e_enhanced.shape}  (batch={len(queries)}, 512-dim)")

    print(f"\n[2] Initialize RL Policy + Value + Environment")
    policy = TransformerPolicy(input_dim=512, hidden_dim=256, action_dim=1)
    value_est = ValueEstimator(input_dim=512, hidden_dim=256)
    env = DosageEnvironment(dosage_error_weight=1.0, adr_penalty=5.0)
    trainer = RLTrainer(policy, value_est, env)

    optimal_dosages = torch.tensor(
        df["dosage_mg_day"].values[: len(queries)], dtype=torch.float32
    ).unsqueeze(1)
    patient_features_list = []
    for i in range(len(queries)):
        patient_features_list.append({
            "gfr_ml_min": float(df["gfr_ml_min"].iloc[i]),
            "comorbidity_count": int(df["comorbidity_count"].iloc[i]),
            "cyp_poor": df["cyp2d6_phenotype"].iloc[i] == "poor",
        })
    print(f"    Optimal dosages: {optimal_dosages.squeeze().tolist()}")
    gfr_vals = [f'{p["gfr_ml_min"]:.0f}' for p in patient_features_list]
    print(f"    Patients: GFR={gfr_vals}")

    print(f"\n[3] Evaluate BEFORE training")
    policy.eval()
    with torch.no_grad():
        mean_before, _ = policy(e_enhanced)
    before_mae = (mean_before - optimal_dosages).abs().mean().item()
    print(f"    MAE before training: {before_mae:.1f} mg/day")

    print(f"\n[4] Train RL Policy (200 steps)")
    t0 = time.time()
    metrics_history = []
    for step in range(200):
        m = trainer.train_step(e_enhanced, optimal_dosages, patient_features_list)
        metrics_history.append(m)
    elapsed = time.time() - t0

    initial = metrics_history[0]
    final = metrics_history[-1]
    print(f"    Time: {elapsed:.1f}s")
    print(f"    Initial: P_loss={initial['policy_loss']:.3f}, V_loss={initial['value_loss']:.3f}, R={initial['reward']:.3f}")
    print(f"    Final:   P_loss={final['policy_loss']:.3f}, V_loss={final['value_loss']:.3f}, R={final['reward']:.3f}")
    print(f"    ΔR: {final['reward'] - initial['reward']:+.3f}")

    print(f"\n[5] Evaluate AFTER training")
    policy.eval()
    with torch.no_grad():
        mean_after, _ = policy(e_enhanced)
    after_mae = (mean_after - optimal_dosages).abs().mean().item()
    improvement = (before_mae - after_mae) / before_mae * 100
    print(f"    MAE after training:  {after_mae:.1f} mg/day")
    print(f"    Improvement:         {improvement:+.1f}%")

    print(f"\n[6] Sample Recommendations")
    print(f"    {'Patient':<8} {'Optimal':>8} {'Predicted':>10} {'Error':>8}")
    print(f"    {'-'*40}")
    for i in range(len(queries)):
        pred = mean_after[i].item()
        opt = optimal_dosages[i].item()
        err = abs(pred - opt)
        gfr = patient_features_list[i]["gfr_ml_min"]
        cyp = "poor" if patient_features_list[i]["cyp_poor"] else "normal"
        print(f"    P{i+1} (GFR={gfr:.0f},CYP={cyp})  {opt:7.1f}  {pred:10.1f}  {err:7.1f}")

    print(f"\n{'=' * 60}")
    print("PHASE 4 COMPLETE — Transformer RL Policy verified.")
    print(f"E_enhanced (512-dim) → dosage recommendation ({after_mae:.1f} mg/day MAE)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
