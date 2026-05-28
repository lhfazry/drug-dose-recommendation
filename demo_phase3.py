import sys, os

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
from drug_dose.digital_twin import (
    DigitalTwin,
    RAGToEmbedding,
    TransformerEncoder,
)
from drug_dose.digital_twin.gan_module import (
    PatientGAN,
    PatientGANDiscriminator,
    PatientGANGenerator,
)


def main():
    print("=" * 60)
    print("PHASE 3: Digital Twin (GAN Module) Demo")
    print("=" * 60)

    fc = FeatureConfig()
    num_features = len(fc.get_numerical_features())
    embed_dim = 256
    noise_dim = 64

    print(f"\n[1] Generate real patient data (Phase 1)")
    gen = SyntheticDataGenerator(n_patients=500, random_seed=42)
    df = gen.generate()
    pipe = PreprocessingPipeline(fc)
    X_processed = pipe.fit_transform(df)
    real_data = torch.tensor(
        X_processed[fc.get_numerical_features()].values, dtype=torch.float32
    )
    print(f"    {real_data.shape[0]} patients × {real_data.shape[1]} numerical features")
    print(f"    Range: [{real_data.min():.2f}, {real_data.max():.2f}]")

    print(f"\n[2] Simulate RAG retrieval (Phase 2)")
    store = build_default_store()
    texts = store.get_document_texts()
    doc_ids = [d.doc_id for d in store.get_all_documents()]
    embedder = DocumentEmbedder("all-MiniLM-L6-v2")
    embeddings = embedder.embed_documents(texts)
    index = FAISSIndex()
    index.build(embeddings, doc_ids)
    retriever = HybridRetriever(embedder, index, store.to_dicts())
    retriever.index_documents(texts, doc_ids)
    pipeline = RAGPipeline(retriever, store)

    queries = [
        "lisinopril starting dose for hypertension with CKD stage 3 and type 2 diabetes",
        "metformin dosing in renal impairment GFR 22",
        "carboplatin AUC dosing for lung cancer with reduced kidney function",
    ]
    rag_results = [pipeline.process_query(q, top_k=5) for q in queries]
    print(f"    {len(rag_results)} RAG queries processed")
    for i, r in enumerate(rag_results):
        print(
            f"    Query {i+1}: top doc = {r['search_output'][0]['doc_id']} "
            f"({r['search_output'][0]['relevance']:.3f})"
        )

    print(f"\n[3] Convert RAG → Embeddings (RAGToEmbedding)")
    rag_embedder = RAGToEmbedding()
    s_out_list, kb_out_list = [], []
    for rag_result in rag_results:
        s_t, kb_t, _ = rag_embedder.prepare_inputs(rag_result, embedder)
        s_out_list.append(s_t)
        kb_out_list.append(kb_t)

    s_out = torch.cat(s_out_list, dim=0)
    kb_out = torch.cat(kb_out_list, dim=0)
    print(f"    S_out:   {s_out.shape}  (batch={len(queries)}, docs=5, dim={embedder.dim})")
    print(f"    KB_out:  {kb_out.shape}  (batch={len(queries)}, 1 summary, dim={embedder.dim})")

    print(f"\n[4] TransformerEncoder → E_input")
    encoder = TransformerEncoder(input_dim=embedder.dim, d_model=embed_dim)
    e_input = encoder(s_out, kb_out)
    print(f"    E_input: {e_input.shape}  (batch={len(queries)}, {embed_dim}-dim)")

    print(f"\n[5] Train PatientGAN on real data")
    gan_gen = PatientGANGenerator(
        embed_dim=embed_dim, noise_dim=noise_dim, feature_dim=num_features
    )
    gan_disc = PatientGANDiscriminator(feature_dim=num_features, embed_dim=embed_dim)
    gan = PatientGAN(gan_gen, gan_disc, feature_dim=num_features, embed_dim=embed_dim)

    n_steps = 100
    for step in range(n_steps):
        idx = np.random.choice(real_data.shape[0], size=min(32, len(queries)))
        batch_real = real_data[idx]
        batch_e = e_input[: len(idx)]
        gan.train_step(batch_real, batch_e)

    m = gan.train_step(batch_real, e_input)
    print(f"    After {n_steps} steps: G_loss={m['g_loss']:.3f}, D_loss={m['d_loss']:.3f}")
    print(f"    D accuracy: real={m['d_real_acc']:.3f}, fake={m['d_fake_acc']:.3f}")

    print(f"\n[6] DigitalTwin: Full Pipeline")
    dt = DigitalTwin(encoder, gan_gen, feature_dim=num_features, embed_dim=embed_dim, noise_dim=noise_dim)
    result = dt(s_out, kb_out)

    print(f"    E_input:      {result['e_input'].shape}")
    print(f"    X_synthetic:   {result['x_synthetic'].shape}  (GAN-generated patient data)")
    print(f"    E_enhanced:    {result['e_enhanced'].shape}  (256+256=512 dim)")

    stats = result["x_synthetic"].detach()
    print(f"    Synthetic range: [{stats.min():.2f}, {stats.max():.2f}]")
    print(f"    Synthetic mean:  {stats.mean(dim=0).mean():.3f}")

    print(f"\n{'=' * 60}")
    print("PHASE 3 COMPLETE — Digital Twin pipeline verified.")
    print("E_enhanced (512-dim) ready for Phase 4 Transformer RL Policy.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
