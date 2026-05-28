# Drug Dose Recommendation — Agent Guide

## Quick start

```bash
# Required env var before running anything with HuggingFace tokenizers:
export TOKENIZERS_PARALLELISM=false

# Install the package from source:
pip install -e "src/."

# Or use the demo scripts directly (they do sys.path.insert(0, "src") for you):
python demo_phase1.py
python demo_full_pipeline.py   # ~5s on CPU for 200 patients
```

## Project structure

```
src/drug_dose/              # Package: `import drug_dose`
  data/                     # Synthetic data generation + FeatureConfig
    feature_config.py       # Central feature registry (27 features, 4 cohorts, 2 targets)
    synthetic_generator.py  # Generates ADR-20K synthetic patient records
  preprocessing/            # GAIN imputation + scaling + encoding
    pipeline.py             # PreprocessingPipeline (pickle-serializable)
  rag/                      # RAG pipeline
    document_store.py       # 60 mock clinical guideline documents (from JSON)
    embeddings.py           # DocumentEmbedder (all-MiniLM-L6-v2) + FAISSIndex
    retriever.py            # HybridRetriever (BM25 30% + dense 70%)
    rag_pipeline.py         # RAGPipeline: query → transform → retrieve → knowledge extraction
    data/clinical_documents.json  # ~60 synthetic clinical documents
  digital_twin/             # GAN-based digital twin
    transformer_encoder.py  # 4-layer, 8-head TransformerEncoder + position encoding
    gan_module.py           # Conditional GAN (generator + discriminator + trainer)
    digital_twin.py         # DigitalTwin orchestrator (E_input → GAN → E_enhanced)
  rl/                       # Reinforcement learning policy
    policy_network.py       # Gaussian TransformerPolicy + ValueEstimator + DosageEnvironment
    trainer.py              # RLTrainer (REINFORCE with value baseline)
  xai/                      # Explainability & recommendation fusion
    shap_explainer.py       # ShapExplainer wrapping KernelExplainer
    llm_fusion.py           # LLMFusion: template-based structured JSON recommendation
demo_phase1.py through demo_phase4.py  # Standalone phase demos
demo_full_pipeline.py        # Everything end-to-end on 200 patients
data/                        # Generated artifacts (parquet, pickle, JSON output)
notebooks/                   # Empty — no notebooks yet
```

## Architecture (5 phases in sequence)

1. **SyntheticDataGenerator** → **PreprocessingPipeline** (GAIN imputer, scaler, encoder)
2. **RAGPipeline** (document store → embedder → hybrid retriever → search + KB output)
3. **TransformerEncoder** + **PatientGAN** → **DigitalTwin** (E_enhanced 512-dim)
4. **TransformerPolicy** + **ValueEstimator** + **DosageEnvironment** → **RLTrainer** (REINFORCE)
5. **ShapExplainer** → **LLMFusion** → structured JSON recommendation

Full pipeline runs in ~5s on CPU for 200 patients.

## Key conventions & gotchas

- **Package name is `drug_dose`** (underscore), not `drug-dose-recommendation`.
- **No package install needed for demos** — each demo script prepends `src/` to `sys.path`. But to import from other scripts, do `pip install -e "src/."` or add `sys.path.insert(0, "src")`.
- **`rank_bm25` is optional** — a scratch BM25 fallback exists, so it degrades gracefully.
- **No GPU detection** — all tensors default to CPU. Add `.to("cuda" if torch.cuda.is_available() else "cpu")` if adding GPU support.
- **`TOKENIZERS_PARALLELISM=false` is mandatory** before importing any HuggingFace model (SentenceTransformer, transformers), or you'll get deadlock warnings.
- **No linter, formatter, type checker, or test framework configured.** Don't add configs without asking.
- **No test suite exists.** The only "test" is the `__main__` smoke-test at the bottom of `synthetic_generator.py`.
- **All data is synthetic** — patient records, clinical documents, everything. No real PHI.
- **`PreprocessingPipeline` uses pickle** for save/load. Version-sensitive.
- **`PreprocessingPipeline` references `feature_config.get_ordinal_features()` and `get_binary_features()` dynamically** — these methods don't exist on `FeatureConfig`, so ordinal/binary feature handling is effectively dead code.
- **`DosageEnvironment.compute_rewards()` double-imports `torch`** inside the method body — don't refactor this without checking imports.
- **SHAP is heavy** — `ShapExplainer` uses KernelExplainer with 100 nsamples, which can be slow for many features.
- **`LLMFusion.generate_recommendation()` accepts **kwargs** — check the docstring for all accepted parameters. It does NOT use an external LLM; it's template-based.
- **Clinical document data** lives in `src/drug_dose/rag/data/clinical_documents.json` as one big JSON array.

## Running individual phases

```bash
# Phase 1 only (2000 patients, saves parquet + pipeline pickle)
python demo_phase1.py

# Phase 2 only (RAG retrieval demo with 3 clinical queries)
python demo_phase2.py

# Phase 3 only (digital twin / GAN training)
python demo_phase3.py

# Phase 4 only (RL policy training)
python demo_phase4.py

# All 5 phases end-to-end (200 patients, saves final_recommendation.json)
python demo_full_pipeline.py
```

## Data flow

- `SyntheticDataGenerator(n_patients=20000, random_seed=42)` → generates 27-feature DataFrame with `cohort` column
- `PreprocessingPipeline(feature_config).fit_transform(df)` → scaled/encoded feature matrix
- `build_default_store()` loads 60 synthetic clinical docs from bundled JSON
- `DocumentEmbedder("all-MiniLM-L6-v2")` → 384-dim L2-normalized embeddings
- `FAISSIndex` → brute-force inner product (cosine) similarity
- `HybridRetriever(bm25_weight=0.3)` → 30% BM25 + 70% dense fusion
- `TransformerEncoder(input_dim=384, d_model=256, n_layers=4, n_heads=8)` → 256-dim E_input
- `PatientGANGenerator` conditioned on E_input + 64-dim noise → synthetic features (tanh-scaled to [-1,1])
- `DigitalTwin` → 512-dim E_enhanced (concat E_input + MLP(synthetic_features))
- `TransformerPolicy(input_dim=512)` → dosage mean in [10, 500] mg/day + learned log_std
- `ShapExplainer` → per-feature SHAP contributions via KernelExplainer
- `LLMFusion` → structured JSON (drug candidates, dose range, SHAP, evidence citations, warnings)
