# Drug and Dosage Recommendation Based on Explainable Generative AI Using Patient-Specific Modeling

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License MIT](https://img.shields.io/badge/license-MIT-green)

An end-to-end framework for personalized drug and dosage recommendation combining synthetic patient data, retrieval-augmented generation (RAG), generative adversarial networks (GANs), reinforcement learning (RL), and explainable AI (XAI). The system generates patient-specific digital twins, learns optimal dosing policies via transformer-based RL, and produces structured clinical recommendations with SHAP-based feature attribution.

This project implements the methodology described in the paper *Drug and Dosage Recommendation Based on Explainable Generative AI Using Patient-Specific Modeling* (see [Citation](#citation)).

---

## Architecture

The pipeline consists of five interconnected phases:

```
Patient Query
     |
     v
Phase 1: Synthetic Data Generation & Preprocessing
  - 20K synthetic patients (4 cohorts, 27 features)
  - GAIN imputation + StandardScaler + encoding
  - Output: cleaned feature matrix X_proc
     |
     v
Phase 2: Retrieval-Augmented Generation (RAG)
  - 60 clinical guideline documents
  - Hybrid retriever (BM25 30% + dense 70%)
  - Query transformation + knowledge-base extraction
  - Output: Search Output + Knowledge-Base Output
     |
     v
Phase 3: Digital Twin (GAN)
  - TransformerEncoder (4 layers, 8 heads, d_model=256)
  - Conditional GAN (generator + discriminator)
  - GAN generates synthetic patient features X_synthetic
  - Concatenated with E_input → E_enhanced (512-dim)
  - Output: enhanced patient embedding E_enhanced
     |
     v
Phase 4: Transformer RL Policy
  - Gaussian policy network (3-layer MLP + residual)
  - Value estimator + REINFORCE with baseline
  - DosageEnvironment: reward = -(error penalty) - (ADR penalty)
  - Output: predicted optimal dosage [10-500] mg/day
     |
     v
Phase 5: XAI + LLM Fusion
  - SHAP KernelExplainer (per-feature contributions)
  - Template-based structured JSON generation
  - Output: drug candidates, dose range, rationale, evidence citations, warnings
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- pip or poetry

### Setup

```bash
git clone <repository-url>
cd drug-dose-recommendation

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Or using poetry:

```bash
poetry install
```

### Environment Variable

Before running any demo script, set the following environment variable to prevent HuggingFace tokenizer parallelism warnings:

```bash
export TOKENIZERS_PARALLELISM=false
```

---

## Usage

The project provides five standalone demo scripts and one integrated end-to-end pipeline. Run them from the project root after activating your virtual environment.

### Phase 1: Synthetic Data Generation & Preprocessing

```bash
python demo_phase1.py
```

Generates 2,000 synthetic patients, validates clinical correlations (age-GFR, age-BP, diabetes-HbA1c), and runs the preprocessing pipeline (GAIN imputation, scaling, encoding). Saves:
- `data/adr20k_synthetic_2000.parquet` — raw generated dataset
- `data/preprocessing_pipeline.pkl` — fitted sklearn pipeline

### Phase 2: RAG Document Retrieval

```bash
python demo_phase2.py
```

Builds a document store of 60 clinical guideline documents, indexes them with FAISS (dense) and BM25 (lexical), and processes three clinical test queries. Displays search results ranked by hybrid relevance score and extracted knowledge-base output (key facts, drug mentions, cohorts, evidence levels).

### Phase 3: Digital Twin (GAN)

```bash
python demo_phase3.py
```

Generates real patient data (Phase 1), runs RAG retrieval (Phase 2), converts results to embeddings, encodes them with the TransformerEncoder, trains the conditional GAN for 100 steps, and produces the 512-dimensional E_enhanced embedding via the DigitalTwin pipeline.

### Phase 4: Transformer RL Policy

```bash
python demo_phase4.py
```

Runs Phases 1-3 to produce E_enhanced, then trains the transformer-based RL policy (REINFORCE with value baseline) for 200 steps. Reports mean absolute error before and after training, and shows per-patient dose predictions with GFR and CYP2D6 context.

### Full Pipeline (Phase 1-5)

```bash
python demo_full_pipeline.py
```

Runs all five phases end-to-end on a 200-patient subset. Outputs a structured JSON recommendation to `data/final_recommendation.json` containing drug candidates with rationale, dose range, SHAP explanations, evidence citations, and clinical warnings. Total runtime is approximately 5 seconds on CPU.

---

## Dataset: ADR-20K (Synthetic)

The system uses a synthetically generated dataset of 20,000 patient records distributed across four cohorts:

| Cohort | Proportion | Size |
|---|---|---|
| Hypertension | 30% | 6,000 |
| Diabetes Mellitus | 25% | 5,000 |
| Renal Impairment | 25% | 5,000 |
| Oncology | 20% | 4,000 |

Each record contains **27 features** across five groups and **2 targets**:

**Features (27):**
- **Demographics (5):** age, sex, weight_kg, height_cm, bmi
- **Medical History (6):** hypertension_diagnosis, diabetes_diagnosis, cancer_type, renal_disease_stage, prior_adr_count, comorbidity_count
- **Laboratory Values (7):** gfr_ml_min, creatinine_mg_dl, alt_u_l, ast_u_l, hba1c_percent, wbc_count, platelet_count
- **Vital Signs (4):** systolic_bp_mmhg, diastolic_bp_mmhg, heart_rate_bpm, temperature_c
- **Genomic Markers (5):** cyp2d6_phenotype, cyp2c19_phenotype, cyp2c9_phenotype, slco1b1_rs4149056, vkorc1_rs9923231

**Targets (2):**
- `dosage_mg_day`: continuous dose recommendation [10-500] mg/day
- `adr_risk`: binary adverse drug reaction risk flag {0, 1}

The data incorporates clinically realistic correlations: GFR declines with age and renal disease stage, HbA1c is elevated in diabetic patients, blood pressure rises with age and hypertension diagnosis, and CYP genotype influences effective dosage.

---

## Methods

### Phase 1: Synthetic Data Generation & Preprocessing

The `SyntheticDataGenerator` produces realistic patient records with controlled covariance structure. Cohort labels are assigned deterministically based on condition flags (hypertension diagnosis, diabetes diagnosis, renal disease stage, cancer type).

The `PreprocessingPipeline` handles missing values using **GAIN (Generative Adversarial Imputation Nets)**, a deep learning imputation method where a generator predicts missing values and a discriminator attempts to distinguish imputed from observed values. Numerical features are normalized with `StandardScaler`, ordinal categoricals with `OrdinalEncoder`, and nominal categoricals with `OneHotEncoder`.

### Phase 2: Retrieval-Augmented Generation (RAG)

A corpus of 60 synthetic clinical guideline documents is curated across 5 categories (guidelines, case studies, drug monographs, evidence summaries) and 5 cohorts (hypertension, diabetes mellitus, renal impairment, oncology, general).

**DocumentEmbedder** uses `all-MiniLM-L6-v2` (SentenceTransformer, 384-dimensional, L2-normalized) for dense embeddings. **FAISS IndexFlatIP** performs brute-force inner-product (cosine) search. **BM25 Okapi** provides lexical keyword matching.

The **HybridRetriever** fuses BM25 and dense scores with a 30:70 weight ratio. A **query transformation** step expands clinical queries via synonym expansion, drug-name detection, and lab-value extraction.

The **RAGPipeline** produces two output streams:
- **Search Output:** ranked documents with relevance scores
- **Knowledge-Base Output:** aggregated key facts, drug mentions, relevant cohorts, and evidence levels

### Phase 3: Digital Twin (GAN)

The **TransformerEncoder** is a 4-layer, 8-head attention network (d_model=256) with sinusoidal positional encoding (Vaswani et al. 2017), GELU activation, and LayerNorm. It fuses three inputs:
- **S_out:** search output embeddings (top-k documents)
- **KB_out:** knowledge-base summary embedding
- **F_loop:** optional feedback from prior recommendations

The **PatientGANGenerator** takes a conditioned embedding E_input (256-dim) concatenated with 64-dimensional Gaussian noise, passes it through hidden layers, and outputs synthetic patient features scaled to [-1, 1] via Tanh. The **PatientGANDiscriminator** uses leaky ReLU activations and Dropout (p=0.3).

Training uses adversarial loss combined with feature matching loss (mean + standard deviation, lambda=0.1). The Adam optimizer runs at lr=0.0002 with betas=(0.5, 0.999).

The **DigitalTwin** pipeline produces **E_enhanced** (512-dim) by:
1. Encoding S_out + KB_out via TransformerEncoder → E_input (256-dim)
2. Generating synthetic features via GAN → X_synthetic
3. Encoding X_synthetic through an MLP (128 → 256)
4. Concatenating E_input with the encoded synthetic features

### Phase 4: Transformer RL Policy

The **TransformerPolicy** implements a Gaussian policy over the continuous action space [10, 500] mg/day. It uses a 3-layer MLP with residual connections, LayerNorm, ReLU, and Dropout. The output mean is tanh-scaled and the log_std is a learned parameter.

The **ValueEstimator** is a 3-layer MLP producing a scalar state-value estimate V(s).

The **DosageEnvironment** defines the reward function:

```
reward = -(1.0 * |error| / 500) - (5.0 * ADR_risk)
```

where ADR risk is modeled as:

```
ADR_risk = sigmoid(b0 + b1 * dosage + b2 * (1 - GFR/120) + b3 * comorbidity/8)
```

CYP2D6 poor metabolizers receive a 2x multiplier on ADR risk.

The **RLTrainer** implements REINFORCE with a learned value baseline. The advantage is computed as `A_t = R_t - V(s_t)`, and the joint loss combines policy gradient loss with MSE value loss.

### Phase 5: XAI + LLM Fusion

The **ShapExplainer** wraps SHAP's `KernelExplainer` for model-agnostic feature attribution. It uses 50 background samples and 100 nsamples for the SHAP approximation, returning per-feature contribution scores (phi_i).

The **LLMFusion** module produces a structured JSON recommendation through template-based generation (no external LLM API required). The output contains:

- **Drug candidates** — top medications with rationale and evidence source
- **Dose range** — recommended dose with [70%-130%] safety range
- **Rationale bullets** — human-readable explanation of contributing factors
- **SHAP explanation** — per-feature contribution values and direction
- **Evidence citations** — supporting documents with relevance scores
- **Warnings** — clinically relevant alerts (ACEi renal monitoring, hypoglycemia risk, renally adjusted dosing, CYP2D6 poor metabolizer considerations)

The prompt builder (`build_prompt()`) assembles a structured prompt from SYSTEM + PATIENT CONTEXT + EVIDENCE + MODEL PREDICTION + SHAP + TASK sections.

---

## Results

Running the full pipeline produces a structured JSON recommendation. Below is a representative output:

```json
{
  "recommendation_id": "REC-20260528074827-462",
  "drug_candidates": [
    {
      "drug": "Lisinopril",
      "rationale": "ACE inhibitor, first-line for HTN with CKD per AHA/ACC guidelines",
      "evidence_source": "HTN-001"
    },
    {
      "drug": "Losartan",
      "rationale": "ARB alternative if ACEi not tolerated",
      "evidence_source": "HTN-002"
    },
    {
      "drug": "Amlodipine",
      "rationale": "CCB, effective for HTN and stable angina per JNC 8",
      "evidence_source": "HTN-003"
    }
  ],
  "dose_range": {
    "recommended": 10.0,
    "range_min": 10.0,
    "range_max": 13.0,
    "unit": "mg/day"
  },
  "shap_explanation": {
    "base_dosage": 10.0,
    "feature_contributions": [
      {"feature": "gfr_ml_min", "contribution": -0.15, "direction": "decrease"},
      {"feature": "comorbidity_count", "contribution": -0.10, "direction": "decrease"},
      {"feature": "renal_disease_stage", "contribution": -0.08, "direction": "decrease"},
      {"feature": "age", "contribution": 0.04, "direction": "increase"},
      {"feature": "weight_kg", "contribution": 0.03, "direction": "increase"}
    ]
  },
  "evidence_citations": [
    {"doc_id": "HTN-008", "title": "Case: 55M Hypertensive with Diabetes, Successful Lisinopril 20 mg", "relevance": 0.841},
    {"doc_id": "HTN-009", "title": "Case: 62F Hypertensive with CKD Stage 3, Dose-Adjusted Losartan", "relevance": 0.574},
    {"doc_id": "HTN-001", "title": "ACE Inhibitor First-Line Guidelines for Hypertension", "relevance": 0.573}
  ],
  "warnings": [
    "Monitor renal function and potassium within 1-2 weeks of dose change (ACEi/ARB therapy)",
    "Risk of hypoglycemia - monitor blood glucose closely",
    "Renally adjusted dosing required (GFR=15.0 mL/min). Recheck renal function before each cycle."
  ],
  "generated_at": "2026-05-28T07:48:27.462265+00:00"
}
```

The RL policy converges to a mean absolute error of approximately 10-30 mg/day after 200 training steps, with the GAN-based digital twin providing effective patient representation augmentation.

---

## Library Dependencies

| Library | Version | Purpose |
|---|---|---|
| numpy | >=1.24.0 | Numerical computation |
| pandas | >=2.0.0 | Data manipulation |
| scikit-learn | >=1.3.0 | Preprocessing, encoding, scaling |
| torch | >=2.0.0 | Deep learning (GAN, Transformer, RL) |
| transformers | >=4.35.0 | HuggingFace utilities |
| faiss-cpu | >=1.7.4 | Vector similarity search |
| sentence-transformers | >=2.2.0 | Document embeddings |
| shap | >=0.42.0 | Model explainability |
| pyyaml | >=6.0 | Configuration |
| tqdm | >=4.65.0 | Progress bars |
| matplotlib | >=3.7.0 | Visualization |
| seaborn | >=0.12.0 | Statistical plotting |
| rank_bm25 | optional | Faster BM25 (falls back to scratch implementation) |

---

## Citation

If you use this work in your research, please cite the associated paper:

```
Drug and Dosage Recommendation Based on Explainable Generative AI Using
Patient-Specific Modeling. Available at:
<paper URL or DOI>
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
