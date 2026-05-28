"""LLM Advisory Fusion — fuses model predictions with evidence and explanations.

Implements Algorithm 3 from the paper: Prompt_augmented = Concat[E_enhanced, optimal_action],
Recommendation = LLM(Prompt_augmented, KB_out). Uses template-based generation
since no real LLM API is available at this stage.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any


_DRUG_PATTERN = re.compile(
    r"\b(metformin|lisinopril|amlodipine|enalapril|losartan|valsartan|"
    r"carvedilol|metoprolol|atenolol|propranolol|hydrochlorothiazide|"
    r"warfarin|apixaban|rivaroxaban|clopidogrel|aspirin|"
    r"ibuprofen|naproxen|acetaminophen|paracetamol|morphine|"
    r"oxycodone|fentanyl|gabapentin|pregabalin|insulin|"
    r"glipizide|glyburide|levothyroxine|prednisone|methotrexate|"
    r"cyclophosphamide|doxorubicin|cisplatin|carboplatin|paclitaxel|"
    r"fluorouracil|capecitabine|tamoxifen|anastrozole|"
    r"digoxin|amiodarone|furosemide|spironolactone|omeprazole|"
    r"pantoprazole|ondansetron|atorvastatin|simvastatin|"
    r"empagliflozin|dapagliflozin|semaglutide|liraglutide|sitagliptin|"
    r"labetalol|nifedipine|clonidine|captopril|chlorthalidone|"
    r"pembrolizumab|nivolumab|imatinib|filgrastim|gentamicin|vancomycin)\b",
    re.IGNORECASE,
)

_DRUG_RATIONALES: dict[str, dict[str, str]] = {
    "lisinopril": {
        "rationale": "ACE inhibitor, first-line for HTN with CKD per AHA/ACC guidelines",
        "evidence_source": "HTN-001",
    },
    "losartan": {
        "rationale": "ARB alternative if ACEi not tolerated",
        "evidence_source": "HTN-002",
    },
    "amlodipine": {
        "rationale": "CCB, effective for HTN and stable angina per JNC 8",
        "evidence_source": "HTN-003",
    },
    "enalapril": {
        "rationale": "ACE inhibitor, proven mortality benefit in HFrEF per SOLVD trial",
        "evidence_source": "HF-001",
    },
    "metoprolol": {
        "rationale": "Beta-blocker, mortality benefit post-MI and in HFrEF",
        "evidence_source": "HF-002",
    },
    "carvedilol": {
        "rationale": "Non-selective beta-blocker with alpha blockade, HFrEF benefit",
        "evidence_source": "HF-003",
    },
    "metformin": {
        "rationale": "First-line oral agent for T2DM per ADA guidelines",
        "evidence_source": "DM-001",
    },
    "empagliflozin": {
        "rationale": "SGLT2 inhibitor with cardiovascular and renal benefit per EMPA-REG",
        "evidence_source": "DM-002",
    },
    "warfarin": {
        "rationale": "Vitamin K antagonist, requires INR monitoring and PGx guidance",
        "evidence_source": "PGX-001",
    },
    "apixaban": {
        "rationale": "DOAC with fixed dosing, lower bleeding risk vs warfarin per ARISTOTLE",
        "evidence_source": "PGX-002",
    },
    "atorvastatin": {
        "rationale": "Statin, first-line for ASCVD prevention per ACC/AHA guidelines",
        "evidence_source": "CV-001",
    },
    "simvastatin": {
        "rationale": "Statin with SLCO1B1 PGx interaction risk at high doses",
        "evidence_source": "PGX-003",
    },
    "cisplatin": {
        "rationale": "Platinum-based chemotherapy, renally cleared — dose per Calvert formula",
        "evidence_source": "ONC-001",
    },
    "carboplatin": {
        "rationale": "Platinum analog with reduced nephrotoxicity vs cisplatin",
        "evidence_source": "ONC-002",
    },
    "paclitaxel": {
        "rationale": "Taxane, CYP2C8-metabolized — neuropathy risk correlates with exposure",
        "evidence_source": "ONC-003",
    },
    "insulin": {
        "rationale": "Basal-bolus regimen titrated to fasting and postprandial glucose",
        "evidence_source": "DM-003",
    },
    "furosemide": {
        "rationale": "Loop diuretic, first-line for volume overload in HF and CKD",
        "evidence_source": "HF-004",
    },
    "spironolactone": {
        "rationale": "Mineralocorticoid antagonist, mortality benefit in HFrEF per RALES",
        "evidence_source": "HF-005",
    },
    "gabapentin": {
        "rationale": "Gabapentinoid, renally adjusted for neuropathic pain",
        "evidence_source": "PAIN-001",
    },
    "methotrexate": {
        "rationale": "DMARD, requires folate supplementation and renal monitoring",
        "evidence_source": "RHEUM-001",
    },
    "clopidogrel": {
        "rationale": "P2Y12 inhibitor, CYP2C19 PGx guidance for reduced efficacy in PMs",
        "evidence_source": "PGX-004",
    },
    "vancomycin": {
        "rationale": "Glycopeptide antibiotic, AUC-guided dosing with TDM",
        "evidence_source": "ID-001",
    },
    "gentamicin": {
        "rationale": "Aminoglycoside, ototoxic/nephrotoxic — extended-interval dosing preferred",
        "evidence_source": "ID-002",
    },
    "pembrolizumab": {
        "rationale": "Anti-PD-1 immunotherapy, fixed dose 200mg Q3W or 400mg Q6W",
        "evidence_source": "ONC-004",
    },
    "levothyroxine": {
        "rationale": "Thyroid hormone replacement, weight-based dosing ~1.6 mcg/kg",
        "evidence_source": "ENDO-001",
    },
    "omeprazole": {
        "rationale": "PPI, CYP2C19 metabolism — consider dose adjustment for PM/UM phenotypes",
        "evidence_source": "PGX-005",
    },
    "acetaminophen": {
        "rationale": "Analgesic/antipyretic, max 3-4g/day, hepatotoxic in overdose",
        "evidence_source": "PAIN-002",
    },
    "prednisone": {
        "rationale": "Corticosteroid, taper dosing to avoid adrenal insufficiency",
        "evidence_source": "RHEUM-002",
    },
}


class LLMFusion:
    """Generates structured clinical recommendations using RAG context and SHAP explanations.

    Implements Algorithm 3 from the paper: fuses model predictions with evidence
    and explanations into a structured JSON output for clinician review.
    """

    _ace_arb_drugs = {
        "lisinopril", "enalapril", "captopril", "losartan",
        "valsartan", "candesartan", "irbesartan",
    }

    def generate_recommendation(self, **kwargs: Any) -> dict[str, Any]:
        """Generate a structured clinical recommendation.

        Keyword Args:
            predicted_dosage: float — recommended dosage in mg/day.
            shap_summary: str — natural language SHAP summary.
            top_features: list[tuple[str, float]] — (name, shap_value) pairs.
            rag_context: dict — output from RAGPipeline.process_query().
            patient_info: dict — patient demographics and labs.
            drug_candidates: list[str] — drugs mentioned in RAG context.
            base_value: float — SHAP baseline prediction E[f(x)].

        Returns:
            Structured recommendation dict with drug_candidates, dose_range,
            rationale_bullets, shap_explanation, evidence_citations,
            uncertainty, warnings, and generated_at.
        """
        predicted_dosage = float(kwargs.get("predicted_dosage", 0))
        shap_summary = str(kwargs.get("shap_summary", ""))
        top_features: list[tuple[str, float]] = kwargs.get("top_features", [])
        rag_context: dict[str, Any] = kwargs.get("rag_context", {})
        patient_info: dict[str, Any] = kwargs.get("patient_info", {})
        drug_candidates: list[str] = kwargs.get("drug_candidates", [])
        base_value: float = float(kwargs.get("base_value", predicted_dosage))

        search_output = rag_context.get("search_output", [])
        kb_output = rag_context.get("knowledge_base_output", {})

        if not drug_candidates:
            drug_candidates = kb_output.get("drug_mentions", [])

        drug_entries = self._build_drug_entries(drug_candidates)

        dose_range = {
            "recommended": predicted_dosage,
            "range_min": max(10.0, predicted_dosage * 0.7),
            "range_max": predicted_dosage * 1.3,
            "unit": "mg/day",
        }

        rationale_bullets = self._build_rationale_bullets(
            predicted_dosage, shap_summary, search_output, kb_output,
            patient_info,
        )

        shap_explanation = self._build_shap_explanation(
            base_value, top_features, shap_summary,
        )

        evidence_citations: list[dict[str, Any]] = []
        for doc in search_output[:3]:
            if isinstance(doc, dict):
                evidence_citations.append({
                    "doc_id": doc.get("doc_id", ""),
                    "title": doc.get("title", "Untitled"),
                    "source": doc.get("source", "Unknown"),
                    "relevance": doc.get("relevance", 0.0),
                })

        warnings = self._build_warnings(drug_candidates, rag_context, patient_info)

        return {
            "recommendation_id": self._generate_recommendation_id(),
            "drug_candidates": drug_entries,
            "dose_range": dose_range,
            "rationale_bullets": rationale_bullets,
            "shap_explanation": shap_explanation,
            "evidence_citations": evidence_citations,
            "uncertainty": {
                "confidence": "moderate",
                "notes": (
                    "Based on synthetic patient model. "
                    "Clinician review required before prescribing."
                ),
            },
            "warnings": warnings,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def build_prompt(
        self,
        e_enhanced: str,
        optimal_action: float,
        rag_context: dict[str, Any],
        shap_features: list[dict[str, Any]],
    ) -> str:
        """Construct the full LLM prompt fusing all components.

        Sections: SYSTEM, PATIENT CONTEXT, EVIDENCE, MODEL PREDICTION,
        SHAP EXPLANATION, TASK.
        """
        kb = rag_context.get("knowledge_base_output", {})
        key_facts = kb.get("key_facts", [])
        fact_lines = "\n".join(f"  - {f}" for f in (key_facts[:5] or ["No key facts available."]))

        shap_lines = "\n".join(
            f"  - {sf['feature']}: SHAP={sf['shap_value']:+.4f} ({sf['direction']})"
            for sf in shap_features[:8]
        ) if shap_features else "  No SHAP features provided."

        search_output = rag_context.get("search_output", [])
        evidence_lines: list[str] = []
        for doc in search_output[:3]:
            if isinstance(doc, dict):
                evidence_lines.append(
                    f"  [{doc.get('doc_id', '?')}] {doc.get('title', 'Untitled')} "
                    f"— {doc.get('snippet', '')[:120]}"
                )
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "  No evidence documents retrieved."

        prompt = f"""SYSTEM:
You are a clinical decision support assistant specialized in drug dosage
recommendation. Your role is to synthesize patient data, model predictions,
evidence from medical literature, and explainability analysis into a clear,
actionable recommendation for a prescribing clinician.

PATIENT CONTEXT:
{e_enhanced}

EVIDENCE FROM KNOWLEDGE BASE:
{evidence_text}

KEY CLINICAL FACTS:
{fact_lines}

MODEL PREDICTION:
Optimal recommended dosage: {optimal_action:.1f} mg/day

SHAP EXPLANATION (Feature Contributions):
{shap_lines}

TASK:
Synthesize the above information into a structured clinical recommendation.
For each drug candidate, provide the rationale and evidence source.
Include the recommended dose range with confidence bounds.
List key rationale points that justify the dosage.
Highlight any safety warnings or monitoring requirements.
Cite evidence sources for each claim.
"""
        return prompt

    def generate_drug_candidates(self, rag_context: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract and rank drug candidates from RAG context.

        Args:
            rag_context: Output from RAGPipeline.process_query().

        Returns:
            List of {drug, rationale, evidence_source} dicts, ranked
            by mention frequency in retrieved documents.
        """
        raw_docs = rag_context.get("raw_documents", [])
        if not raw_docs:
            search_output = rag_context.get("search_output", [])
            raw_docs = search_output

        drug_counter: Counter[str] = Counter()
        for doc in raw_docs:
            if isinstance(doc, dict):
                content = doc.get("content", "") or doc.get("snippet", "")
            else:
                content = str(doc)
            for match in _DRUG_PATTERN.finditer(content):
                drug_counter[match.group(1).lower()] += 1

        candidates: list[dict[str, Any]] = []
        for drug, count in drug_counter.most_common(10):
            info = _DRUG_RATIONALES.get(drug, {
                "rationale": f"Medication relevant to clinical context (mentions: {count})",
                "evidence_source": "GEN-000",
            })
            candidates.append({
                "drug": drug.capitalize(),
                "rationale": info["rationale"],
                "evidence_source": info["evidence_source"],
                "mention_count": count,
            })

        return candidates

    @staticmethod
    def _generate_recommendation_id() -> str:
        now = datetime.now(timezone.utc)
        return f"REC-{now.strftime('%Y%m%d%H%M%S')}-{now.microsecond // 1000:03d}"

    def _build_drug_entries(
        self, drug_candidates: list[str],
    ) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        for drug in drug_candidates:
            drug_lower = drug.lower()
            info = _DRUG_RATIONALES.get(drug_lower, {
                "rationale": "Candidate medication based on clinical context",
                "evidence_source": "GEN-000",
            })
            entries.append({
                "drug": drug.capitalize(),
                "rationale": info["rationale"],
                "evidence_source": info["evidence_source"],
            })
        if not entries:
            entries.append({
                "drug": "Unknown",
                "rationale": "No specific drug candidates identified from RAG context",
                "evidence_source": "N/A",
            })
        return entries

    @staticmethod
    def _build_rationale_bullets(
        predicted_dosage: float,
        shap_summary: str,
        search_output: list[dict[str, Any]],
        kb_output: dict[str, Any],
        patient_info: dict[str, Any],
    ) -> list[str]:
        bullets: list[str] = []

        bullets.append(
            f"Recommended dosage of {predicted_dosage:.0f} mg/day "
            f"based on patient-specific factors"
        )

        if shap_summary:
            bullets.append(shap_summary)

        if search_output:
            first_doc = search_output[0]
            snippet = first_doc.get("snippet", "")[:100]
            doc_id = first_doc.get("doc_id", "?")
            bullets.append(
                f"Evidence from {doc_id}: {snippet}"
            )

        key_facts = kb_output.get("key_facts", [])
        fact_text = key_facts[0] if key_facts else "N/A"
        bullets.append(
            f"Key clinical consideration: {fact_text}"
        )

        age = patient_info.get("age")
        gfr = patient_info.get("gfr_ml_min")
        if age is not None or gfr is not None:
            parts: list[str] = []
            if age is not None:
                parts.append(f"age={age}")
            if gfr is not None:
                parts.append(f"GFR={gfr} mL/min")
            bullets.append(
                f"Patient factors considered: {', '.join(parts)}"
            )

        return bullets

    @staticmethod
    def _build_shap_explanation(
        base_value: float,
        top_features: list[tuple[str, float]],
        shap_summary: str,
    ) -> dict[str, Any]:
        feature_contributions: list[dict[str, Any]] = []
        for name, val in top_features[:10]:
            feature_contributions.append({
                "feature": name,
                "contribution": round(val, 5),
                "direction": "increase" if val > 0 else "decrease",
            })

        return {
            "base_dosage": base_value,
            "feature_contributions": feature_contributions,
            "summary": shap_summary,
        }

    def _build_warnings(
        self,
        drug_candidates: list[str],
        rag_context: dict[str, Any],
        patient_info: dict[str, Any],
    ) -> list[str]:
        warnings: list[str] = []

        drug_text = " ".join(drug_candidates).lower() if drug_candidates else ""
        ace_or_arb = any(d in drug_text for d in self._ace_arb_drugs)
        if ace_or_arb:
            warnings.append(
                "Monitor renal function and potassium within 1-2 weeks of "
                "dose change (ACEi/ARB therapy)"
            )

        rag_text = str(rag_context).lower()
        if "diabetes" in rag_text:
            warnings.append(
                "Risk of hypoglycemia — monitor blood glucose closely"
            )

        if "warfarin" in drug_text or "anticoagul" in rag_text:
            warnings.append(
                "Monitor INR regularly per anticoagulation protocol"
            )

        if "nephrotox" in rag_text or "renal" in rag_text:
            gfr = patient_info.get("gfr_ml_min")
            if gfr is not None and gfr < 60:
                warnings.append(
                    f"Renally adjusted dosing required (GFR={gfr} mL/min). "
                    "Recheck renal function before each cycle."
                )

        cyp2d6 = patient_info.get("cyp2d6_phenotype", "")
        if cyp2d6 and cyp2d6.lower() == "poor":
            warnings.append(
                "CYP2D6 poor metabolizer — consider dose reduction for "
                "CYP2D6-metabolized drugs"
            )

        return warnings


def generate_recommendation(**kwargs: Any) -> dict[str, Any]:
    """Convenience function — creates LLMFusion and generates recommendation.

    See LLMFusion.generate_recommendation for parameter details.
    """
    fusion = LLMFusion()
    return fusion.generate_recommendation(**kwargs)
