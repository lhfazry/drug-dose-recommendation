"""Phase 5: SHAP explainability and LLM Advisory Fusion.

Implements Section III-B.2 (SHAP feature importance) and Algorithm 3
(RAG-enhanced LLM recommendation fusion) from the paper.
"""

from drug_dose.xai.shap_explainer import ShapExplainer
from drug_dose.xai.llm_fusion import LLMFusion, generate_recommendation

__all__ = [
    "ShapExplainer",
    "LLMFusion",
    "generate_recommendation",
]
