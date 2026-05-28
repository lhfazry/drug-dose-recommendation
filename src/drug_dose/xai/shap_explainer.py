"""SHAP-based feature importance analysis for dosage recommendations.

Implements Section III-B.2 of the paper: Per-patient explainability via SHAP
(KernelExplainer for closed-box transformer/ensemble models). φ_i values are
computed for each feature in the patient profile to identify which factors
most influence the recommended dosage.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import shap
import torch


class ShapExplainer:
    """SHAP-based feature importance for dosage recommendations.

    Uses KernelExplainer for closed-box models (transformer/ensemble).
    Computes SHAP values φ_i for each feature in the patient profile.
    """

    def __init__(
        self,
        policy: Callable[[np.ndarray], np.ndarray],
        feature_names: list[str],
        background_samples: int = 50,
    ) -> None:
        """Initialize the SHAP explainer.

        Args:
            policy: A callable that takes (n_samples, n_features) numpy array
                    and returns dosage predictions as a numpy array of shape
                    (n_samples,). Typically wraps TransformerPolicy + encoder.
            feature_names: Human-readable feature names (e.g. from
                           FeatureConfig.get_numerical_features()).
            background_samples: Number of background samples for KernelExplainer.
        """
        self.policy = policy
        self.feature_names = feature_names
        self.background_samples = background_samples
        self.explainer: shap.KernelExplainer | None = None
        self.is_fitted: bool = False

    def _predict_fn(self, x: np.ndarray) -> np.ndarray:
        """Prediction wrapper for SHAP KernelExplainer.

        Args:
            x: Feature array of shape (n_samples, n_features).

        Returns:
            Dosage predictions as (n_samples,) numpy array.
        """
        return self.policy(x)

    def fit(self, background_data: np.ndarray) -> None:
        """Initialize KernelExplainer with background reference data.

        Args:
            background_data: Array of shape (n_background, n_features)
                             representing the reference distribution.
        """
        if background_data.shape[0] > self.background_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                background_data.shape[0],
                size=self.background_samples,
                replace=False,
            )
            background_data = background_data[indices]

        self.explainer = shap.KernelExplainer(self._predict_fn, background_data)
        self.is_fitted = True

    def explain(self, patient_data: np.ndarray) -> dict[str, Any]:
        """Compute SHAP values for a single patient.

        Args:
            patient_data: Feature array of shape (n_features,) or
                          (1, n_features).

        Returns:
            Dictionary with shap_values, base_value, predicted_value,
            feature_importance, and top_features.
        """
        if not self.is_fitted or self.explainer is None:
            raise RuntimeError("Explainer not fitted. Call fit() first.")

        patient_data = np.atleast_2d(patient_data)

        raw_shap_values = self.explainer.shap_values(
            patient_data, nsamples=100
        )

        if isinstance(raw_shap_values, list):
            shap_values_arr = np.array(raw_shap_values)
            if shap_values_arr.ndim == 3:
                shap_values_arr = shap_values_arr[0]
        else:
            shap_values_arr = np.asarray(raw_shap_values)

        if shap_values_arr.ndim == 2:
            shap_values_arr = shap_values_arr[0]

        base_value = float(self.explainer.expected_value)
        predicted = float(self._predict_fn(patient_data)[0])

        feature_importance: list[tuple[str, float]] = sorted(
            zip(self.feature_names, shap_values_arr.tolist()),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )

        top_features: list[dict[str, Any]] = []
        for name, val in feature_importance[:10]:
            top_features.append({
                "feature": name,
                "shap_value": round(val, 5),
                "direction": "increase" if val > 0 else "decrease",
                "magnitude_pct": round(abs(val) / (abs(base_value) + 1e-8) * 100, 1)
                if base_value != 0 else 0.0,
            })

        return {
            "shap_values": shap_values_arr.tolist(),
            "base_value": base_value,
            "predicted_value": predicted,
            "feature_importance": feature_importance,
            "top_features": top_features,
        }

    def summarize(self, explanation: dict[str, Any]) -> str:
        """Generate a natural language summary from SHAP values.

        Example:
            "GFR (SHAP: -0.15) decreases recommended dose by 15%.
             CYP2D6 poor metabolizer (SHAP: -0.08) further reduces dose
             by 8%. Age (SHAP: +0.03) slightly increases dose."

        Args:
            explanation: Dictionary returned by explain().

        Returns:
            A human-readable summary string.
        """
        top = explanation.get("top_features", [])
        if not top:
            return "No significant feature contributions detected."

        parts: list[str] = []
        for item in top[:8]:
            name = item["feature"]
            val = item["shap_value"]
            pct = item.get("magnitude_pct", abs(val) * 100)
            direction = item["direction"]
            verb = "increases" if direction == "increase" else "decreases"

            abs_val = abs(val)
            if abs_val > 0.1:
                magnitude = f"by {pct:.0f}%"
            elif abs_val > 0.03:
                magnitude = f"slightly {verb} dose"
            else:
                magnitude = f"has minimal effect"

            parts.append(
                f"{name} (SHAP: {val:+.4f}) {verb} recommended dose {magnitude}."
            )

        return " ".join(parts)
