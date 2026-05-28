"""Synthetic patient data generator for the ADR-20K drug dosage recommendation system.

Produces realistic mock data with clinically plausible correlations across
demographics, labs, vitals, medical history, and genomic markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data.feature_config import COHORTS, FeatureConfig

# Population prevalence estimates for CYP phenotypes
CYP2D6_PREVALENCE: dict[str, float] = {
    "poor": 0.07,
    "intermediate": 0.12,
    "extensive": 0.75,
    "ultrarapid": 0.06,
}

CYP2C19_PREVALENCE: dict[str, float] = {
    "poor": 0.03,
    "intermediate": 0.25,
    "extensive": 0.46,
    "ultrarapid": 0.26,
}

CYP2C9_PREVALENCE: dict[str, float] = {
    "poor": 0.05,
    "intermediate": 0.35,
    "extensive": 0.60,
}

SLCO1B1_PREVALENCE: dict[str, float] = {"TT": 0.70, "TC": 0.26, "CC": 0.04}
VKORC1_PREVALENCE: dict[str, float] = {"GG": 0.37, "GA": 0.47, "AA": 0.16}

DEFAULT_COHORT_DISTRIBUTION: dict[str, int] = {
    "hypertension": 6000,
    "diabetes_mellitus": 5000,
    "oncology": 4000,
    "renal_impairment": 5000,
}


class SyntheticDataGenerator:
    """Generate synthetic patient records with realistic clinical correlations.

    Parameters
    ----------
    n_patients : int
        Total number of synthetic patients to generate (default 20 000).
    random_seed : int
        Seed for reproducible random number generation.
    cohort_distribution : dict[str, int] | None
        Mapping of cohort name to count. Must sum to ``n_patients``.
        Defaults to hypertension=6000, diabetes_mellitus=5000,
        oncology=4000, renal_impairment=5000.
    """

    def __init__(
        self,
        n_patients: int = 20000,
        random_seed: int = 42,
        cohort_distribution: Optional[dict[str, int]] = None,
    ) -> None:
        self.n_patients = n_patients
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        self.cfg = FeatureConfig()

        if cohort_distribution is not None:
            dist = dict(cohort_distribution)
        else:
            total_default = sum(DEFAULT_COHORT_DISTRIBUTION.values())
            dist = {
                k: round(v / total_default * n_patients)
                for k, v in DEFAULT_COHORT_DISTRIBUTION.items()
            }
            diff = n_patients - sum(dist.values())
            dist[list(dist.keys())[0]] += diff
        _validate_distribution(dist, n_patients)
        self.cohort_distribution = dist

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic dataset.

        Returns
        -------
        pd.DataFrame
            DataFrame with all feature columns defined in FeatureConfig plus a
            ``cohort`` column.  Both targets (``dosage_mg_day``,
            ``adr_risk``) are included.
        """
        cohort_arrays = self._build_cohort_array()
        df = self._generate_features(cohort_arrays)
        df = self._generate_targets(df)
        df = self._apply_dtypes(df)
        return df

    def save(self, path: str | Path) -> None:
        """Save the generated dataset to disk.

        The format is inferred from the file extension:
        ``.parquet`` → Parquet, ``.csv`` → CSV, otherwise CSV.
        """
        df = self.generate()
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_cohort_array(self) -> dict[str, np.ndarray]:
        """Return per-cohort boolean index arrays."""
        idx = 0
        arrays: dict[str, np.ndarray] = {}
        all_indices = self.rng.permutation(self.n_patients)
        for cohort_name in COHORTS:
            count = self.cohort_distribution.get(cohort_name, 0)
            positions = all_indices[idx : idx + count]
            arr = np.zeros(self.n_patients, dtype=bool)
            arr[positions] = True
            arrays[cohort_name] = arr
            idx += count
        return arrays

    def _generate_features(
        self, cohort: dict[str, np.ndarray]
    ) -> pd.DataFrame:
        rng = self.rng
        n = self.n_patients

        # ── Demographics ──────────────────────────────────────────────────
        age = np.clip(rng.normal(55, 15, n), 18, 90).astype(np.float64)
        sex = rng.choice(["M", "F"], size=n, p=[0.48, 0.52])
        height_cm = np.clip(
            rng.normal(np.where(sex == "M", 175, 162), 8, n), 140, 200
        ).astype(np.float64)
        weight_kg = np.clip(
            rng.normal(np.where(sex == "M", 82, 68), 15, n), 40, 150
        ).astype(np.float64)
        bmi = weight_kg / ((height_cm / 100) ** 2)

        # ── Medical History ───────────────────────────────────────────────
        # Cohort-driven conditions
        hypertension_dx = np.where(
            cohort["hypertension"], 1, rng.binomial(1, 0.35, n)
        )
        diabetes_dx = np.where(
            cohort["diabetes_mellitus"], 1, rng.binomial(1, 0.15, n)
        )

        cancer_types = ["none", "breast", "lung", "colorectal", "prostate"]
        cancer_probs = [0.90, 0.025, 0.025, 0.025, 0.025]
        cancer_type = np.where(
            cohort["oncology"],
            rng.choice(cancer_types[1:], size=n),
            rng.choice(cancer_types, size=n, p=cancer_probs),
        )

        renal_stage_base = rng.choice([0, 1, 2, 3, 4], size=n, p=[0.60, 0.20, 0.10, 0.07, 0.03])
        renal_stage = np.where(
            cohort["renal_impairment"],
            rng.choice([2, 3, 4], size=n, p=[0.30, 0.40, 0.30]),
            renal_stage_base,
        )

        prior_adr_base = np.clip(rng.poisson(1.0, n), 0, 10)
        prior_adr_count = np.where(
            cohort["oncology"],
            np.clip(rng.poisson(3.5, n), 0, 10),  # oncology → more prior ADRs
            prior_adr_base,
        )

        comorbidity_count = np.clip(
            rng.poisson(1.8, n), 0, 8
        ).astype(np.int64)

        # ── Labs (with clinical correlations) ─────────────────────────────

        # GFR: declines with age, lower in renal impairment
        gfr_expected = 110 - 0.8 * (age - 18)
        gfr_expected = np.where(renal_stage >= 2, gfr_expected * 0.55, gfr_expected)
        gfr_expected = np.where(renal_stage >= 3, gfr_expected * 0.45, gfr_expected)
        gfr_ml_min = np.clip(rng.normal(gfr_expected, 15), 15, 120).astype(np.float64)

        # Creatinine: inverse to GFR, higher in renal disease
        creatinine_expected = 1.2 + (120 - gfr_ml_min) * 0.06
        creatinine_mg_dl = np.clip(
            rng.normal(creatinine_expected, 0.8), 0.5, 10.0
        ).astype(np.float64)

        alt_u_l = np.clip(rng.normal(30, 15, n), 10, 200).astype(np.float64)
        ast_u_l = np.clip(rng.normal(28, 14, n), 10, 200).astype(np.float64)

        # HbA1c: elevated in diabetes
        hba1c_expected = np.where(diabetes_dx == 1, 8.2, 5.4)
        hba1c_percent = np.clip(
            rng.normal(hba1c_expected, np.where(diabetes_dx == 1, 1.5, 0.35)),
            4.0,
            14.0,
        ).astype(np.float64)

        # WBC & platelets: lower in oncology (chemo effect)
        wbc_expected = np.where(cohort["oncology"], 4.5, 7.5)
        wbc_count = np.clip(
            rng.normal(wbc_expected, 2.5), 2.0, 20.0
        ).astype(np.float64)

        plt_expected = np.where(cohort["oncology"], 180, 260)
        platelet_count = np.clip(
            rng.normal(plt_expected, 65), 50, 500
        ).astype(np.float64)

        # ── Vitals (with clinical correlations) ───────────────────────────

        # Systolic BP rises with age, higher in hypertension cohort
        sbp_expected = 110 + 0.65 * (age - 18)
        sbp_expected = np.where(cohort["hypertension"], sbp_expected + 20, sbp_expected)
        systolic_bp_mmhg = np.clip(
            rng.normal(sbp_expected, 14), 90, 200
        ).astype(np.float64)

        diastolic_expected = 70 + 0.15 * (age - 18)
        diastolic_expected = np.where(
            cohort["hypertension"], diastolic_expected + 8, diastolic_expected
        )
        diastolic_bp_mmhg = np.clip(
            rng.normal(diastolic_expected, 9), 50, 120
        ).astype(np.float64)

        heart_rate_bpm = np.clip(
            rng.normal(75, 11, n), 50, 120
        ).astype(np.float64)

        temperature_c = np.clip(
            rng.normal(36.8, 0.4, n), 36.0, 40.0
        ).astype(np.float64)

        # ── Genomic Markers ───────────────────────────────────────────────
        cyp2d6 = _sample_categorical(
            rng, n, list(CYP2D6_PREVALENCE.keys()), list(CYP2D6_PREVALENCE.values())
        )
        cyp2c19 = _sample_categorical(
            rng, n, list(CYP2C19_PREVALENCE.keys()), list(CYP2C19_PREVALENCE.values())
        )
        cyp2c9 = _sample_categorical(
            rng, n, list(CYP2C9_PREVALENCE.keys()), list(CYP2C9_PREVALENCE.values())
        )
        slco1b1_rs4149056 = _sample_categorical(
            rng, n, list(SLCO1B1_PREVALENCE.keys()), list(SLCO1B1_PREVALENCE.values())
        )
        vkorc1_rs9923231 = _sample_categorical(
            rng, n, list(VKORC1_PREVALENCE.keys()), list(VKORC1_PREVALENCE.values())
        )

        # ── Assemble cohort column ────────────────────────────────────────
        cohort_col = np.full(n, "hypertension", dtype=object)
        for name in COHORTS:
            cohort_col[cohort[name]] = name

        # ── Build DataFrame ───────────────────────────────────────────────
        data: dict[str, np.ndarray] = {
            "age": age,
            "sex": sex,
            "weight_kg": weight_kg,
            "height_cm": height_cm,
            "bmi": bmi,
            "hypertension_diagnosis": hypertension_dx,
            "diabetes_diagnosis": diabetes_dx,
            "cancer_type": cancer_type,
            "renal_disease_stage": renal_stage,
            "prior_adr_count": prior_adr_count,
            "comorbidity_count": comorbidity_count,
            "gfr_ml_min": gfr_ml_min,
            "creatinine_mg_dl": creatinine_mg_dl,
            "alt_u_l": alt_u_l,
            "ast_u_l": ast_u_l,
            "hba1c_percent": hba1c_percent,
            "wbc_count": wbc_count,
            "platelet_count": platelet_count,
            "systolic_bp_mmhg": systolic_bp_mmhg,
            "diastolic_bp_mmhg": diastolic_bp_mmhg,
            "heart_rate_bpm": heart_rate_bpm,
            "temperature_c": temperature_c,
            "cyp2d6_phenotype": cyp2d6,
            "cyp2c19_phenotype": cyp2c19,
            "cyp2c9_phenotype": cyp2c9,
            "slco1b1_rs4149056": slco1b1_rs4149056,
            "vkorc1_rs9923231": vkorc1_rs9923231,
            "cohort": cohort_col,
        }
        return pd.DataFrame(data)

    def _generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add dosage_mg_day and adr_risk columns."""

        # ── Dosage ────────────────────────────────────────────────────────
        # Base dosage adjusted by key clinical features
        gfr_factor = 0.3 * (df["gfr_ml_min"] - 80) / 40
        weight_factor = 0.2 * (df["weight_kg"] - 70) / 30
        age_factor = -0.1 * (df["age"] - 55) / 20

        # CYP phenotype impact on drug metabolism → dosage adjustment
        cyp2d6_map = {"poor": -15, "intermediate": -5, "extensive": 0, "ultrarapid": 15}
        cyp2c19_map = {"poor": -10, "intermediate": -3, "extensive": 0, "ultrarapid": 10}
        cyp2c9_map = {"poor": -12, "intermediate": -4, "extensive": 0}

        cyp2d6_effect = df["cyp2d6_phenotype"].map(cyp2d6_map).fillna(0).values
        cyp2c19_effect = df["cyp2c19_phenotype"].map(cyp2c19_map).fillna(0).values
        cyp2c9_effect = df["cyp2c9_phenotype"].map(cyp2c9_map).fillna(0).values

        base_dosage = 100.0
        dosage = (
            base_dosage
            + gfr_factor * 40
            + weight_factor * 30
            + age_factor * 20
            + cyp2d6_effect
            + cyp2c19_effect
            + cyp2c9_effect
        )
        dosage = dosage + self.rng.normal(0, 12, self.n_patients)
        dosage_mg_day = np.clip(dosage, 10, 500).astype(np.float64)

        # ── ADR Risk ──────────────────────────────────────────────────────
        # Linear combination of risk features
        risk_score = -0.72 + (
            0.02 * (df["age"] - 55) / 20
            + 0.55 * df["renal_disease_stage"] / 4
            + 0.12 * df["prior_adr_count"] / 5
            + 0.18 * (120 - df["gfr_ml_min"]) / 60
            + 0.10 * df["comorbidity_count"] / 4
        )

        cyp2d6_risk = df["cyp2d6_phenotype"].map(
            {"poor": 0.30, "intermediate": 0.10, "extensive": 0.0, "ultrarapid": -0.08}
        ).fillna(0).values
        cyp2c19_risk = df["cyp2c19_phenotype"].map(
            {"poor": 0.22, "intermediate": 0.07, "extensive": 0.0, "ultrarapid": -0.05}
        ).fillna(0).values
        cyp2c9_risk = df["cyp2c9_phenotype"].map(
            {"poor": 0.22, "intermediate": 0.07, "extensive": 0.0}
        ).fillna(0).values

        risk_score += cyp2d6_risk + cyp2c19_risk + cyp2c9_risk
        risk_score += self.rng.normal(0, 0.25, self.n_patients)
        adr_prob = 1.0 / (1.0 + np.exp(-risk_score))
        adr_risk = (adr_prob >= 0.5).astype(np.int64)

        df["dosage_mg_day"] = dosage_mg_day
        df["adr_risk"] = adr_risk
        return df

    def _apply_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast columns to the dtypes declared in FeatureConfig."""
        for col in df.columns:
            meta = self.cfg.feature_meta.get(col)
            if meta is None:
                continue
            dtype = meta["dtype"]
            if dtype == "category":
                df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype(dtype)
        return df


# ── Module helpers ──────────────────────────────────────────────────────────


def _validate_distribution(dist: dict[str, int], expected_total: int) -> None:
    """Ensure cohort counts sum to expected_total."""
    total = sum(dist.values())
    if total != expected_total:
        raise ValueError(
            f"Cohort distribution sums to {total}, expected {expected_total}"
        )
    unknown = set(dist) - set(COHORTS)
    if unknown:
        raise ValueError(f"Unknown cohort(s): {unknown}")


def _sample_categorical(
    rng: np.random.Generator,
    n: int,
    categories: list[str],
    probs: list[float],
) -> np.ndarray:
    """Draw n samples from categorical distribution."""
    return rng.choice(categories, size=n, p=np.array(probs))


# ── Quick smoke-test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    gen = SyntheticDataGenerator(n_patients=100, random_seed=42)
    data = gen.generate()
    print(data.shape)
    print(data.dtypes)
    print(data.head(3))
    print("\nCohort distribution:")
    print(data["cohort"].value_counts())
    print("\nADR risk prevalence:", data["adr_risk"].mean().round(3))
