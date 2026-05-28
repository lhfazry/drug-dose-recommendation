"""Feature configuration for the ADR-20K drug dosage recommendation system.

Single source of truth for all feature definitions, organized into the groups
described in Table 2 of the paper.
"""

from dataclasses import dataclass, field
from typing import Any

# ── Cohort definitions ──────────────────────────────────────────────────────

COHORTS: list[str] = [
    "hypertension",
    "diabetes_mellitus",
    "oncology",
    "renal_impairment",
]

# ── Feature group definitions ───────────────────────────────────────────────

FEATURE_GROUPS: dict[str, list[str]] = {
    "demographics": [
        "age",
        "sex",
        "weight_kg",
        "height_cm",
        "bmi",
    ],
    "medical_history": [
        "hypertension_diagnosis",
        "diabetes_diagnosis",
        "cancer_type",
        "renal_disease_stage",
        "prior_adr_count",
        "comorbidity_count",
    ],
    "labs": [
        "gfr_ml_min",
        "creatinine_mg_dl",
        "alt_u_l",
        "ast_u_l",
        "hba1c_percent",
        "wbc_count",
        "platelet_count",
    ],
    "vitals": [
        "systolic_bp_mmhg",
        "diastolic_bp_mmhg",
        "heart_rate_bpm",
        "temperature_c",
    ],
    "genomic_markers": [
        "cyp2d6_phenotype",
        "cyp2c19_phenotype",
        "cyp2c9_phenotype",
        "slco1b1_rs4149056",
        "vkorc1_rs9923231",
    ],
    "targets": [
        "dosage_mg_day",
        "adr_risk",
    ],
}

# ── Detailed feature metadata ───────────────────────────────────────────────

FEATURE_META: dict[str, dict[str, Any]] = {
    # ── Demographics ──────────────────────────────────────────────────────
    "age": {
        "dtype": "float64",
        "value_range": (18, 90),
        "description": "Patient age in years",
        "feature_group": "demographics",
    },
    "sex": {
        "dtype": "category",
        "value_range": ("M", "F"),
        "description": "Biological sex",
        "feature_group": "demographics",
    },
    "weight_kg": {
        "dtype": "float64",
        "value_range": (40, 150),
        "description": "Body weight in kilograms",
        "feature_group": "demographics",
    },
    "height_cm": {
        "dtype": "float64",
        "value_range": (140, 200),
        "description": "Height in centimeters",
        "feature_group": "demographics",
    },
    "bmi": {
        "dtype": "float64",
        "value_range": (12, 55),
        "description": "Body mass index (derived from weight_kg / (height_cm/100)^2)",
        "feature_group": "demographics",
    },
    # ── Medical History ────────────────────────────────────────────────────
    "hypertension_diagnosis": {
        "dtype": "int64",
        "value_range": (0, 1),
        "description": "Diagnosed hypertension (0=No, 1=Yes)",
        "feature_group": "medical_history",
    },
    "diabetes_diagnosis": {
        "dtype": "int64",
        "value_range": (0, 1),
        "description": "Diagnosed diabetes mellitus (0=No, 1=Yes)",
        "feature_group": "medical_history",
    },
    "cancer_type": {
        "dtype": "category",
        "value_range": ("none", "breast", "lung", "colorectal", "prostate"),
        "description": "Cancer diagnosis type, 'none' if not applicable",
        "feature_group": "medical_history",
    },
    "renal_disease_stage": {
        "dtype": "int64",
        "value_range": (0, 4),
        "description": "Chronic kidney disease stage (0=normal through 4=ESRD)",
        "feature_group": "medical_history",
    },
    "prior_adr_count": {
        "dtype": "int64",
        "value_range": (0, 10),
        "description": "Number of prior adverse drug reactions",
        "feature_group": "medical_history",
    },
    "comorbidity_count": {
        "dtype": "int64",
        "value_range": (0, 8),
        "description": "Total number of comorbidities",
        "feature_group": "medical_history",
    },
    # ── Laboratory Values ──────────────────────────────────────────────────
    "gfr_ml_min": {
        "dtype": "float64",
        "value_range": (15, 120),
        "description": "Glomerular filtration rate in mL/min",
        "feature_group": "labs",
    },
    "creatinine_mg_dl": {
        "dtype": "float64",
        "value_range": (0.5, 10.0),
        "description": "Serum creatinine in mg/dL",
        "feature_group": "labs",
    },
    "alt_u_l": {
        "dtype": "float64",
        "value_range": (10, 200),
        "description": "Alanine aminotransferase in U/L",
        "feature_group": "labs",
    },
    "ast_u_l": {
        "dtype": "float64",
        "value_range": (10, 200),
        "description": "Aspartate aminotransferase in U/L",
        "feature_group": "labs",
    },
    "hba1c_percent": {
        "dtype": "float64",
        "value_range": (4.0, 14.0),
        "description": "Glycated hemoglobin A1c in percent",
        "feature_group": "labs",
    },
    "wbc_count": {
        "dtype": "float64",
        "value_range": (2.0, 20.0),
        "description": "White blood cell count in 10^9/L",
        "feature_group": "labs",
    },
    "platelet_count": {
        "dtype": "float64",
        "value_range": (50, 500),
        "description": "Platelet count in 10^9/L",
        "feature_group": "labs",
    },
    # ── Vital Signs ──────────────────────────────────────────────────────
    "systolic_bp_mmhg": {
        "dtype": "float64",
        "value_range": (90, 200),
        "description": "Systolic blood pressure in mmHg",
        "feature_group": "vitals",
    },
    "diastolic_bp_mmhg": {
        "dtype": "float64",
        "value_range": (50, 120),
        "description": "Diastolic blood pressure in mmHg",
        "feature_group": "vitals",
    },
    "heart_rate_bpm": {
        "dtype": "float64",
        "value_range": (50, 120),
        "description": "Resting heart rate in beats per minute",
        "feature_group": "vitals",
    },
    "temperature_c": {
        "dtype": "float64",
        "value_range": (36.0, 40.0),
        "description": "Body temperature in degrees Celsius",
        "feature_group": "vitals",
    },
    # ── Genomic Markers ────────────────────────────────────────────────────
    "cyp2d6_phenotype": {
        "dtype": "category",
        "value_range": ("poor", "intermediate", "extensive", "ultrarapid"),
        "description": "CYP2D6 metabolizer phenotype",
        "feature_group": "genomic_markers",
    },
    "cyp2c19_phenotype": {
        "dtype": "category",
        "value_range": ("poor", "intermediate", "extensive", "ultrarapid"),
        "description": "CYP2C19 metabolizer phenotype",
        "feature_group": "genomic_markers",
    },
    "cyp2c9_phenotype": {
        "dtype": "category",
        "value_range": ("poor", "intermediate", "extensive"),
        "description": "CYP2C9 metabolizer phenotype",
        "feature_group": "genomic_markers",
    },
    "slco1b1_rs4149056": {
        "dtype": "category",
        "value_range": ("TT", "TC", "CC"),
        "description": "SLCO1B1 rs4149056 genotype (T=risk allele)",
        "feature_group": "genomic_markers",
    },
    "vkorc1_rs9923231": {
        "dtype": "category",
        "value_range": ("GG", "GA", "AA"),
        "description": "VKORC1 rs9923231 genotype (A=sensitive allele)",
        "feature_group": "genomic_markers",
    },
    # ── Targets ─────────────────────────────────────────────────────────────
    "dosage_mg_day": {
        "dtype": "float64",
        "value_range": (10.0, 500.0),
        "description": "Recommended daily dosage in mg",
        "feature_group": "targets",
    },
    "adr_risk": {
        "dtype": "int64",
        "value_range": (0, 1),
        "description": "Adverse drug reaction risk label (0=Low, 1=High)",
        "feature_group": "targets",
    },
}


@dataclass
class FeatureConfig:
    """Central registry for all feature definitions in the ADR-20K dataset.

    Provides convenience methods to filter features by type and group.
    """

    cohorts: list[str] = field(default_factory=lambda: COHORTS.copy())
    feature_groups: dict[str, list[str]] = field(
        default_factory=lambda: {k: v.copy() for k, v in FEATURE_GROUPS.items()}
    )
    feature_meta: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {k: v.copy() for k, v in FEATURE_META.items()}
    )

    # ── Public API ───────────────────────────────────────────────────────

    def get_numerical_features(self) -> list[str]:
        """Return feature names whose dtype is float64 or int64, excluding targets."""
        targets = set(self.feature_groups.get("targets", []))
        return [
            name
            for name, meta in self.feature_meta.items()
            if meta["dtype"] in ("float64", "int64") and name not in targets
        ]

    def get_categorical_features(self) -> list[str]:
        """Return feature names whose dtype is category, excluding targets."""
        targets = set(self.feature_groups.get("targets", []))
        return [
            name
            for name, meta in self.feature_meta.items()
            if meta["dtype"] == "category" and name not in targets
        ]

    def get_target_features(self) -> list[str]:
        """Return target feature names."""
        return list(self.feature_groups.get("targets", []))

    def get_features_by_group(self, group_name: str) -> list[str]:
        """Return feature names belonging to a specific group."""
        return list(self.feature_groups.get(group_name, []))

    def get_feature_meta(self, feature_name: str) -> dict[str, Any]:
        """Return metadata dict for a single feature."""
        if feature_name not in self.feature_meta:
            raise KeyError(f"Unknown feature: {feature_name}")
        return dict(self.feature_meta[feature_name])

    def all_feature_names(self) -> list[str]:
        """Return every defined feature name (including targets)."""
        return list(self.feature_meta.keys())
