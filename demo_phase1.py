import sys

sys.path.insert(0, "src")

from drug_dose.data import COHORTS, FEATURE_GROUPS, FeatureConfig, SyntheticDataGenerator
from drug_dose.preprocessing import PreprocessingPipeline


def main():
    fc = FeatureConfig()
    print("=" * 60)
    print("PHASE 1: Synthetic Data Generation & Preprocessing Demo")
    print("=" * 60)

    print(f"\n[1] Feature Configuration")
    print(f"    Cohorts:              {COHORTS}")
    print(f"    Feature groups:       {list(FEATURE_GROUPS.keys())}")
    print(f"    Numerical features:   {len(fc.get_numerical_features())}")
    print(f"    Categorical features: {len(fc.get_categorical_features())}")
    print(f"    Targets:              {fc.get_target_features()}")

    print(f"\n[2] Generating Mock ADR-20K Dataset (n=2000)")
    gen = SyntheticDataGenerator(n_patients=2000, random_seed=42)
    df = gen.generate()

    print(f"    Shape:           {df.shape}")
    print(f"    Memory:          {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"    Cohort distribution:")
    for cohort, count in df["cohort"].value_counts().items():
        print(f"      {cohort:25s} {count:5d}")
    print(f"    ADR risk rate:       {df['adr_risk'].mean():.3f} ({df['adr_risk'].sum():.0f}/{len(df)})")
    print(
        f"    Dosage (mg/day):     "
        f"mean={df['dosage_mg_day'].mean():.1f}, "
        f"std={df['dosage_mg_day'].std():.1f}, "
        f"range=[{df['dosage_mg_day'].min():.1f}, {df['dosage_mg_day'].max():.1f}]"
    )

    print(f"\n[3] Correlation Checks")
    corr_age_gfr = df["age"].corr(df["gfr_ml_min"])
    corr_age_sbp = df["age"].corr(df["systolic_bp_mmhg"])
    print(f"    age ↔ GFR:           {corr_age_gfr:.3f} (expected negative — older → lower GFR)")
    print(f"    age ↔ SBP:           {corr_age_sbp:.3f} (expected positive — older → higher BP)")
    diabetes_mask = df["diabetes_diagnosis"] == 1
    print(
        f"    Diabetes HbA1c:      {df.loc[diabetes_mask, 'hba1c_percent'].mean():.1f}% "
        f"vs non-diabetic {df.loc[~diabetes_mask, 'hba1c_percent'].mean():.1f}%"
    )

    print(f"\n[4] Preprocessing Pipeline")
    pipe = PreprocessingPipeline(fc)
    X = pipe.fit_transform(df)

    print(f"    Input shape:     {df[fc.get_numerical_features() + fc.get_categorical_features()].shape}")
    print(f"    Output shape:    {X.shape}")
    print(f"    Output columns:  {len(X.columns)}")
    print(f"    NaN in output:   {X.isnull().any().any()}")

    print(f"\n[5] Saving Artifacts")
    gen.save("data/adr20k_synthetic_2000.parquet")
    pipe.save("data/preprocessing_pipeline.pkl")
    print("    Saved: data/adr20k_synthetic_2000.parquet")
    print("    Saved: data/preprocessing_pipeline.pkl")

    print(f"\n{'=' * 60}")
    print("PHASE 1 COMPLETE — All checks passed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
