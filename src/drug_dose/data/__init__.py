"""Data generation and feature configuration modules."""

from .feature_config import FeatureConfig, COHORTS, FEATURE_GROUPS
from .synthetic_generator import SyntheticDataGenerator

__all__ = [
    "FeatureConfig",
    "COHORTS",
    "FEATURE_GROUPS",
    "SyntheticDataGenerator",
]
