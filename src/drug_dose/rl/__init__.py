"""RL-based dosage recommendation — policy gradient training and environment."""

from drug_dose.rl.trainer import RLTrainer

try:
    from drug_dose.rl.policy_network import (
        DosageEnvironment,
        TransformerPolicy,
        ValueEstimator,
    )
except ImportError:
    TransformerPolicy = None  # type: ignore[assignment]
    ValueEstimator = None  # type: ignore[assignment]
    DosageEnvironment = None  # type: ignore[assignment]

__all__ = ["TransformerPolicy", "ValueEstimator", "DosageEnvironment", "RLTrainer"]
