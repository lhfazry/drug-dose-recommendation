"""Transformer-based RL policy network, value estimator, and dosage environment.

Phase 4: Reinforcement learning components for the drug dosage recommendation system.
Policy π(at|E_enhanced) = Gaussian(TransformerPolicy(E_enhanced)).
Value V(E_enhanced) = ValueEstimator(E_enhanced).
Environment simulates clinical outcomes with dosage error and ADR risk rewards.
"""

import math

import torch
import torch.nn as nn
from torch.distributions import Normal


class TransformerPolicy(nn.Module):
    """Gaussian policy: E_enhanced → action distribution over dosage [action_min, action_max].

    Architecture: 3-layer MLP with residual connections, LayerNorm, ReLU, Dropout.
    Outputs mean (scaled from [-1,1] to [action_min, action_max]) and learned log_std.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        action_dim: int = 1,
        action_min: float = 10.0,
        action_max: float = 500.0,
    ):
        super().__init__()
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = (action_max - action_min) / 2.0
        self.action_center = (action_max + action_min) / 2.0

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.dropout3 = nn.Dropout(0.1)

        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, e_enhanced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.

        Args:
            e_enhanced: Enhanced patient embedding (batch, input_dim).

        Returns:
            mean: Scaled dosage mean in [action_min, action_max] (batch, action_dim).
            std: Gaussian std clamped to [0.01, 1.0] (action_dim,).
        """
        x = self.dropout1(torch.relu(self.ln1(self.linear1(e_enhanced))))
        x = x + self.dropout2(torch.relu(self.ln2(self.linear2(x))))
        x = x + self.dropout3(torch.relu(self.ln3(self.linear3(x))))

        mean = torch.tanh(self.mean_head(x))
        mean = mean * self.action_range + self.action_center
        std = torch.exp(self.log_std).clamp(0.01, 1.0)
        return mean, std

    def sample(
        self, e_enhanced: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from the policy via reparameterization.

        Args:
            e_enhanced: Enhanced patient embedding (batch, input_dim).

        Returns:
            action: Clamped dosage in [action_min, action_max] (batch, action_dim).
            log_prob: Log probability of the sampled action (batch,).
        """
        mean, std = self.forward(e_enhanced)
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.clamp(action, self.action_min, self.action_max)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(
        self, e_enhanced: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate log probability of a given action under the policy.

        Args:
            e_enhanced: Enhanced patient embedding (batch, input_dim).
            action: Dosage action tensor (batch, action_dim).

        Returns:
            log_prob: Log probability for each sample (batch,).
        """
        mean, std = self.forward(e_enhanced)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob


class ValueEstimator(nn.Module):
    """Estimates state value V(E_enhanced) for advantage computation.

    Architecture: 3-layer MLP (input → hidden → hidden → 1).
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, e_enhanced: torch.Tensor) -> torch.Tensor:
        """Compute state-value estimate.

        Args:
            e_enhanced: Enhanced patient embedding (batch, input_dim).

        Returns:
            value: Estimated V(E_enhanced) (batch, 1).
        """
        x = torch.relu(self.linear1(e_enhanced))
        x = torch.relu(self.linear2(x))
        value = self.value_head(x)
        return value


class DosageEnvironment:
    """Simulates clinical outcomes for single-step dosage decisions.

    Reward = -(dosage_error_weight * |predicted - optimal| / 500)
             - (adr_penalty * ADR_risk)

    ADR_risk uses a sigmoid model based on dosage, GFR, and comorbidity count,
    with CYP poor metabolizers receiving a 2x risk multiplier.
    """

    def __init__(
        self,
        dosage_error_weight: float = 1.0,
        adr_penalty: float = 5.0,
    ):
        self.dosage_error_weight = dosage_error_weight
        self.adr_penalty = adr_penalty
        self._state: torch.Tensor | None = None
        self._optimal_dosage: float | None = None
        self._patient_features: dict | None = None

    def reset(
        self,
        e_enhanced: torch.Tensor,
        optimal_dosage: float,
        patient_features: dict,
    ) -> torch.Tensor:
        """Reset environment with a new patient.

        Args:
            e_enhanced: Patient embedding tensor.
            optimal_dosage: Ground truth dosage in mg/day.
            patient_features: Dict with keys gfr_ml_min, comorbidity_count, cyp_poor.

        Returns:
            e_enhanced: Initial state observation.
        """
        self._state = e_enhanced
        self._optimal_dosage = optimal_dosage
        self._patient_features = patient_features
        return e_enhanced

    def step(self, action) -> tuple[float, bool, dict]:
        """Execute dosage decision and compute reward.

        Args:
            action: Predicted dosage in mg/day (scalar, 0-d tensor, or 1-element array).

        Returns:
            reward: Scalar reward (higher is better).
            done: Always True (single-step episodic).
            info: Dict with keys dosage_error, adr_risk, optimal_dosage.
        """
        predicted = self._extract_scalar(action)
        optimal = float(self._optimal_dosage)
        dosage_error = abs(predicted - optimal)

        adr_risk = self._estimate_adr_risk(self._patient_features, predicted)

        reward = (
            -self.dosage_error_weight * (dosage_error / 500.0)
            - self.adr_penalty * adr_risk
        )

        info = {
            "dosage_error": dosage_error,
            "adr_risk": adr_risk,
            "optimal_dosage": optimal,
        }
        return float(reward), True, info

    @staticmethod
    def compute_rewards(
        actions: "torch.Tensor",
        optimal_dosages: "torch.Tensor",
        patient_features_list: list[dict],
        dosage_error_weight: float = 1.0,
        adr_penalty: float = 5.0,
    ) -> "torch.Tensor":
        """Compute batched rewards without storing state.
        
        Args:
            actions: Predicted dosages (batch, 1) or (batch,).
            optimal_dosages: Ground truth (batch, 1) or (batch,).
            patient_features_list: List of patient feature dicts, one per batch item.
            
        Returns:
            reward: Tensor of shape (batch,).
        """
        import torch
        
        actions = actions.view(-1)
        optimal_dosages = optimal_dosages.view(-1)
        dosage_errors = (actions - optimal_dosages).abs()
        
        adr_risks = []
        for i, pf in enumerate(patient_features_list):
            risk = DosageEnvironment._estimate_adr_risk(pf, actions[i].item())
            adr_risks.append(risk)
        adr_risks_t = torch.tensor(adr_risks, dtype=torch.float32, device=actions.device)
        
        reward = (
            -dosage_error_weight * (dosage_errors / 500.0)
            - adr_penalty * adr_risks_t
        )
        return reward

    @staticmethod
    def _estimate_adr_risk(
        patient_features: dict, dosage: float
    ) -> float:
        """Estimate ADR risk via pharmacological sigmoid model.

        ADR_risk = sigmoid(b0 + b1*dosage/500 + b2*(1-GFR/120) + b3*comorbidity/8)
        CYP poor metabolizers: 2x multiplier (capped at 1.0).
        """
        b0, b1, b2, b3 = -2.0, 2.0, 1.5, 1.5

        gfr = float(patient_features.get("gfr_ml_min", 80.0))
        comorbidity = float(patient_features.get("comorbidity_count", 0))
        cyp_poor = bool(patient_features.get("cyp_poor", False))

        score = (
            b0
            + b1 * (dosage / 500.0)
            + b2 * (1.0 - gfr / 120.0)
            + b3 * (comorbidity / 8.0)
        )
        risk = 1.0 / (1.0 + math.exp(-score))

        if cyp_poor:
            risk = min(risk * 2.0, 1.0)

        return risk

    @staticmethod
    def _extract_scalar(action) -> float:
        """Extract a Python float from tensor, numpy array, or scalar."""
        if isinstance(action, torch.Tensor):
            return float(action.detach().cpu().item())
        if hasattr(action, "item"):
            return float(action.item())
        return float(action)
