import torch
import torch.nn.functional as F

from drug_dose.rl.policy_network import DosageEnvironment


class RLTrainer:

    def __init__(self, policy, value_estimator, env, policy_lr=1e-4, value_lr=3e-4):
        self.policy = policy
        self.value_estimator = value_estimator
        self.env = env
        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.value_optimizer = torch.optim.Adam(value_estimator.parameters(), lr=value_lr)

    def train_step(self, e_enhanced, optimal_dosage, patient_features):
        self.policy.train()
        self.value_estimator.train()

        action, log_prob = self.policy.sample(e_enhanced)
        reward = DosageEnvironment.compute_rewards(
            action, optimal_dosage, patient_features,
            self.env.dosage_error_weight, self.env.adr_penalty,
        ).to(e_enhanced.device)

        value = self.value_estimator(e_enhanced).squeeze()
        advantage = reward - value.detach()

        policy_loss = -(log_prob * advantage).mean()
        value_loss = F.mse_loss(value, reward)

        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.policy_optimizer.step()
        self.value_optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "reward": reward.mean().item(),
            "dosage_error": (action - optimal_dosage).abs().mean().item(),
            "adr_risk": float("nan"),
        }

    def train_epoch(self, e_enhanced_batch, optimal_dosages, patient_features, batch_size=32):
        n = e_enhanced_batch.size(0)
        indices = torch.randperm(n)
        metrics_accum = {}

        for start in range(0, n, batch_size):
            idx = indices[start : start + batch_size]
            batch_e = e_enhanced_batch[idx]
            batch_opt = optimal_dosages[idx]
            batch_pf = [patient_features[i] for i in idx.tolist()]

            m = self.train_step(batch_e, batch_opt, batch_pf)
            for k, v in m.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        n_batches = max(1, (n + batch_size - 1) // batch_size)
        return {k: v / n_batches for k, v in metrics_accum.items()}

    def recommend(self, e_enhanced):
        self.policy.eval()
        self.value_estimator.eval()
        with torch.no_grad():
            mean, _ = self.policy(e_enhanced)
            value = self.value_estimator(e_enhanced)
        conf = 1.0 / (value.std().item() + 1e-6) if value.numel() > 1 else 1.0
        return {"dosage": mean, "confidence": conf, "value": value.mean().item()}
