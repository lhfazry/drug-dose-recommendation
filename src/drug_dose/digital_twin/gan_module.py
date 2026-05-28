"""GAN module for synthetic patient data generation (Phase 3).

Generates realistic synthetic patient feature vectors conditioned on dense
embeddings from the TransformerEncoder, enabling data augmentation for
dose recommendation training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatientGANGenerator(nn.Module):
    """GAN Generator: E_input + noise → X_synthetic (realistic patient data).

    Takes the dense embedding E_input (from TransformerEncoder) and a noise vector,
    generates synthetic patient feature vectors that mirror real-world variability.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        noise_dim: int = 64,
        feature_dim: int = 20,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.noise_dim = noise_dim

        self.input_proj = nn.Linear(embed_dim + noise_dim, hidden_dim)
        self.cond_proj = nn.Linear(embed_dim, hidden_dim)

        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.output = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self, e_input: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Generate synthetic patient features conditioned on E_input.

        Args:
            e_input: Embedding tensor of shape (batch, embed_dim).
            noise: Optional noise tensor of shape (batch, noise_dim).
                   Sampled from N(0,1) if not provided.

        Returns:
            X_synthetic of shape (batch, feature_dim) in [-1, 1].
        """
        batch_size = e_input.size(0)

        if noise is None:
            noise = torch.randn(
                batch_size, self.noise_dim, device=e_input.device
            )

        cond = self.cond_proj(e_input)

        x = torch.cat([e_input, noise], dim=-1)
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)
        x = x + cond

        x = self.hidden1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + cond

        x = self.hidden2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + cond

        x = self.output(x)
        x = torch.tanh(x)

        return x


class PatientGANDiscriminator(nn.Module):
    """GAN Discriminator: distinguishes real vs synthetic patient data.

    Takes patient feature vectors and determines if they are real or generated.
    """

    def __init__(
        self,
        feature_dim: int = 20,
        embed_dim: int = 256,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.input_layer = nn.Linear(feature_dim + embed_dim, hidden_dim)

        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)

        self.hidden2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)

        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor, e_input: torch.Tensor) -> torch.Tensor:
        """Classify patient features as real or fake.

        Args:
            x: Patient feature tensor of shape (batch, feature_dim).
            e_input: Conditioning embedding of shape (batch, embed_dim).

        Returns:
            Probability tensor of shape (batch, 1).  1 = real, 0 = fake.
        """
        combined = torch.cat([x, e_input], dim=-1)

        out = F.leaky_relu(self.input_layer(combined), negative_slope=0.2)
        out = self.dropout1(out)

        out = F.leaky_relu(self.hidden1(out), negative_slope=0.2)
        out = self.dropout1(out)

        out = F.leaky_relu(self.hidden2(out), negative_slope=0.2)
        out = self.dropout2(out)

        out = torch.sigmoid(self.output(out))

        return out


class PatientGAN:
    """GAN trainer wrapping Generator + Discriminator.

    Handles training loop with adversarial loss and feature matching.
    """

    def __init__(
        self,
        generator: Optional[PatientGANGenerator] = None,
        discriminator: Optional[PatientGANDiscriminator] = None,
        feature_dim: int = 20,
        embed_dim: int = 256,
        noise_dim: int = 64,
        device: str = "cpu",
        lr: float = 0.0002,
        betas: tuple[float, float] = (0.5, 0.999),
        fm_lambda: float = 0.1,
    ) -> None:
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim
        self.device = torch.device(device)
        self.fm_lambda = fm_lambda

        self.generator = generator or PatientGANGenerator(
            embed_dim=embed_dim,
            noise_dim=noise_dim,
            feature_dim=feature_dim,
        ).to(self.device)

        self.discriminator = discriminator or PatientGANDiscriminator(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
        ).to(self.device)

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr, betas=betas
        )

        self.criterion = nn.BCELoss()
        self._train_mode = False

    def train(self) -> None:
        self._train_mode = True
        self.generator.train()
        self.discriminator.train()

    def eval(self) -> None:
        self._train_mode = False
        self.generator.eval()
        self.discriminator.eval()

    def train_step(
        self, real_data: torch.Tensor, e_input: torch.Tensor
    ) -> dict[str, float]:
        """Single GAN training step.

        Args:
            real_data: Real patient features of shape (batch, feature_dim).
            e_input: Dense embeddings of shape (batch, embed_dim).

        Returns:
            Dictionary with loss and accuracy metrics.
        """
        batch_size = real_data.size(0)
        real_data = real_data.to(self.device)
        e_input = e_input.to(self.device).detach()

        ones = torch.ones(batch_size, 1, device=self.device)
        zeros = torch.zeros(batch_size, 1, device=self.device)

        # ---- Discriminator step ----
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake = self.generator(e_input, noise).detach()

        d_real = self.discriminator(real_data, e_input)
        d_fake = self.discriminator(fake, e_input)

        d_real_loss = self.criterion(d_real, ones)
        d_fake_loss = self.criterion(d_fake, zeros)
        d_loss = (d_real_loss + d_fake_loss) * 0.5

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        d_real_acc = (d_real >= 0.5).float().mean().item()
        d_fake_acc = (d_fake < 0.5).float().mean().item()

        # ---- Generator step ----
        noise = torch.randn(batch_size, self.noise_dim, device=self.device)
        fake = self.generator(e_input, noise)

        d_fake_g = self.discriminator(fake, e_input)
        g_adv_loss = self.criterion(d_fake_g, ones)

        # Feature matching loss: match batch statistics (mean, std)
        fm_loss = torch.tensor(0.0, device=self.device)
        if self.fm_lambda > 0:
            fm_loss = F.mse_loss(fake.mean(dim=0), real_data.mean(dim=0)) + F.mse_loss(
                fake.std(dim=0), real_data.std(dim=0)
            )

        g_loss = g_adv_loss + self.fm_lambda * fm_loss

        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item(),
            "d_real_acc": d_real_acc,
            "d_fake_acc": d_fake_acc,
        }

    def generate(
        self, e_input: torch.Tensor, n_samples: Optional[int] = None
    ) -> torch.Tensor:
        """Generate synthetic patient data conditioned on E_input.

        Args:
            e_input: Embeddings of shape (batch, embed_dim).
            n_samples: If provided, repeat each e_input n_samples times
                       to generate multiple variations per patient.

        Returns:
            X_synthetic of shape (batch * n_samples, feature_dim) in [-1, 1].
        """
        was_training = self._train_mode
        self.eval()

        if n_samples is not None:
            e_input = e_input.repeat_interleave(n_samples, dim=0)

        e_input = e_input.to(self.device)

        with torch.no_grad():
            generated = self.generator(e_input)

        if was_training:
            self.train()

        return generated

    def to(self, device: str) -> "PatientGAN":
        self.device = torch.device(device)
        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        return self
