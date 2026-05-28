"""Patient-specific digital twin model orchestrating RAG, GAN, and encoder.

Implements Algorithm 1 from the paper:
    1. E_input = TransformerEncoder([S_out, KB_out, F_loop])
    2. X_synthetic = GAN_Generator(E_input, noise)
    3. E_enhanced = Concat([E_input, MLP(X_synthetic)])
"""

import torch
import torch.nn as nn


class DigitalTwin(nn.Module):
    """Patient-specific digital twin model.

    Pipeline: E_input → GAN_Generator → X_synthetic → MLP → Concat → E_enhanced
    """

    def __init__(
        self,
        encoder: nn.Module,
        gan_generator: nn.Module,
        feature_dim: int = 20,
        embed_dim: int = 256,
        noise_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.gan_generator = gan_generator
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.noise_dim = noise_dim

        self.synthetic_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(
        self,
        s_out: torch.Tensor,
        kb_out: torch.Tensor,
        f_loop: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        e_input = self.encoder(s_out, kb_out, f_loop)
        batch_size = e_input.size(0)

        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=e_input.device)

        x_synthetic = self.gan_generator(e_input, noise)
        synthetic_encoded = self.synthetic_encoder(x_synthetic)
        e_enhanced = torch.cat([e_input, synthetic_encoded], dim=-1)

        return {
            "e_input": e_input,
            "x_synthetic": x_synthetic,
            "e_enhanced": e_enhanced,
        }
