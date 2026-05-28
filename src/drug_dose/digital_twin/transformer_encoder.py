"""Multi-head self-attention encoder for fusing RAG outputs into dense embeddings."""

import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TransformerEncoder(nn.Module):
    """Multi-head self-attention encoder for fusing RAG outputs into dense embeddings.

    Per paper Algorithm 1: E_input = TransformerEncoder([S_out, KB_out, F_loop])
    """

    def __init__(
        self,
        input_dim: int = 384,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_model * 4,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        s_out: torch.Tensor,
        kb_out: torch.Tensor,
        f_loop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse RAG outputs into a single embedding vector E_input.

        Args:
            s_out: (batch, seq_len_s, input_dim) — search output embeddings.
            kb_out: (batch, seq_len_kb, input_dim) — knowledge-base output embeddings.
            f_loop: (batch, seq_len_f, input_dim) or None — optional feedback.

        Returns:
            E_input: (batch, d_model) — fused embedding.
        """
        to_cat = [s_out, kb_out]
        if f_loop is not None:
            to_cat.append(f_loop)

        combined = torch.cat(to_cat, dim=1)
        x = self.input_proj(combined)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.layer_norm(x)
        e_input = x.mean(dim=1)
        return e_input


class RAGToEmbedding:
    """Converts RAG pipeline output dict into tensors suitable for TransformerEncoder."""

    def prepare_inputs(
        self,
        rag_result: dict,
        embedder,
        feedback_text: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Convert RAG result dict into (S_out, KB_out, F_loop) tensors.

        Args:
            rag_result: dict from RAGPipeline.process_query()
            embedder: callable with encode(texts) → np.ndarray
            feedback_text: optional user feedback string.

        Returns:
            (s_out, kb_out, f_loop) — each of shape (1, seq_len, embed_dim).
            f_loop is None if feedback_text is None.
        """
        search_texts = []
        for item in rag_result.get("search_output", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            search_texts.append(f"{title}. {snippet}")

        kb_out_dict = rag_result.get("knowledge_base_output", {})
        kb_summary = kb_out_dict.get("summary", "")
        kb_facts = " ".join(kb_out_dict.get("key_facts", []))
        kb_drugs = ", ".join(kb_out_dict.get("drug_mentions", []))
        kb_cohorts = ", ".join(kb_out_dict.get("relevant_cohorts", []))
        kb_text = f"Summary: {kb_summary} | Facts: {kb_facts} | Drugs: {kb_drugs} | Cohorts: {kb_cohorts}"

        s_emb = self._embed_texts(search_texts, embedder)
        kb_emb = self._embed_texts([kb_text], embedder)

        f_emb = None
        if feedback_text is not None:
            f_emb = self._embed_texts([feedback_text], embedder)

        return s_emb, kb_emb, f_emb

    @staticmethod
    def _embed_texts(texts: list[str], embedder) -> torch.Tensor:
        if not texts:
            dim = getattr(embedder, "dim", 384)
            return torch.empty(1, 0, dim)
        if hasattr(embedder, "embed_documents"):
            embeddings = embedder.embed_documents(texts)
        elif hasattr(embedder, "encode"):
            embeddings = embedder.encode(texts, convert_to_numpy=True)
        else:
            raise AttributeError("embedder must have embed_documents() or encode() method")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        tensor = torch.from_numpy(embeddings.astype(np.float32)).unsqueeze(0)
        return tensor
