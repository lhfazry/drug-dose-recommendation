"""Preprocessing pipeline: normalization, encoding, and GAIN-based imputation."""

import pickle
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from tqdm import tqdm


class _Generator(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor, m: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, m, z], dim=1))


class _Discriminator(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

    def forward(self, x_hat: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_hat, h], dim=1))


class GAINImputer:
    """GAIN-based missing value imputation for numerical features."""

    def __init__(self, hint_rate: float = 0.9, alpha: float = 10.0):
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.generator: Optional[_Generator] = None
        self.discriminator: Optional[_Discriminator] = None
        self._dim: Optional[int] = None
        self._min_vals: Optional[np.ndarray] = None
        self._max_vals: Optional[np.ndarray] = None
        self._fitted = False

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.001,
    ) -> "GAINImputer":
        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        n, self._dim = X_np.shape

        self._min_vals = np.nanmin(X_np, axis=0)
        self._max_vals = np.nanmax(X_np, axis=0)
        rng = self._max_vals - self._min_vals
        rng[rng == 0] = 1.0

        X_norm = 2 * (X_np - self._min_vals) / rng - 1
        X_norm[np.isnan(X_np)] = 0.0
        X_t = torch.tensor(X_norm, dtype=torch.float32)

        self.generator = _Generator(self._dim)
        self.discriminator = _Discriminator(self._dim)
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        bce = nn.BCELoss()

        pbar = tqdm(range(epochs), desc="GAIN training")
        for epoch in pbar:
            perm = torch.randperm(n)
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                x = X_t[idx]
                b = x.shape[0]

                m = (torch.rand(b, self._dim) < 0.2).float()
                x_masked = x.clone()
                x_masked[m.bool()] = 0.0
                z = torch.randn(b, 1)

                b_hint = (torch.rand(b, self._dim) < self.hint_rate).float()
                h = b_hint * m + 0.5 * (1 - b_hint)

                g_hat = self.generator(x_masked, m, z)
                x_hat = m * g_hat + (1 - m) * x
                d_pred = self.discriminator(x_hat.detach(), h)
                d_loss = bce(d_pred, m)

                d_opt.zero_grad()
                d_loss.backward()
                d_opt.step()

                g_hat = self.generator(x_masked, m, z)
                x_hat = m * g_hat + (1 - m) * x
                d_pred = self.discriminator(x_hat, h)

                g_loss_recon = ((g_hat - x) ** 2 * m).sum() / (m.sum() + 1e-8)
                g_loss_adv = -(torch.log(1 - d_pred + 1e-8) * m).sum() / (m.sum() + 1e-8)
                g_loss = g_loss_recon + self.alpha * g_loss_adv

                g_opt.zero_grad()
                g_loss.backward()
                g_opt.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                batches += 1

            pbar.set_postfix(
                G=f"{epoch_g_loss / max(batches, 1):.4f}",
                D=f"{epoch_d_loss / max(batches, 1):.4f}",
            )

        self._fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("GAINImputer must be fitted before transform()")

        X_np = X.values if isinstance(X, pd.DataFrame) else np.array(X, dtype=np.float32)
        rng = self._max_vals - self._min_vals
        rng[rng == 0] = 1.0

        X_norm = 2 * (X_np - self._min_vals) / rng - 1
        actual_observed = ~np.isnan(X_np)
        X_norm[~actual_observed] = 0.0

        missing_mask = (~actual_observed).astype(np.float32)
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        m_t = torch.tensor(missing_mask, dtype=torch.float32)
        z = torch.randn(X_np.shape[0], 1)

        self.generator.eval()
        with torch.no_grad():
            g_hat = self.generator(X_t, m_t, z)
            imputed = m_t * g_hat + (1 - m_t) * X_t

        result = (imputed.numpy() + 1) / 2 * rng + self._min_vals
        result[actual_observed] = X_np[actual_observed]
        return result


class PreprocessingPipeline:
    """Normalization, categorical encoding, and GAIN-based missing value imputation."""

    def __init__(self, feature_config):
        self.feature_config = feature_config
        self._num_features: List[str] = feature_config.get_numerical_features()
        self._cat_features: List[str] = feature_config.get_categorical_features()
        self._ord_features: List[str] = (
            feature_config.get_ordinal_features()
            if hasattr(feature_config, "get_ordinal_features")
            else []
        )
        self._bin_features: List[str] = (
            feature_config.get_binary_features()
            if hasattr(feature_config, "get_binary_features")
            else []
        )

        non_num = set(self._ord_features) | set(self._bin_features)
        self._num_features = [f for f in self._num_features if f not in non_num]

        self._nom_features: List[str] = [
            f
            for f in self._cat_features
            if f not in self._ord_features and f not in self._bin_features
        ]

        self.scaler = StandardScaler()
        self.gain = GAINImputer()
        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")

        transformers = []
        if self._ord_features:
            transformers.append(
                (
                    "ord",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    self._ord_features,
                )
            )
        if self._nom_features:
            transformers.append(
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self._nom_features,
                )
            )
        if self._bin_features:
            transformers.append(("bin", "passthrough", self._bin_features))

        self._ct: Optional[ColumnTransformer] = (
            ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
            if transformers
            else None
        )

        self._feature_names_out: Optional[List[str]] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PreprocessingPipeline":
        X_num = df[self._num_features].copy()

        self.gain.fit(X_num.values)
        self.scaler.fit(X_num)

        if self._cat_features:
            X_cat = df[self._cat_features].copy()
            self.cat_imputer.fit(X_cat)
            if self._ct is not None:
                self._ct.fit(X_cat)

        names: List[str] = list(self._num_features)
        names += self._ord_features
        if self._nom_features and self._ct is not None:
            ohe = self._ct.named_transformers_.get("ohe")
            if ohe is not None:
                names += list(ohe.get_feature_names_out(self._nom_features))
        names += self._bin_features
        self._feature_names_out = names

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Pipeline must be fitted before transform()")

        X_num_imp = self.gain.transform(df[self._num_features])
        X_num = pd.DataFrame(X_num_imp, columns=self._num_features, index=df.index)

        X_num_scaled = pd.DataFrame(
            self.scaler.transform(X_num),
            columns=self._num_features,
            index=df.index,
        )

        if self._cat_features and self._ct is not None:
            X_cat_imp = self.cat_imputer.transform(df[self._cat_features])
            X_cat = pd.DataFrame(X_cat_imp, columns=self._cat_features, index=df.index)
            X_cat_enc = self._ct.transform(X_cat)
            cat_cols = [n for n in self._feature_names_out if n not in self._num_features]
            X_cat_out = pd.DataFrame(X_cat_enc, columns=cat_cols, index=df.index)
        else:
            X_cat_out = pd.DataFrame(index=df.index)

        result = pd.concat([X_num_scaled, X_cat_out], axis=1)
        return result[self._feature_names_out]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        with open(path, "rb") as f:
            return pickle.load(f)
