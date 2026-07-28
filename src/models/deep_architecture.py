"""Shared trainer for the benchmark's four deliberately distinct neural models.

Three architectures are faithful ports of the production pipeline in
``neuro-marketing-research``: its flat MLP, electrode-axis CNN, and published
reference FFTLSTM. The fourth is an FT-Transformer representing modern tabular
attention. All use the same scaling, validation, optimization, probability, and
checkpoint code so the comparison changes architecture rather than procedure.

Input is always the existing channel-major ``(batch, 160)`` feature matrix.
Only the network reshapes it: the CNN uses ``(5 bands, 32 electrodes)`` while
the reference LSTM deliberately reproduces the published, artificial
``(160 steps, 1 feature)`` interpretation.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import (
    NEURAL_BATCH_SIZE,
    NEURAL_EPOCHS,
    NEURAL_LEARNING_RATE,
    NEURAL_PATIENCE,
    NEURAL_VALIDATION_FRACTION,
    NEURAL_WEIGHT_DECAY,
)


class TorchTabularClassifier(BaseEstimator, ClassifierMixin):
    """Expose bundled PyTorch networks through a probabilistic sklearn API.

    Parameters configure the common trainer, not individual architecture
    topology. ``device='auto'`` selects CUDA when available. Fitted attributes
    use sklearn's trailing underscore convention.
    """

    def __init__(
        self,
        architecture: str = "feature_mlp",
        epochs: int = NEURAL_EPOCHS,
        batch_size: int = NEURAL_BATCH_SIZE,
        learning_rate: float = NEURAL_LEARNING_RATE,
        weight_decay: float = NEURAL_WEIGHT_DECAY,
        patience: int = NEURAL_PATIENCE,
        validation_fraction: float = NEURAL_VALIDATION_FRACTION,
        random_state: int = 42,
        device: str = "auto",
    ) -> None:
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchTabularClassifier":
        """Fit train-only scaling and restore the lowest-validation-loss epoch.

        A fixed stratified 15% validation subset is created only from the
        benchmark training rows. Test data never affects scaling or stopping.
        """
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        if not np.array_equal(self.classes_, np.array([0, 1])):
            raise ValueError(
                "DEAP neural models require labels containing both 0 and 1"
            )
        self.scaler_ = StandardScaler().fit(X)
        transformed = self.scaler_.transform(X).astype(np.float32)
        x_tr, x_va, y_tr, y_va = train_test_split(
            transformed,
            np.asarray(y, dtype=np.int64),
            test_size=self.validation_fraction,
            random_state=self.random_state,
            stratify=y,
        )
        self.device_ = torch.device(
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )
        self.model_ = _make_network(self.architecture, transformed.shape[1]).to(
            self.device_
        )
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        loss_fn = nn.CrossEntropyLoss()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            generator=torch.Generator().manual_seed(self.random_state),
        )
        x_val = torch.from_numpy(x_va).to(self.device_)
        y_val = torch.from_numpy(y_va).to(self.device_)
        best_loss, best_state, stale = float("inf"), None, 0
        self.history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(self.model_(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += float(loss) * len(xb)
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.model_(x_val), y_val))
            scheduler.step(val_loss)
            self.history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss / len(x_tr),
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.model_.load_state_dict(best_state)
        self.n_parameters_ = sum(
            p.numel() for p in self.model_.parameters() if p.requires_grad
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``[P(Low), P(High)]`` from batched softmax logits."""
        import torch

        if not hasattr(self, "model_"):
            raise RuntimeError("fit must be called before predict_proba")
        values = self.scaler_.transform(X).astype(np.float32)
        outputs = []
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(values[start : start + self.batch_size]).to(
                    self.device_
                )
                outputs.append(torch.softmax(self.model_(batch), dim=1).cpu().numpy())
        return np.concatenate(outputs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the class with the largest predicted probability."""
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, path) -> None:
        """Save CPU weights, scaler, classes, trainer settings, and history."""
        import torch

        state = {
            key: value.detach().cpu() for key, value in self.model_.state_dict().items()
        }
        torch.save(
            {
                "architecture": self.architecture,
                "n_features": self.scaler_.n_features_in_,
                "state_dict": state,
                "scaler_mean": self.scaler_.mean_,
                "scaler_scale": self.scaler_.scale_,
                "classes": self.classes_,
                "parameters": self.get_params(deep=False),
                "history": self.history_,
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str = "auto") -> "TorchTabularClassifier":
        """Restore a checkpoint produced by :meth:`save`."""
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        params = checkpoint["parameters"]
        params["device"] = device
        instance = cls(**params)
        instance.classes_ = checkpoint["classes"]
        instance.scaler_ = StandardScaler()
        instance.scaler_.mean_ = checkpoint["scaler_mean"]
        instance.scaler_.scale_ = checkpoint["scaler_scale"]
        instance.scaler_.var_ = instance.scaler_.scale_**2
        instance.scaler_.n_features_in_ = checkpoint["n_features"]
        instance.device_ = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu" if device == "auto" else device
        )
        instance.model_ = _make_network(
            instance.architecture, checkpoint["n_features"]
        ).to(instance.device_)
        instance.model_.load_state_dict(checkpoint["state_dict"])
        instance.history_ = checkpoint.get("history", [])
        instance.n_parameters_ = sum(
            p.numel() for p in instance.model_.parameters() if p.requires_grad
        )
        return instance


def _make_network(name: str, n_features: int):
    """Build one untrained two-logit network from its registry key."""
    import torch
    from torch import nn

    if n_features != 160:
        raise ValueError(
            f"Neural benchmark expects 160 DEAP features, got {n_features}"
        )

    class FeatureMLP(nn.Module):
        """Flat feature MLP: 160-256-128-64-2 with BN/ReLU/dropout."""

        def __init__(self) -> None:
            super().__init__()
            layers = []
            previous = 160
            for width in (256, 128, 64):
                layers.extend(
                    (
                        nn.Linear(previous, width),
                        nn.BatchNorm1d(width),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                    )
                )
                previous = width
            layers.append(nn.Linear(previous, 2))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x.reshape(x.size(0), -1))

    class BandElectrodeCNN(nn.Module):
        """Convolve five band channels over the 32-position electrode axis."""

        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(5, 32, 3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(32, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(64, 2)

        def forward(self, x):
            grid = x.reshape(x.size(0), 32, 5).transpose(1, 2)
            return self.head(self.features(grid).squeeze(-1))

    class FFTLSTM(nn.Module):
        """Published five-layer LSTM over 160 artificial feature steps."""

        def __init__(self) -> None:
            super().__init__()
            self.lstm1 = nn.LSTM(1, 128, batch_first=True, bidirectional=True)
            self.lstm2 = nn.LSTM(256, 256, batch_first=True)
            self.lstm3 = nn.LSTM(256, 64, batch_first=True)
            self.lstm4 = nn.LSTM(64, 64, batch_first=True)
            self.lstm5 = nn.LSTM(64, 32, batch_first=True)
            self.dropouts = nn.ModuleList(
                [nn.Dropout(rate) for rate in (0.6, 0.6, 0.6, 0.4, 0.4)]
            )
            self.head = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 2))

        def forward(self, x):
            x = x.reshape(x.size(0), 160, 1)
            x, _ = self.lstm1(x)
            x = self.dropouts[0](x)
            x, _ = self.lstm2(x)
            x = self.dropouts[1](x)
            x, _ = self.lstm3(x)
            x = self.dropouts[2](x)
            x, _ = self.lstm4(x)
            x = self.dropouts[3](x)
            x, _ = self.lstm5(x)
            x = self.dropouts[4](x[:, -1, :])
            return self.head(x)

    class FTTransformer(nn.Module):
        """Embed scalar features as tokens, self-attend, then mean-pool."""

        def __init__(self) -> None:
            super().__init__()
            width = 16
            self.weight = nn.Parameter(torch.empty(160, width))
            self.bias = nn.Parameter(torch.zeros(160, width))
            nn.init.xavier_uniform_(self.weight)
            layer = nn.TransformerEncoderLayer(
                width,
                nhead=4,
                dim_feedforward=64,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2))

        def forward(self, x):
            tokens = x.unsqueeze(-1) * self.weight + self.bias
            return self.head(self.encoder(tokens).mean(dim=1))

    factories = {
        "feature_mlp": FeatureMLP,
        "band_electrode_cnn": BandElectrodeCNN,
        "fft_lstm": FFTLSTM,
        "ft_transformer": FTTransformer,
    }
    if name not in factories:
        raise ValueError(
            f"Unknown neural architecture {name!r}; expected {tuple(factories)}"
        )
    return factories[name]()
