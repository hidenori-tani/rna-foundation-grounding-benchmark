#!/usr/bin/env python3
"""
classifiers.py — 軽量分類器・回帰器の実装（Task 2.6）

提供モデル:
    - LogisticRegressionClassifier (scikit-learn)
    - MLPClassifier (PyTorch, 3-layer, hidden=128, dropout=0.3)
    - RidgeRegressor (scikit-learn) for continuous half-life
    - MLPRegressor (PyTorch, 同アーキテクチャで出力=1)

入力: embedding [N, D]（各モデル固有の次元数）
出力: 二値ラベル予測確率 [N, 2] または 連続値予測 [N]

API:
    clf = LogisticRegressionClassifier()
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)
    preds = clf.predict(X_test)

テスト: `python classifiers.py --self-test` で dummy input 整合性を確認。
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


class LogisticRegressionClassifier:
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str | None = "balanced"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict_proba(Xs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


class RidgeRegressor:
    def __init__(self, alpha: float = 1.0):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


class _MLPBase:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ):
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        ).to(self.device)

        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

    def _to_tensor(self, X: np.ndarray, dtype):
        import torch

        return torch.tensor(X, dtype=dtype, device=self.device)


class MLPClassifier(_MLPBase):
    def __init__(self, in_dim: int, n_classes: int = 2, **kwargs):
        super().__init__(in_dim=in_dim, out_dim=n_classes, **kwargs)
        import torch.nn as nn

        self.loss = nn.CrossEntropyLoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        import torch

        Xt = self._to_tensor(X, torch.float32)
        yt = self._to_tensor(y, torch.long)
        n = len(Xt)
        self.net.train()
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                b = idx[i : i + self.batch_size]
                self.optim.zero_grad()
                logits = self.net(Xt[b])
                l = self.loss(logits, yt[b])
                l.backward()
                self.optim.step()
                tot += float(l) * len(b)
            if (ep + 1) % 25 == 0:
                log.debug(f"epoch {ep+1}: loss={tot/n:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        Xt = self._to_tensor(X, torch.float32)
        with torch.no_grad():
            logits = self.net(Xt)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=-1)


class MLPRegressor(_MLPBase):
    def __init__(self, in_dim: int, **kwargs):
        super().__init__(in_dim=in_dim, out_dim=1, **kwargs)
        import torch.nn as nn

        self.loss = nn.MSELoss()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPRegressor":
        import torch

        Xt = self._to_tensor(X, torch.float32)
        yt = self._to_tensor(y.reshape(-1, 1), torch.float32)
        n = len(Xt)
        self.net.train()
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            tot = 0.0
            for i in range(0, n, self.batch_size):
                b = idx[i : i + self.batch_size]
                self.optim.zero_grad()
                pred = self.net(Xt[b])
                l = self.loss(pred, yt[b])
                l.backward()
                self.optim.step()
                tot += float(l) * len(b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch

        self.net.eval()
        Xt = self._to_tensor(X, torch.float32)
        with torch.no_grad():
            pred = self.net(Xt).cpu().numpy().flatten()
        return pred


def self_test():
    """Dummy input でshape整合性を確認（Step 2.6.4）"""
    np.random.seed(0)
    n, d = 40, 32
    X = np.random.randn(n, d).astype(np.float32)
    y_cls = np.random.randint(0, 2, size=n)
    y_reg = np.random.randn(n).astype(np.float32)

    log.info("Testing LogisticRegressionClassifier...")
    clf = LogisticRegressionClassifier().fit(X, y_cls)
    probs = clf.predict_proba(X)
    assert probs.shape == (n, 2), probs.shape
    log.info(f"  probs shape OK: {probs.shape}")

    log.info("Testing RidgeRegressor...")
    rg = RidgeRegressor().fit(X, y_reg)
    pred = rg.predict(X)
    assert pred.shape == (n,), pred.shape
    log.info(f"  pred shape OK: {pred.shape}")

    log.info("Testing MLPClassifier (short run)...")
    mlp_c = MLPClassifier(in_dim=d, n_classes=2, epochs=5).fit(X, y_cls)
    probs = mlp_c.predict_proba(X)
    assert probs.shape == (n, 2), probs.shape
    log.info(f"  probs shape OK: {probs.shape}")

    log.info("Testing MLPRegressor (short run)...")
    mlp_r = MLPRegressor(in_dim=d, epochs=5).fit(X, y_reg)
    pred = mlp_r.predict(X)
    assert pred.shape == (n,), pred.shape
    log.info(f"  pred shape OK: {pred.shape}")

    log.info("All self-tests passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        log.info("Use --self-test to run integrity checks.")
