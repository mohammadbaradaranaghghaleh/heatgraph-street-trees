"""Deep ensemble of HeatGraphGNN surrogates (different seeds / data order).

The ensemble mean is the point prediction; the across-member standard deviation is a
per-segment difficulty estimate used by NormalizedConformal (Block G) and by the
active-learning acquisition (Block I). Lakshminarayanan et al. (2017) style, without
the variance head, since conformal calibration supplies the coverage guarantee.
"""
from __future__ import annotations

import torch

from .model import HeatGraphGNN
from .train import fit


class Ensemble:
    def __init__(self, members: list):
        self.members = members

    @classmethod
    def train(cls, train_data, val_data=None, n_members: int = 3, epochs: int = 60, seed: int = 0,
              model_kwargs: dict | None = None, **fit_kwargs) -> "Ensemble":
        members = []
        for i in range(n_members):
            torch.manual_seed(seed + i)
            m = HeatGraphGNN(**(model_kwargs or {}))
            fit(m, train_data, val_data, epochs=epochs, seed=seed + i, **fit_kwargs)
            members.append(m)
        return cls(members)

    def refit(self, train_data, val_data=None, epochs: int = 30, seed: int = 0, **fit_kwargs):
        """Warm-start every member on an enlarged training set (active learning)."""
        for i, m in enumerate(self.members):
            fit(m, train_data, val_data, epochs=epochs, seed=seed + 100 + i, **fit_kwargs)
        return self

    def eval(self):
        for m in self.members:
            m.eval()
        return self

    def parameters(self):
        for m in self.members:
            yield from m.parameters()

    @torch.no_grad()
    def _stack(self, data):
        return torch.stack([m.eval().predict(data) for m in self.members])

    def predict(self, data):
        return self._stack(data).mean(0)

    def predict_std(self, data):
        return self._stack(data).std(0, unbiased=False)
