"""Utility functions for world model training."""

from typing import Any, Dict

import torch
import torch.nn as nn


def configure_opt(model: nn.Module, **kwargs) -> torch.optim.Optimizer:
    """Configure optimizer for world model."""
    lr = kwargs.get("lr", 1e-4)
    eps = kwargs.get("eps", 1e-5)
    weight_decay = kwargs.get("weight_decay", 0.0)
    clip_grad = kwargs.get("clip_grad", 100.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay)
    return optimizer


class DictConfig:
    """Simple dict-based config for compatibility."""

    def __init__(self, data: Dict[str, Any] = None):
        if data is None:
            data = {}
        self._data = data

    def __getattr__(self, name: str) -> Any:
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any):
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any):
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._data
