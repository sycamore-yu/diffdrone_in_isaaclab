"""Parameter randomization system for sim2real transfer.

Provides UniformRandomizer, NormalRandomizer, and RandomizerManager
for randomizing physical parameters (mass, inertia, drag) during training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, List, Optional
import torch


@dataclass
class RandomizerConfig:
    """Base configuration for randomizers."""
    enabled: bool = True
    default: float = 0.0


@dataclass
class UniformRandomizerConfig(RandomizerConfig):
    """Configuration for uniform randomization."""
    min: float = 0.0
    max: float = 1.0


@dataclass
class NormalRandomizerConfig(RandomizerConfig):
    """Configuration for normal (Gaussian) randomization."""
    mean: float = 0.0
    std: float = 1.0


class RandomizerBase:
    """Base class for all randomizers."""

    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: float,
        device: torch.device,
        enabled: bool = True,
        dtype: torch.dtype = torch.float,
    ):
        self.value = torch.zeros(shape, device=device, dtype=dtype)
        self.default_value = default_value
        self.enabled = enabled
        self.device = device
        self.dtype = dtype
        self._shape = shape
        self.default()

    def default(self) -> torch.Tensor:
        """Reset to default value."""
        self.value.fill_(self.default_value)
        return self.value

    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomize values. Override in subclasses."""
        raise NotImplementedError

    # Operator overloads for tensor-like behavior
    def __add__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.value + other

    def __radd__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return other + self.value

    def __sub__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.value - other

    def __rsub__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return other - self.value

    def __mul__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.value * other

    def __rmul__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return other * self.value

    def __truediv__(self, other: Union[float, torch.Tensor]) -> torch.Tensor:
        return self.value / other

    def __neg__(self) -> torch.Tensor:
        return -self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value={self.value}, enabled={self.enabled})"

    @property
    def shape(self) -> torch.Size:
        return self.value.shape


class UniformRandomizer(RandomizerBase):
    """Randomizer that samples uniformly within [low, high]."""

    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: float,
        device: torch.device,
        enabled: bool = True,
        low: float = 0.0,
        high: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__(shape, default_value, device, enabled, dtype)
        self.low = low
        self.high = high

    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomize using uniform distribution."""
        if not self.enabled:
            return self.default()

        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new_values = torch.empty_like(self.value).uniform_(self.low, self.high)
            self.value = torch.where(mask, new_values, self.value)
        else:
            self.value.uniform_(self.low, self.high)
        return self.value

    @classmethod
    def from_config(
        cls,
        config: UniformRandomizerConfig,
        shape: Union[int, List[int], torch.Size],
        device: torch.device,
        dtype: torch.dtype = torch.float,
    ) -> "UniformRandomizer":
        """Create from config dataclass."""
        return cls(
            shape=shape,
            default_value=config.default,
            device=device,
            enabled=config.enabled,
            low=config.min,
            high=config.max,
            dtype=dtype,
        )


class NormalRandomizer(RandomizerBase):
    """Randomizer that samples from a normal (Gaussian) distribution."""

    def __init__(
        self,
        shape: Union[int, List[int], torch.Size],
        default_value: float,
        device: torch.device,
        enabled: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__(shape, default_value, device, enabled, dtype)
        self.mean = mean
        self.std = std

    def randomize(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Randomize using normal distribution."""
        if not self.enabled:
            return self.default()

        if idx is not None:
            mask = torch.zeros_like(self.value, dtype=torch.bool)
            mask[idx] = True
            new_values = torch.empty_like(self.value).normal_(self.mean, self.std)
            self.value = torch.where(mask, new_values, self.value)
        else:
            self.value.normal_(self.mean, self.std)
        return self.value

    @classmethod
    def from_config(
        cls,
        config: NormalRandomizerConfig,
        shape: Union[int, List[int], torch.Size],
        device: torch.device,
        dtype: torch.dtype = torch.float,
    ) -> "NormalRandomizer":
        """Create from config dataclass."""
        return cls(
            shape=shape,
            default_value=config.default,
            device=device,
            enabled=config.enabled,
            mean=config.mean,
            std=config.std,
            dtype=dtype,
        )


class RandomizerManager:
    """Manages multiple randomizers and provides batch refresh."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._randomizers: List[RandomizerBase] = []

    def add(self, randomizer: RandomizerBase) -> None:
        """Add a randomizer to the manager."""
        self._randomizers.append(randomizer)

    def refresh(self, idx: Optional[torch.Tensor] = None) -> None:
        """Refresh all randomizers."""
        if not self.enabled:
            for randomizer in self._randomizers:
                randomizer.default()
            return

        for randomizer in self._randomizers:
            if randomizer.enabled:
                randomizer.randomize(idx)
            else:
                randomizer.default()

    def __len__(self) -> int:
        return len(self._randomizers)

    def __getitem__(self, idx: int) -> RandomizerBase:
        return self._randomizers[idx]

    def __repr__(self) -> str:
        return f"RandomizerManager(enabled={self.enabled}, randomizers={len(self._randomizers)})"


def create_randomizer(
    config: Union[UniformRandomizerConfig, NormalRandomizerConfig],
    shape: Union[int, List[int], torch.Size],
    device: torch.device,
    dtype: torch.dtype = torch.float,
) -> RandomizerBase:
    """Factory function to create randomizer from config."""
    if isinstance(config, UniformRandomizerConfig):
        return UniformRandomizer.from_config(config, shape, device, dtype)
    elif isinstance(config, NormalRandomizerConfig):
        return NormalRandomizer.from_config(config, shape, device, dtype)
    else:
        raise ValueError(f"Unknown randomizer config type: {type(config)}")
