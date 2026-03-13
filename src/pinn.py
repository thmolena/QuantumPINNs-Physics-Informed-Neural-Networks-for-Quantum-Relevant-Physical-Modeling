"""
pinn.py — Neural network architectures for Physics-Informed Neural Networks.

Provides:
- PINN: fully connected network for single-output real-valued problems.
- ComplexPINN: two-headed network for real/imaginary wavefunction components.
- SinePINN: network with sinusoidal activations (better for oscillatory PDEs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Sine activation ────────────────────────────────────────────────────────


class Sine(nn.Module):
    """Sinusoidal activation function f(x) = sin(x).

    Recommended for quantum systems with oscillatory solutions
    (Sitzmann et al., SIREN networks).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# ── Base PINN ──────────────────────────────────────────────────────────────


class PINN(nn.Module):
    """Fully connected physics-informed neural network.

    Maps (x, t) → scalar output (e.g., real part of wavefunction,
    probability density, or energy eigenstate).

    Args:
        in_dim:      Input dimension (default 2 for (x, t); 1 for t-independent).
        out_dim:     Output dimension.
        hidden_dim:  Width of each hidden layer.
        n_layers:    Number of hidden layers.
        activation:  'tanh' (default), 'sine', or 'gelu'.
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        act_map = {
            "tanh": nn.Tanh,
            "sine": Sine,
            "gelu": nn.GELU,
        }
        if activation not in act_map:
            raise ValueError(f"Unknown activation '{activation}'. Choose from {list(act_map)}")
        act_cls = act_map[activation]

        layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * n_layers + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())

        self.net = nn.Sequential(*layers)
        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        for m in self.net.modules():
            if not isinstance(m, nn.Linear):
                continue
            if activation == "sine":
                # SIREN initialization
                fan_in = m.weight.size(1)
                bound = math.sqrt(6 / fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Complex (two-headed) PINN ──────────────────────────────────────────────


class ComplexPINN(nn.Module):
    """Two-headed PINN for complex-valued wavefunctions.

    Outputs (ψ_real, ψ_imag) from a shared trunk network followed by two
    separate linear heads.

    Args:
        in_dim, hidden_dim, n_layers, activation: Same as PINN.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 4,
        activation: str = "tanh",
    ) -> None:
        super().__init__()

        act_map = {"tanh": nn.Tanh, "sine": Sine, "gelu": nn.GELU}
        act_cls = act_map[activation]

        trunk_layers: list[nn.Module] = []
        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(act_cls())
        self.trunk = nn.Sequential(*trunk_layers)

        self.head_real = nn.Linear(hidden_dim, 1)
        self.head_imag = nn.Linear(hidden_dim, 1)

        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        for m in self.modules():
            if not isinstance(m, nn.Linear):
                continue
            if activation == "sine":
                fan_in = m.weight.size(1)
                bound = math.sqrt(6 / fan_in)
                nn.init.uniform_(m.weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inp = torch.cat([x, t], dim=-1)
        h = self.trunk(inp)
        return self.head_real(h), self.head_imag(h)

    def probability_density(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return |ψ|² = ψ_real² + ψ_imag²."""
        psi_r, psi_i = self.forward(x, t)
        return psi_r**2 + psi_i**2

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
