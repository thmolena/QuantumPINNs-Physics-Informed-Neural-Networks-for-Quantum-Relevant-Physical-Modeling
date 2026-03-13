"""
physics.py — PDE residuals and physical constraint functions.

Implements:
- Schrödinger equation residual (time-dependent, 1D).
- Quantum harmonic oscillator (time-independent Schrödinger equation) residual.
- Anharmonic / double-well potential residual.
- Hamiltonian energy conservation residual.
- Composite PINN loss assembly.

All residuals use PyTorch autograd for exact derivative computation.
"""

import torch
import torch.nn as nn
from typing import Callable


# ── Autograd helpers ───────────────────────────────────────────────────────


def grad(output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """First-order derivative d(output)/d(inp) via autograd."""
    return torch.autograd.grad(
        output, inp,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


# ── Potential energy functions ─────────────────────────────────────────────


def harmonic_potential(x: torch.Tensor, omega: float = 1.0,
                       mass: float = 1.0) -> torch.Tensor:
    """V(x) = ½ m ω² x²."""
    return 0.5 * mass * omega**2 * x**2


def anharmonic_quartic_potential(x: torch.Tensor, lam: float = 0.1,
                                  omega: float = 1.0, mass: float = 1.0) -> torch.Tensor:
    """V(x) = ½ m ω² x² + λ x⁴  (quartic anharmonic oscillator)."""
    return harmonic_potential(x, omega, mass) + lam * x**4


def double_well_potential(x: torch.Tensor, a: float = 1.0,
                          b: float = 0.0) -> torch.Tensor:
    """V(x) = -a x² + b x⁴  (double-well potential)."""
    return -a * x**2 + b * x**4


# ── PDE residuals ──────────────────────────────────────────────────────────


def schrodinger_residual_1d(
    psi_r: torch.Tensor,
    psi_i: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    hbar: float = 1.0,
    mass: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PDE residual of the time-dependent 1D Schrödinger equation.

    iℏ ∂ψ/∂t = −(ℏ²/2m) ∂²ψ/∂x² + V(x)ψ

    Splitting ψ = ψ_r + i ψ_i, this becomes two coupled real equations:
        ℏ ∂ψ_r/∂t = −(ℏ²/2m) ∂²ψ_i/∂x² + V ψ_i
       −ℏ ∂ψ_i/∂t = −(ℏ²/2m) ∂²ψ_r/∂x² + V ψ_r

    Args:
        psi_r: Real part ψ_r(x, t), shape (N, 1).
        psi_i: Imaginary part ψ_i(x, t), shape (N, 1).
        x, t:  Input tensors (requires_grad=True).
        potential_fn: V(x) callable.

    Returns:
        (res_r, res_i): residual tensors for the real and imaginary equations.
    """
    V = potential_fn(x)

    # First-order time derivatives
    dpsi_r_dt = grad(psi_r, t)
    dpsi_i_dt = grad(psi_i, t)

    # Second-order spatial derivatives
    dpsi_r_dx  = grad(psi_r, x)
    dpsi_r_dxx = grad(dpsi_r_dx, x)
    dpsi_i_dx  = grad(psi_i, x)
    dpsi_i_dxx = grad(dpsi_i_dx, x)

    coeff = hbar**2 / (2 * mass)

    res_r = hbar * dpsi_r_dt - (-coeff * dpsi_i_dxx + V * psi_i)
    res_i = hbar * dpsi_i_dt - ( coeff * dpsi_r_dxx - V * psi_r)

    return res_r, res_i


def tise_residual_1d(
    psi: torch.Tensor,
    x: torch.Tensor,
    energy: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    hbar: float = 1.0,
    mass: float = 1.0,
) -> torch.Tensor:
    """PDE residual of the time-independent Schrödinger equation (TISE).

    −(ℏ²/2m) d²ψ/dx² + V(x)ψ = E ψ
    Residual: H ψ − E ψ

    Args:
        psi:         Network prediction ψ(x), shape (N, 1).
        x:           Spatial input with requires_grad=True.
        energy:      Scalar or (1,) tensor representing the predicted eigenvalue E.
        potential_fn: V(x) callable.

    Returns:
        Residual tensor of shape (N, 1).
    """
    V = potential_fn(x)
    dpsi_dx  = grad(psi, x)
    dpsi_dxx = grad(dpsi_dx, x)
    return -(hbar**2 / (2 * mass)) * dpsi_dxx + V * psi - energy * psi


def hamiltonian_energy_residual(
    q: torch.Tensor,
    p: torch.Tensor,
    t: torch.Tensor,
    potential_fn: Callable[[torch.Tensor], torch.Tensor],
    mass: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residuals of Hamilton's equations of motion.

    dq/dt = p/m
    dp/dt = −dV/dq

    Args:
        q, p: Generalized coordinate and momentum, shape (N, 1).
        t:    Time tensor with requires_grad=True.
        potential_fn: V(q).

    Returns:
        (res_q, res_p) residual tensors.
    """
    V  = potential_fn(q)
    dV = grad(V, q)

    dq_dt = grad(q, t)
    dp_dt = grad(p, t)

    res_q = dq_dt - p / mass
    res_p = dp_dt + dV
    return res_q, res_p


# ── Normalization constraint ───────────────────────────────────────────────


def normalization_residual(
    psi: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Approximate ∫ |ψ|² dx = 1 via trapezoidal summation over x.

    Returns a scalar tensor (||∫|ψ|²dx| − 1)².
    """
    density = psi**2
    # Simple rectangular quadrature over the sampled x points
    dx = (x.max() - x.min()) / (x.shape[0] - 1)
    norm = (density * dx).sum()
    return (norm - 1.0)**2


# ── Composite PINN loss ────────────────────────────────────────────────────


class PINNLoss(nn.Module):
    """Weighted composite loss for PINN training.

    Total loss:
        L = λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic + λ_data * L_data + λ_norm * L_norm

    Args:
        lambda_pde:  Weight for the PDE residual term.
        lambda_bc:   Weight for boundary condition violation.
        lambda_ic:   Weight for initial condition violation.
        lambda_data: Weight for data fidelity term.
        lambda_norm: Weight for normalization constraint (optional; 0 = disabled).
    """

    def __init__(
        self,
        lambda_pde: float = 1.0,
        lambda_bc: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_data: float = 1.0,
        lambda_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.lw = {
            "pde":  lambda_pde,
            "bc":   lambda_bc,
            "ic":   lambda_ic,
            "data": lambda_data,
            "norm": lambda_norm,
        }

    def mse(self, r: torch.Tensor) -> torch.Tensor:
        return (r**2).mean()

    def forward(
        self,
        pde_residuals: list[torch.Tensor],
        bc_residuals: list[torch.Tensor],
        ic_residuals: list[torch.Tensor],
        data_residuals: list[torch.Tensor],
        norm_residual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute and return individual and total losses.

        Each argument is a list of residual tensors (may be empty for unused terms).

        Returns:
            Dict with keys corresponding to each term plus 'total'.
        """
        losses = {}

        losses["pde"]  = sum(self.mse(r) for r in pde_residuals)  if pde_residuals  else torch.tensor(0.0)
        losses["bc"]   = sum(self.mse(r) for r in bc_residuals)   if bc_residuals   else torch.tensor(0.0)
        losses["ic"]   = sum(self.mse(r) for r in ic_residuals)   if ic_residuals   else torch.tensor(0.0)
        losses["data"] = sum(self.mse(r) for r in data_residuals) if data_residuals else torch.tensor(0.0)
        losses["norm"] = norm_residual if norm_residual is not None else torch.tensor(0.0)

        losses["total"] = sum(self.lw[k] * v for k, v in losses.items() if k != "total")
        return losses
