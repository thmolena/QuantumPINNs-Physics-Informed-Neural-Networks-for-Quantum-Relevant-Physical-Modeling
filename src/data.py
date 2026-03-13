"""
data.py — Domain sampling and collocation point generation.

Provides utilities for:
- Sampling interior collocation points for PDE residual evaluation.
- Sampling boundary and initial condition points.
- Loading and preprocessing sparse observational data.
"""

import numpy as np
import torch


# ── Domain samplers ────────────────────────────────────────────────────────


def sample_interior(n_points: int, x_range: tuple, t_range: tuple,
                    device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniformly distributed interior collocation points.

    Args:
        n_points: Number of collocation points.
        x_range:  (x_min, x_max) spatial domain.
        t_range:  (t_min, t_max) temporal domain. Use (0, 0) for time-independent problems.
        device:   PyTorch device string.

    Returns:
        (x, t) tensors of shape (n_points, 1) with requires_grad=True.
    """
    x = torch.FloatTensor(n_points, 1).uniform_(*x_range).to(device)
    t = torch.FloatTensor(n_points, 1).uniform_(*t_range).to(device)
    x.requires_grad_(True)
    t.requires_grad_(True)
    return x, t


def sample_boundary(n_points: int, x_range: tuple, t_range: tuple,
                    device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample boundary points (x = x_min or x = x_max) at random times.

    Returns:
        (x_bc, t_bc) tensors of shape (n_points, 1).
    """
    x_min, x_max = x_range
    sides = np.random.choice([x_min, x_max], size=n_points)
    x_bc = torch.FloatTensor(sides).unsqueeze(1).to(device)
    t_bc = torch.FloatTensor(n_points, 1).uniform_(*t_range).to(device)
    x_bc.requires_grad_(True)
    t_bc.requires_grad_(True)
    return x_bc, t_bc


def sample_initial(n_points: int, x_range: tuple, t0: float = 0.0,
                   device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample initial condition points at t = t0.

    Returns:
        (x_ic, t_ic) tensors of shape (n_points, 1).
    """
    x_ic = torch.FloatTensor(n_points, 1).uniform_(*x_range).to(device)
    t_ic = torch.full((n_points, 1), t0, dtype=torch.float32).to(device)
    x_ic.requires_grad_(True)
    t_ic.requires_grad_(True)
    return x_ic, t_ic


# ── Analytic reference solutions ───────────────────────────────────────────


def harmonic_oscillator_ground_state(x: np.ndarray, omega: float = 1.0) -> np.ndarray:
    """Analytic ground-state wavefunction for the quantum harmonic oscillator.

    ψ_0(x) = (m*ω/π*ℏ)^(1/4) * exp(-m*ω*x² / 2ℏ)

    With ℏ = m = 1 and given ω, this simplifies to:
        ψ_0(x) = (ω/π)^(1/4) * exp(-ω*x² / 2)
    """
    return (omega / np.pi) ** 0.25 * np.exp(-omega * x**2 / 2)


def gaussian_wavepacket(x: np.ndarray, t: float, x0: float = 0.0,
                        k0: float = 5.0, sigma: float = 0.5) -> np.ndarray:
    """Free-particle Gaussian wavepacket at time t (analytic solution).

    ψ(x, t) = A(t) * exp(-(x - x0 - k0*t)² / (2*σ_t²)) * exp(i*(k0*x - k0²*t/2))

    Returns the real part for visualization purposes.
    """
    sigma_t = sigma * np.sqrt(1 + 1j * t / sigma**2)
    A = (sigma / (sigma_t * np.sqrt(2 * np.pi))) ** 0.5
    psi = A * np.exp(-(x - x0 - k0 * t)**2 / (2 * sigma_t**2)) \
            * np.exp(1j * (k0 * x - k0**2 * t / 2))
    return np.real(psi)


# ── Sparse data loader ─────────────────────────────────────────────────────


def load_sparse_observations(csv_path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load sparse measurement data from a CSV file.

    Expected columns: x, t, psi_real, psi_imag (psi_imag optional).

    Returns a dict with keys 'x', 't', 'psi_real', and optionally 'psi_imag'.
    """
    import csv

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        raise ValueError(f"No data found in {csv_path}")

    def col(name):
        return torch.FloatTensor(
            [r[name] for r in rows if name in r]
        ).unsqueeze(1).to(device)

    data = {"x": col("x"), "t": col("t"), "psi_real": col("psi_real")}
    if "psi_imag" in rows[0]:
        data["psi_imag"] = col("psi_imag")
    return data
