"""
train.py — Training loop and experiment management for QuantumPINNs.

Supports the following problems via --problem flag:
    harmonic_oscillator   Time-independent Schrödinger equation, harmonic potential.
    schrodinger           Time-dependent Schrödinger equation, 1D particle in box.
    anharmonic            Time-independent Schrödinger, quartic anharmonic potential.
    hamiltonian           Classical Hamiltonian dynamics under energy conservation.

Usage:
    python -m src.train --problem harmonic_oscillator --epochs 5000 --collocation 2000
"""

import argparse
import csv
import os
import time

import numpy as np
import torch

from src.data import sample_interior, sample_boundary, sample_initial
from src.pinn import PINN, ComplexPINN
from src.physics import (
    harmonic_potential,
    anharmonic_quartic_potential,
    schrodinger_residual_1d,
    tise_residual_1d,
    hamiltonian_energy_residual,
    PINNLoss,
)


# ── Problem registry ───────────────────────────────────────────────────────


PROBLEM_DEFAULTS = {
    "harmonic_oscillator": dict(
        x_range=(-5.0, 5.0), t_range=(0.0, 0.0),
        in_dim=1, n_layers=4, hidden_dim=64, activation="tanh",
    ),
    "schrodinger": dict(
        x_range=(-8.0, 8.0), t_range=(0.0, 2.0),
        in_dim=2, n_layers=5, hidden_dim=80, activation="tanh",
    ),
    "anharmonic": dict(
        x_range=(-4.0, 4.0), t_range=(0.0, 0.0),
        in_dim=1, n_layers=4, hidden_dim=64, activation="tanh",
    ),
    "hamiltonian": dict(
        x_range=(-3.0, 3.0), t_range=(0.0, 10.0),
        in_dim=1, n_layers=3, hidden_dim=48, activation="tanh",
    ),
}


# ── Training function ──────────────────────────────────────────────────────


def train(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[train] device={device}  problem={args.problem}  epochs={args.epochs}"
          f"  collocation={args.collocation}")

    cfg    = PROBLEM_DEFAULTS[args.problem]
    x_range, t_range = cfg["x_range"], cfg["t_range"]
    time_dep = t_range[1] > t_range[0]

    # ── Build model ────────────────────────────────────────────────────────
    if args.problem == "schrodinger":
        model = ComplexPINN(
            in_dim=cfg["in_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            activation=cfg["activation"],
        ).to(device)
    else:
        model = PINN(
            in_dim=cfg["in_dim"],
            out_dim=1,
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            activation=cfg["activation"],
        ).to(device)

    print(f"[train] parameters: {model.count_parameters():,}")

    energy = torch.tensor([1.5], requires_grad=True, device=device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + ([energy] if args.problem in ("harmonic_oscillator", "anharmonic") else []),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=500, min_lr=1e-6, verbose=False
    )

    loss_fn = PINNLoss(
        lambda_pde=args.lam_pde,
        lambda_bc=args.lam_bc,
        lambda_ic=args.lam_ic,
        lambda_data=args.lam_data,
        lambda_norm=args.lam_norm,
    )

    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()

        # ── Sample points ──────────────────────────────────────────────────
        x_int, t_int = sample_interior(
            args.collocation, x_range,
            t_range if time_dep else (0.0, 0.0), device
        )
        if time_dep:
            x_bc, t_bc = sample_boundary(args.collocation // 4, x_range, t_range, device)
            x_ic, t_ic = sample_initial(args.collocation // 4, x_range, 0.0, device)
        else:
            x_bc = torch.FloatTensor([[x_range[0]], [x_range[1]]]).to(device).requires_grad_(True)
            t_bc = torch.zeros_like(x_bc).requires_grad_(True)
            x_ic, t_ic = x_bc, t_bc

        # ── PDE residuals ──────────────────────────────────────────────────
        if args.problem == "harmonic_oscillator":
            psi = model(x_int, torch.zeros_like(x_int))
            pde_res = [tise_residual_1d(psi, x_int, energy,
                                        lambda x: harmonic_potential(x, omega=1.0))]
            bc_res  = [model(x_bc, torch.zeros_like(x_bc))]   # ψ → 0 at boundary
            ic_res  = []
            norm_res = None

        elif args.problem == "anharmonic":
            psi = model(x_int, torch.zeros_like(x_int))
            pde_res = [tise_residual_1d(psi, x_int, energy,
                                        lambda x: anharmonic_quartic_potential(x))]
            bc_res  = [model(x_bc, torch.zeros_like(x_bc))]
            ic_res  = []
            norm_res = None

        elif args.problem == "schrodinger":
            psi_r, psi_i = model(x_int, t_int)
            res_r, res_i = schrodinger_residual_1d(
                psi_r, psi_i, x_int, t_int,
                lambda x: harmonic_potential(x, omega=1.0)
            )
            pde_res = [res_r, res_i]
            psi_r_bc, psi_i_bc = model(x_bc, t_bc)
            bc_res  = [psi_r_bc, psi_i_bc]
            # Initial Gaussian wavepacket ψ(x, 0) = exp(-x²/2) (real, σ=1, k₀=0)
            psi_r_ic, psi_i_ic = model(x_ic, t_ic)
            ic_r_target = torch.exp(-x_ic**2 / 2)
            ic_i_target = torch.zeros_like(ic_r_target)
            ic_res  = [psi_r_ic - ic_r_target, psi_i_ic - ic_i_target]
            norm_res = None

        elif args.problem == "hamiltonian":
            # Network predicts (q, p) from t
            t_int_1d = t_int
            x_zero_int = torch.zeros_like(t_int_1d)
            qp = model(x_zero_int, t_int_1d)  # shape (N, 1) — using as q; secondary head not yet available
            # For simplicity, treat q = ψ, p = dψ/dt
            q = qp
            p = torch.autograd.grad(q, t_int_1d,
                                     grad_outputs=torch.ones_like(q),
                                     create_graph=True)[0]
            res_q, res_p = hamiltonian_energy_residual(
                q, p, t_int_1d, lambda q: harmonic_potential(q, omega=1.0)
            )
            pde_res = [res_q, res_p]
            bc_res  = []
            ic_res  = []
            norm_res = None
        else:
            raise ValueError(f"Unknown problem: {args.problem}")

        losses = loss_fn(pde_res, bc_res, ic_res, [], norm_res)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(losses["total"].item())

        if epoch % args.print_every == 0 or epoch == 1:
            elapsed = time.time() - t_start
            pde_val = losses["pde"].item()
            total_val = losses["total"].item()
            print(f"  epoch {epoch:6d}/{args.epochs}  "
                  f"total={total_val:.4e}  pde={pde_val:.4e}  "
                  f"elapsed={elapsed:.1f}s")
            history.append({"epoch": epoch, "total": total_val, "pde": pde_val})

    # ── Save model ─────────────────────────────────────────────────────────
    if args.model_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "args": vars(args)}, args.model_path)
        print(f"[train] model saved → {args.model_path}")

    # ── Save loss history ──────────────────────────────────────────────────
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    loss_path = os.path.join(out_dir, f"{args.problem}_loss.csv")
    with open(loss_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "total", "pde"])
        writer.writeheader()
        writer.writerows(history)
    print(f"[train] loss history saved → {loss_path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a QuantumPINN model.")
    p.add_argument("--problem", default="harmonic_oscillator",
                   choices=list(PROBLEM_DEFAULTS), help="Physical problem to solve.")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--collocation", type=int, default=2000,
                   help="Number of interior collocation points per batch.")
    p.add_argument("--hidden-dim", type=int, default=None,
                   help="Hidden layer width (overrides problem default).")
    p.add_argument("--n-layers", type=int, default=None,
                   help="Number of hidden layers (overrides problem default).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam-pde",  type=float, default=1.0)
    p.add_argument("--lam-bc",   type=float, default=1.0)
    p.add_argument("--lam-ic",   type=float, default=1.0)
    p.add_argument("--lam-data", type=float, default=1.0)
    p.add_argument("--lam-norm", type=float, default=0.0)
    p.add_argument("--model-path", default="model.pt")
    p.add_argument("--print-every", type=int, default=500)
    p.add_argument("--cpu", action="store_true", help="Force CPU training.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
