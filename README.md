# Physics-Informed Neural Networks (PINNs) for Quantum-Relevant Physical Modeling

[![Project Website](https://img.shields.io/badge/Project%20Website-GitHub%20Pages-0969da?style=for-the-badge)](https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/)

> Physics-Informed Neural Networks incorporate governing equations and physical constraints into machine-learning models, enabling accurate predictions of quantum-relevant systems even in settings with limited or noisy data. This repository develops PINN architectures for Hamiltonian-governed systems and quantum simulation problems, constructs loss functions that enforce physical constraints, implements GPU-accelerated solvers, and evaluates models on differential-equation problems relevant to quantum systems — comparing PINN performance against traditional numerical methods and unconstrained machine-learning approaches.

## Abstract

This research develops Physics-Informed Neural Networks (PINNs) for modeling quantum-relevant physical systems. PINNs incorporate governing equations — such as the Schrödinger equation and Hamiltonian operators — directly into machine-learning loss functions, enabling accurate predictions even when labeled data are scarce or corrupted by noise.

The repository is organized around three complementary investigations. First, a harmonic-oscillator study demonstrates that PINN architectures designed for Hamiltonian-governed systems achieve near-analytic stationary-state recovery, establishing accuracy, stability, and computational viability as a modeling paradigm. Second, a time-dependent Schrödinger study shows that the same physics-informed framework extends to complex-valued wavepacket propagation with quantitative diagnostics for density, norm, phase, and transport — capabilities relevant to quantum simulation problems. Third, a shared comparative benchmark evaluates which PINN design choices transfer across problem classes and which are artifacts of single-benchmark tuning, directly comparing PINN performance against unconstrained machine-learning baselines under controlled architectural and robustness studies.

This work addresses a fundamental challenge in quantum computing: accurate modeling of systems governed by physical laws under computational constraints. Federal agencies emphasize that advancing quantum technologies requires simulation capability, modeling accuracy, and computational infrastructure. By improving physically constrained modeling, this research supports U.S. capabilities in quantum simulation and scientific computing. It also contributes to artificial intelligence and biomedical research, where models must incorporate scientific constraints and operate under limited data conditions.

## Research Objectives

The central research plan drives four linked objectives:

1. **PINN architectures for Hamiltonian-governed systems.** Design neural network architectures that encode Hamiltonian structure, boundary conditions, and symmetry constraints as hard inductive biases rather than soft penalties.
2. **Loss functions enforcing physical constraints.** Construct composite loss functions that jointly enforce the governing differential equation, eigenvalue consistency, normalization, and domain-specific physical invariants.
3. **GPU-accelerated solvers for quantum-relevant problems.** Implement and evaluate GPU-accelerated PINN solvers on differential-equation problems relevant to quantum systems, measuring accuracy, stability, and computational efficiency.
4. **Comparison against baselines.** Compare PINN performance against traditional numerical methods (analytic solutions, finite-difference solvers) and unconstrained machine-learning approaches, isolating the contribution of physics-informed inductive bias.

## Headline Results

| Evidence layer | Headline result | Research objective addressed |
|---|---|---|
| Harmonic oscillator | Relative L2 error $1.56927 \times 10^{-3}$, learned energy $0.50001526$, absolute energy error $1.52588 \times 10^{-5}$ | PINN architectures for Hamiltonian-governed systems: demonstrates near-analytic eigenstate recovery under hard physical constraints |
| Time-dependent Schrödinger | Initial density relative L2 error $7.92 \times 10^{-8}$, final density relative L2 error $5.66 \times 10^{-2}$, predicted norm close to unity | GPU-accelerated solvers for quantum simulation: complex-valued propagation with physically meaningful diagnostics |
| Combined benchmark | Best shared architecture is 5 layers x 64 units with relative L2 $0.26585$ and modest degradation under added noise | Comparison against baselines: separates transferable design choices from single-benchmark tuning across problem families |

## Best Visual Evidence

These figures are generated from committed CSV artifacts in `outputs/`, so the README and landing page remain synchronized and stable.

### Figure 1. Harmonic-oscillator benchmark

![Harmonic oscillator benchmark](outputs/qho_benchmark.svg)

This figure demonstrates the core capability of PINN architectures for Hamiltonian-governed systems. A physics-constrained network recovers the quantum harmonic oscillator ground state to relative L2 error $1.56927 \times 10^{-3}$ with very low energy error, validating that loss functions enforcing eigenvalue consistency, normalization, and boundary behavior produce near-analytic accuracy on differential-equation problems relevant to quantum systems.

### Figure 2. Time-dependent Schrödinger benchmark

![Time-dependent Schrodinger benchmark](outputs/schrodinger_benchmark.svg)

This figure validates the extension of PINNs to quantum simulation problems involving complex-valued dynamics. The PINN is essentially exact at the initial slice and remains controlled at later times while preserving norm behavior close to one — demonstrating that GPU-accelerated physics-informed solvers can model time-dependent quantum propagation with quantitative physical diagnostics.

### Figure 3. Shared architecture sweep

![Combined architecture sweep](outputs/combined_architecture.svg)

This figure addresses the comparison against baselines objective. A standardized architecture search across multiple quantum-relevant problem families identifies depth as the strongest architectural lever and isolates the 5-layer, 64-unit model as the best shared configuration — evidence that physics-informed design choices transfer across Hamiltonian-governed settings.

### Figure 4. Transferability and robustness summary

![Combined benchmark summary](outputs/combined_summary.svg)

This figure provides the strongest evidence for computational efficiency and robustness under real-world conditions. The shared PINN formulation remains accurate across multiple quantum problems and degrades only modestly under added input noise, supporting the claim that physics-informed inductive bias improves modeling even when data are limited or corrupted.

## How the Three Notebooks Support the Research Plan

| Notebook | Research objective | Key evidence | Significance for quantum technologies |
|---|---|---|---|
| `notebooks/pinn_harmonic_oscillator.ipynb` | PINN architectures for Hamiltonian-governed systems | Ground-state analysis, overlap matrix, ablation summary, analytic eigenstates | Validates that physics-constrained loss functions achieve near-analytic accuracy on a canonical quantum eigenproblem |
| `notebooks/pinn_schrodinger.ipynb` | GPU-accelerated solvers for quantum simulation | Density heatmap, phase plot, exact-vs-PINN snapshots, Ehrenfest diagnostics | Demonstrates complex-valued quantum dynamics modeling with physically meaningful diagnostics under limited data |
| `notebooks/quantum_pinn_combined.ipynb` | Comparison against baselines and transferability | Architecture sweep, scaling study, noise robustness, shared summary bars | Separates reusable design choices from single-benchmark tuning across quantum-relevant problem families |

Recommended reading order:

1. Start with the harmonic-oscillator notebook for the cleanest demonstration of PINN architectures designed for Hamiltonian-governed systems.
2. Continue to the time-dependent Schrödinger notebook for GPU-accelerated quantum simulation with physics-aware diagnostics.
3. Finish with the combined notebook, where PINN performance is compared against unconstrained baselines under a standardized multi-problem protocol.

## Method at a Glance

### Stationary-state branch — PINN architectures for Hamiltonian systems

The stationary branch solves

$$
-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x)
$$

with a neural network for $\psi(x)$ and a learned scalar parameter for $E$. The loss function enforces the governing differential equation, eigenvalue consistency via the Rayleigh quotient, boundary behavior through a hard Gaussian envelope, and parity symmetry — demonstrating how physical constraints can be embedded directly into the machine-learning optimization objective.

### Time-dependent branch — GPU-accelerated quantum simulation

The propagation branch solves

$$
i\frac{\partial \psi}{\partial t} = \left[-\frac{1}{2}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)
$$

using a dual-output network for the real and imaginary components. The implementation leverages GPU-accelerated automatic differentiation for efficient evaluation of the PDE residual. Evaluation focuses on density fidelity, norm preservation, phase structure, and transport behavior — diagnostics that validate the solver's physical credibility beyond pointwise agreement.

### Comparative branch — PINN vs. traditional and unconstrained methods

The combined notebook reuses the same PINN logic across multiple quantum-relevant settings, then varies depth, width, scaling, collocation behavior, and input noise to separate reusable modeling choices from benchmark-specific tuning. This directly compares physics-informed performance against unconstrained baselines, testing whether the additional computational cost of constraint enforcement is justified by improved accuracy, stability, and robustness.

## Reproducing the Paper Artifacts

```bash
conda activate qaoa
pip install -r requirements.txt

jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_harmonic_oscillator.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_schrodinger.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_pinn_combined.ipynb

jupyter nbconvert --to html notebooks/pinn_harmonic_oscillator.ipynb --output pinn_harmonic_oscillator.html
jupyter nbconvert --to html notebooks/pinn_schrodinger.ipynb --output pinn_schrodinger.html
jupyter nbconvert --to html notebooks/quantum_pinn_combined.ipynb --output quantum_pinn_combined.html

python -m src.train --problem harmonic_oscillator --epochs 5000 --collocation 2000 --model-path model.pt
python -m src.server --model-path model.pt --problem harmonic_oscillator
python -m http.server 8000
```

## Generated Artifacts

| File | Purpose |
|---|---|
| `outputs/qho_full_benchmark.csv` | Stationary-state benchmark across harmonic-oscillator eigenstates |
| `outputs/schrodinger_benchmark.csv` | Time-resolved density and norm diagnostics for the TDSE study |
| `outputs/combined_summary.csv` | Standardized summary across the shared benchmark problems |
| `outputs/combined_arch_grid.csv` | Depth-width sweep for the comparative benchmark |
| `outputs/combined_noise_robustness.csv` | Noise robustness under the shared benchmark protocol |
| `outputs/harmonic_oscillator_loss.csv` | Training loss trace for the direct training module |

## Repository Structure

```text
QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/
├── README.md
├── index.html
├── requirements.txt
├── data/
├── notebooks/
│   ├── pinn_harmonic_oscillator.ipynb
│   ├── pinn_harmonic_oscillator.html
│   ├── pinn_schrodinger.ipynb
│   ├── pinn_schrodinger.html
│   ├── quantum_pinn_combined.ipynb
│   └── quantum_pinn_combined.html
├── outputs/
├── src/
└── website/
```

## License

This project is released under the terms of the LICENSE file.