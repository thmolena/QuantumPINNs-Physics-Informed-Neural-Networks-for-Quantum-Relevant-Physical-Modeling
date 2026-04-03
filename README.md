# QuantumPINNs: Physics-Informed Neural Networks for Quantum-Relevant Physical Modeling

[![Project Website](https://img.shields.io/badge/Project%20Website-GitHub%20Pages-0969da?style=for-the-badge)](https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/)

> A unified PINN study for stationary and time-dependent quantum systems, organized around three complementary claims: high-accuracy eigenstate recovery, structured wavepacket propagation, and cross-problem transferability under a shared benchmark protocol.

## Overview

This repository studies whether physics-informed neural networks can recover useful quantum solutions when the governing equation is known and dense supervision is unnecessary or expensive. The project is built around three executed notebooks that serve different roles rather than repeating the same story.

1. The harmonic-oscillator notebook is the specialist stationary-state study. It jointly learns the wavefunction and eigenvalue and provides the strongest single-problem accuracy result in the repository.
2. The time-dependent Schrödinger notebook is the specialist propagation study. It learns the real and imaginary parts of the wavefunction and evaluates them with conservation-aware diagnostics.
3. The combined notebook is the comparative study. It asks which modeling choices remain credible when the problem class changes, using architecture and noise sweeps across multiple systems.

Taken together, the notebooks show that the same PINN framework can support near-analytic accuracy on a canonical eigenproblem, stable full-interval propagation on a complex-valued dynamics benchmark, and useful design guidance in a standardized cross-problem setting.

## Key Results

| Study | Main result | Why it matters |
|---|---|---|
| Harmonic oscillator | Relative L2 error $1.56927 \times 10^{-3}$ and learned energy $0.50001526$ for the ground state | Shows that the stationary-state PINN can recover both shape and eigenvalue to near-analytic precision |
| Time-dependent Schrödinger | Initial density relative L2 error $7.92 \times 10^{-8}$ and final density relative L2 error $5.66 \times 10^{-2}$ with predicted norm near unity | Shows that the propagation model remains accurate and physically structured across the time interval |
| Combined benchmark | Best shared architecture is 5 layers x 64 units with relative L2 $0.26585$ | Supplies the project-level transferability result rather than another specialist accuracy claim |

## Benchmark Figures

These figures are generated from the benchmark CSV files committed in the repository, so they render reliably in both the README and the landing page.

### Harmonic oscillator benchmark

![Harmonic oscillator benchmark](outputs/qho_benchmark.svg)

The stationary-state notebook is strongest on the ground state, where it reaches relative L2 error $1.56927 \times 10^{-3}$ and absolute energy error $1.52588 \times 10^{-5}$. Higher modes remain informative but become progressively harder.

### Time-dependent Schrödinger benchmark

![Time-dependent Schrodinger benchmark](outputs/schrodinger_benchmark.svg)

The propagation notebook starts essentially exact at $t = 0$ and retains controlled density error through the time window while keeping the predicted norm close to one.

### Shared architecture sweep

![Combined architecture sweep](outputs/combined_architecture.svg)

The combined benchmark identifies depth as the strongest lever in the tested grid, with the 5-layer, 64-unit model clearly outperforming the shallower alternatives.

### Combined benchmark robustness

![Combined benchmark summary](outputs/combined_summary.svg)

The cross-problem study should be read as a transferability benchmark. Its main value is systematic comparison across systems and robustness under noisy inputs, not replacing the specialist notebooks as the highest-accuracy results.

## Notebook Guide

| Notebook | Role | Best reported result |
|---|---|---|
| notebooks/pinn_harmonic_oscillator.ipynb | Specialist stationary-state study | Ground-state relative L2 $1.56927 \times 10^{-3}$, learned energy $0.50001526$, absolute energy error $1.52588 \times 10^{-5}$ |
| notebooks/pinn_schrodinger.ipynb | Specialist time-dependent study | Initial density relative L2 $7.92 \times 10^{-8}$, final density relative L2 $5.66 \times 10^{-2}$, norm range approximately $[0.993, 1.012]$ |
| notebooks/quantum_pinn_combined.ipynb | Comparative benchmark study | Best shared architecture 5 layers x 64 units with relative L2 $0.26585$ and mild sensitivity to added noise |

For the clearest reading order, start with the harmonic oscillator notebook, continue to the time-dependent notebook, and then use the combined notebook to interpret transferability and design tradeoffs.

## Method Summary

### Time-independent branch

The stationary branch solves

$$
-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x)
$$

with a neural network for $\psi(x)$ and a learned scalar parameter for $E$. The training objective combines PDE residual minimization with boundary behavior and problem-specific physical consistency terms.

### Time-dependent branch

The propagation branch solves

$$
i\frac{\partial \psi}{\partial t} = \left[-\frac{1}{2}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)
$$

using a dual-output network for the real and imaginary wavefunction components. The executed benchmark uses hard initial conditioning and sparse analytic guidance to stabilize learning over the whole spacetime domain.

### Comparative branch

The combined notebook reuses the same PINN logic across multiple quantum problems, then varies architecture depth, width, collocation settings, and noisy inputs to separate specialist tuning from choices that transfer more broadly.

## Repository Structure

```text
QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/
├── README.md
├── index.html
├── requirements.txt
├── data/
│   ├── molecular_vibrational_anchors.csv
│   ├── quantum_application_domains.csv
│   ├── schrodinger_sample.csv
│   └── wavepacket_application_anchors.csv
├── notebooks/
│   ├── pinn_harmonic_oscillator.ipynb
│   ├── pinn_harmonic_oscillator.html
│   ├── pinn_schrodinger.ipynb
│   ├── pinn_schrodinger.html
│   ├── quantum_pinn_combined.ipynb
│   └── quantum_pinn_combined.html
├── outputs/
│   ├── combined_arch_grid.csv
│   ├── combined_architecture.svg
│   ├── combined_noise_robustness.csv
│   ├── combined_scaling_matrix.csv
│   ├── combined_summary.csv
│   ├── combined_summary.svg
│   ├── harmonic_oscillator_loss.csv
│   ├── qho_activation_ablation.csv
│   ├── qho_benchmark.svg
│   ├── qho_collocation_ablation.csv
│   ├── qho_depth_ablation.csv
│   ├── qho_full_benchmark.csv
│   ├── schrodinger_benchmark.csv
│   ├── schrodinger_benchmark.svg
│   └── schrodinger_predictions.csv
├── src/
│   ├── data.py
│   ├── physics.py
│   ├── pinn.py
│   ├── server.py
│   └── train.py
└── website/
    ├── demo.js
    ├── index.html
    ├── README_SITE.md
    └── style.css
```

## Reproducing the Results

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
| outputs/qho_full_benchmark.csv | Stationary-state benchmark across harmonic-oscillator eigenstates |
| outputs/schrodinger_benchmark.csv | Time-resolved density and norm diagnostics for the TDSE study |
| outputs/combined_summary.csv | Standardized summary across the shared benchmark problems |
| outputs/combined_arch_grid.csv | Depth-width sweep for the comparative benchmark |
| outputs/combined_noise_robustness.csv | Robustness of the shared benchmark to noisy inputs |
| outputs/harmonic_oscillator_loss.csv | Training loss trace for the direct training module |

## License

This project is released under the terms of the LICENSE file.