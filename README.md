# QuantumPINNs: Physics-Informed Neural Networks for Quantum-Relevant Physical Modeling

[![Project Website](https://img.shields.io/badge/Project%20Website-GitHub%20Pages-0969da?style=for-the-badge)](https://thmolena.github.io/QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/)

> A single-paper PINN study for quantum modeling: specialist stationary-state accuracy, physics-checked time-dependent propagation, and a shared comparative benchmark that separates reusable design choices from one-problem tuning.

## Abstract

This repository is organized as one coherent paper supported by three executed notebooks. The paper asks whether physics-informed neural networks can recover quantum states and quantum dynamics from governing equations and sparse physical structure, without relying on dense supervised solution labels.

The empirical story is intentionally cumulative. The harmonic-oscillator notebook establishes the best specialist accuracy result in the repository, the time-dependent Schrödinger notebook shows that the same framework supports complex-valued wavepacket propagation with quantitative physical diagnostics, and the combined benchmark evaluates which modeling choices remain defensible when the problem class changes. Read together, the notebooks support a single claim: physics-structured inductive bias improves both the credibility and the reproducibility of PINN-based quantum modeling.

For a NeurIPS audience, the most important point is not merely that a PINN performs well on one canonical benchmark. The important point is that the modeling claim is explicit, the evidence is comparative rather than anecdotal, the ablations identify what matters, and the released artifacts make the argument reproducible.

## Central Paper Claim

The submitted paper makes four linked claims.

1. Physics-structured inductive bias can produce near-analytic stationary-state recovery on a canonical quantum eigenproblem.
2. The same framework can support complex-valued time-dependent propagation with density, norm, phase, and transport-aware diagnostics.
3. A shared benchmark is necessary to distinguish specialist tuning from design choices that transfer across problem classes.
4. The strongest machine-learning contribution is therefore comparative evidence and reproducibility, not a single benchmark win in isolation.

## Headline Results

| Evidence layer | Headline result | Why it matters |
|---|---|---|
| Harmonic oscillator | Relative L2 error $1.56927 \times 10^{-3}$, learned energy $0.50001526$, absolute energy error $1.52588 \times 10^{-5}$ | Establishes the paper's strongest specialist accuracy result and shows near-analytic recovery of both state shape and eigenvalue |
| Time-dependent Schrödinger | Initial density relative L2 error $7.92 \times 10^{-8}$, final density relative L2 error $5.66 \times 10^{-2}$, predicted norm close to unity | Shows that the framework extends beyond static eigenproblems to controlled complex-valued dynamics |
| Combined benchmark | Best shared architecture is 5 layers x 64 units with relative L2 $0.26585$ and modest degradation under added noise | Supplies the paper's most NeurIPS-relevant evidence: which design choices remain effective across tasks |

## Best Visual Evidence

These figures are generated from committed CSV artifacts in `outputs/`, so the README and landing page remain synchronized and stable.

### Figure 1. Harmonic-oscillator benchmark

![Harmonic oscillator benchmark](outputs/qho_benchmark.svg)

This figure carries the strongest single numerical claim in the repository. The ground state reaches relative L2 error $1.56927 \times 10^{-3}$ with very low energy error, and the executed notebook expands that result with the highest-value specialist visualizations: `qho_ground_state_analysis.png`, `qho_overlap_matrix.png`, and `qho_ablation.png`.

### Figure 2. Time-dependent Schrödinger benchmark

![Time-dependent Schrodinger benchmark](outputs/schrodinger_benchmark.svg)

This figure summarizes the dynamics claim. The PINN is essentially exact at the initial slice and remains controlled at later times while preserving norm behavior close to one. The executed notebook adds the richer visual record through `schrodinger_density_heatmap.png`, `schrodinger_phase.png`, `schrodinger_current_snapshots.png`, and `schrodinger_ehrenfest.png`.

### Figure 3. Shared architecture sweep

![Combined architecture sweep](outputs/combined_architecture.svg)

This figure is where the paper moves beyond one-problem performance reporting. It identifies depth as the strongest lever in the tested shared benchmark and isolates the 5-layer, 64-unit model as the best architecture under the common protocol.

### Figure 4. Transferability and robustness summary

![Combined benchmark summary](outputs/combined_summary.svg)

This is the most important figure for the overall machine-learning narrative. It shows that the shared formulation remains interpretable across multiple quantum problems and degrades only modestly under added input noise. The combined notebook supplements it with `combined_scaling_study.png`, `combined_noise_study.png`, and `combined_summary_barchart.png`.

## How the Three Notebooks Support One Paper

| Notebook | Paper role | Best figures to inspect | Main claim it supports |
|---|---|---|---|
| `notebooks/pinn_harmonic_oscillator.ipynb` | Specialist stationary-state evidence | Ground-state analysis, overlap matrix, ablation summary, analytic eigenstates | Physics-structured PINNs can recover a canonical eigenstate with near-analytic accuracy |
| `notebooks/pinn_schrodinger.ipynb` | Specialist time-dependent evidence | Density heatmap, phase plot, exact-vs-PINN snapshots, Ehrenfest diagnostics | The same framework supports complex-valued quantum dynamics with physically meaningful checks |
| `notebooks/quantum_pinn_combined.ipynb` | Comparative machine-learning evidence | Architecture sweep, scaling study, noise robustness, shared summary bars | The broader claim should be judged by transferability, ablation, and robustness rather than a single favorable benchmark |

Recommended reading order:

1. Start with the harmonic-oscillator notebook for the cleanest specialist accuracy claim.
2. Continue to the time-dependent Schrödinger notebook for the strongest dynamics evidence.
3. Finish with the combined notebook, where the paper's broader comparative and transferability claims are tested.

## Method at a Glance

### Stationary-state branch

The stationary branch solves

$$
-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x)
$$

with a neural network for $\psi(x)$ and a learned scalar parameter for $E$. The formulation emphasizes hard physical structure, boundary behavior, parity control, and operator-consistency terms that make the eigenvalue claim quantitatively defensible.

### Time-dependent branch

The propagation branch solves

$$
i\frac{\partial \psi}{\partial t} = \left[-\frac{1}{2}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)
$$

using a dual-output network for the real and imaginary components. The evaluation focuses on density fidelity, norm preservation, phase structure, and transport-adjacent behavior rather than only pointwise visual agreement.

### Comparative branch

The combined notebook reuses the same PINN logic across multiple quantum-relevant settings, then varies depth, width, scaling, collocation behavior, and input noise to separate reusable modeling choices from benchmark-specific tuning.

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