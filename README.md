# QuantumPINNs: Physics-Informed Neural Networks for Quantum-Relevant Physical Modeling

> A physics-informed learning framework for stationary and time-dependent quantum systems, developed as a unified project spanning eigenvalue recovery, complex wavepacket propagation, cross-problem benchmarking, architecture ablations, and robustness analysis.

---

## Abstract

This project studies whether physics-informed neural networks can serve as a credible computational framework for quantum differential equations when the governing physics is known but dense labeled solution data is either unnecessary or expensive to generate. The central idea is to treat the Schrodinger equation itself as the supervision signal: neural networks are trained by minimizing differential-equation residuals, normalization constraints, and physically motivated consistency terms rather than by fitting large precomputed datasets.

The project covers both major one-dimensional settings. For the time-independent Schrodinger equation, it trains neural networks to recover wavefunctions and eigenvalues simultaneously. For the time-dependent Schrodinger equation, it learns complex-valued spacetime dynamics for a Gaussian wavepacket under free-particle evolution. These specialist studies are then consolidated into a broader multi-problem benchmark that evaluates cross-problem transfer, architecture sensitivity, collocation scaling, and robustness to noisy initial-condition data.

The strongest accuracy-first results are obtained in the two specialist notebooks. In the harmonic-oscillator study, the ground-state PINN reaches relative L2 error $1.56934 \times 10^{-3}$, max pointwise error $1.25322 \times 10^{-3}$, learned energy $0.50001526$, absolute energy error $1.52588 \times 10^{-5}$, fidelity $0.99999754$, and Rayleigh-consistency gap $3.59893 \times 10^{-6}$. In the time-dependent study, the upgraded ComplexPINN reaches initial density relative L2 error $8.0 \times 10^{-8}$, mean density relative L2 error $4.18\%$ over $t \in [0,1]$, final-time density relative L2 error $6.71\%$, norm range $[0.990997, 1.010868]$, mean Ehrenfest position error $2.56 \times 10^{-2}$, mean Ehrenfest momentum error $2.99 \times 10^{-2}$, and five-snapshot probability-current L2 error $7.494 \times 10^{-2}$. The combined benchmark complements these specialist results by showing that a shared PINN framework remains informative across four quantum problems, with the best depth-width configuration reaching relative L2 error $0.26585$ and the noise study varying by only $0.00619$ in relative L2 between noise amplitudes $0.00$ and $0.20$.

---

## Project Highlights

| Theme | Result |
|---|---|
| Best stationary-state accuracy | QHO ground state: rel-L2 $1.56934 \times 10^{-3}$, $\hat{E}_0 = 0.50001526$, $|\Delta E| = 1.52588 \times 10^{-5}$, fidelity $0.99999754$ |
| Best time-dependent accuracy | TDSE Gaussian wavepacket: initial density rel-L2 $8.0 \times 10^{-8}$, mean rel-L2 $4.18\%$, final-time rel-L2 $6.71\%$ |
| Strongest physical consistency check | TDSE norm range $[0.990997, 1.010868]$ with mean Ehrenfest errors $2.56 \times 10^{-2}$ for position and $2.99 \times 10^{-2}$ for momentum |
| Broadest benchmark evidence | Four-problem benchmark covering QHO ground state, QHO first excited state, anharmonic confinement, and double-well tunneling |
| Best architecture in the combined ablation | 5 layers x 64 hidden units, rel-L2 $0.26585$ |
| Noise robustness | Relative L2 changes from $0.25654$ to $0.25035$ as noise amplitude increases from $0.00$ to $0.20$ |
| Cross-problem energy estimates | QHO $(n=0)$: $0.54976$; QHO $(n=1)$: $1.56084$; anharmonic: $0.57282$; double well: $0.23770$ |

---

## Why This Project Matters

Three research questions organize the project.

1. Can a neural network recover quantum eigenstates and eigenvalues directly from the governing differential equation, without labeled wavefunction data?
2. Can the same framework be extended to complex-valued, time-dependent quantum dynamics while preserving the physical structure of the solution?
3. When the problem class changes, which modeling decisions remain robust and which are only problem-specific optimizations?

The project answers these questions through a layered experimental design. Two notebooks establish accuracy-first results for representative stationary and time-dependent quantum systems. A third notebook then provides the comparative evidence needed to evaluate transferability, architecture sensitivity, and robustness in a standardized multi-problem setting.

---

## Main Contributions

1. A high-accuracy PINN formulation for the quantum harmonic oscillator that jointly recovers the wavefunction and eigenvalue under an accuracy-oriented training regime.
2. A dual-output ComplexPINN for the time-dependent Schrodinger equation that recovers real and imaginary wavefunction components simultaneously and validates them using norm conservation, probability current, and Ehrenfest diagnostics.
3. A unified benchmark harness that compares multiple quantum problems under a common training design, making it possible to distinguish specialist accuracy from broadly transferable behavior.
4. Controlled ablations over architecture depth, collocation density, optimization budget, and input noise, converting the project from a set of isolated examples into a reproducible design study.
5. A complete research-to-interface path through notebooks, exported CSV artifacts, a lightweight inference API, and a browser-based demonstration.

---

## Three-Notebook Evidence Structure

| Notebook | Role in the project | Best reported result | Contribution |
|---|---|---|---|
| `notebooks/pinn_harmonic_oscillator.ipynb` | Accuracy-first stationary-state benchmark | rel-L2 $1.56934 \times 10^{-3}$; learned energy $0.50001526$; energy error $1.52588 \times 10^{-5}$; fidelity $0.99999754$ | Establishes that the time-independent Schrodinger equation can be solved to near-analytic precision with a physics-informed neural formulation |
| `notebooks/pinn_schrodinger.ipynb` | Accuracy-first time-dependent benchmark | initial density rel-L2 $8.0 \times 10^{-8}$; mean rel-L2 $4.18\%$; final-time rel-L2 $6.71\%$; norm range $[0.990997, 1.010868]$ | Establishes that the same project can handle complex-valued spacetime dynamics with strong physical-consistency diagnostics |
| `notebooks/quantum_pinn_combined.ipynb` | Comparative benchmark and ablation layer | best architecture rel-L2 $0.26585$; noise study $0.25654 \rightarrow 0.25035$ | Explains how the framework behaves across problem classes and which design choices remain effective under broader transfer |

---

## Quantitative Summary

### Accuracy-First Harmonic Oscillator Result

| Metric | Value |
|---|---|
| Relative L2 error | $1.56934 \times 10^{-3}$ |
| Max pointwise error | $1.25322 \times 10^{-3}$ |
| Learned energy | $0.50001526$ |
| Exact energy | $0.50000000$ |
| Absolute energy error | $1.52588 \times 10^{-5}$ |
| Fidelity | $0.99999754$ |
| Rayleigh energy | $0.50001166$ |
| Rayleigh-consistency gap | $3.59893 \times 10^{-6}$ |

These numbers define the strongest single stationary-state result in the project and show that the harmonic-oscillator PINN can recover both wavefunction shape and eigenvalue to near-analytic precision.

### Accuracy-First Time-Dependent Schrodinger Result

| Metric | Value |
|---|---|
| Initial density rel-L2 | $8.0 \times 10^{-8}$ |
| Mean density rel-L2 over $t \in [0,1]$ | $4.182754\%$ |
| Final-time density rel-L2 | $6.709014\%$ |
| Norm range | $[0.990997, 1.010868]$ |
| Mean Ehrenfest position error | $2.5570 \times 10^{-2}$ |
| Mean Ehrenfest momentum error | $2.9903 \times 10^{-2}$ |
| Five-snapshot current L2 error | $7.4940 \times 10^{-2}$ |

This result shows that the time-dependent branch is not only visually plausible, but quantitatively constrained by conservation and transport diagnostics.

### Standardized Combined Benchmark

| Problem | Learned quantity | Reference | Delta | Relative L2 | Wall time |
|---|---|---|---|---|---|
| QHO $(n=0)$ | $0.54976$ | $0.50000$ | $0.04976$ | $0.11963$ | $2.14$ s |
| QHO $(n=1)$ | $1.56084$ | $1.50000$ | $0.06084$ | $0.13641$ | $2.29$ s |
| Anharmonic quartic well | $0.57282$ | $0.53750$ | $0.03532$ | - | $2.56$ s |
| Double well | $0.23770$ | about $0.300$ | - | - | $2.46$ s |

The combined benchmark should be interpreted differently from the specialist notebooks. It is a transferability and ablation study, not the final accuracy ceiling for each individual problem.

### Architecture and Noise Findings

| Ablation | Result |
|---|---|
| Best architecture | 5 layers x 64 units, rel-L2 $0.26585$ |
| Depth 2, width 32 | rel-L2 $1.36282$ |
| Depth 3, width 64 | rel-L2 $0.69420$ |
| Depth 5, width 32 | rel-L2 $0.88432$ |
| Noise amplitude $0.00$ | rel-L2 $0.25654$ |
| Noise amplitude $0.05$ | rel-L2 $0.25498$ |
| Noise amplitude $0.20$ | rel-L2 $0.25035$ |

The ablation layer supports two conclusions: depth matters strongly in the standard benchmark, and the physics-informed objective is comparatively stable under moderate noise in the supplied initial-condition data.

---

## Method Overview

### Time-Independent Branch

The stationary-state branch solves

$$
-\frac{1}{2}\psi''(x) + V(x)\psi(x) = E\psi(x)
$$

with a neural network that represents the wavefunction and a scalar parameter that represents the learned energy. The optimizer therefore updates both the function approximation and the eigenvalue simultaneously. Normalization, symmetry structure, and Rayleigh consistency are incorporated where appropriate in the accuracy-first notebook.

### Time-Dependent Branch

The time-dependent branch solves

$$
i\frac{\partial \psi}{\partial t} = \left[-\frac{1}{2}\frac{\partial^2}{\partial x^2} + V(x)\right]\psi(x,t)
$$

through a dual-output network that predicts the real and imaginary parts of the wavefunction. In the executed benchmark configuration, a hard initial-condition construction and sparse analytic anchor terms are used to stabilize full-domain propagation accuracy while preserving the physics-informed structure of the training objective.

### Comparative Benchmark Layer

The combined notebook applies shared PINN ideas across four problem classes and uses ablations to distinguish three effects:

1. representational capacity through depth-width architecture changes,
2. optimization and quadrature effects through collocation and epoch sweeps,
3. robustness to imperfect inputs through noisy initial-condition experiments.

---

## Notebook Guide

### `notebooks/pinn_harmonic_oscillator.ipynb`

This notebook is the project's highest-accuracy stationary-state study. It should be read when the priority is precision on a canonical quantum eigenvalue problem.

- Focus: QHO ground state, excited states, symmetry-aware regularization, Rayleigh consistency
- Strongest result: rel-L2 $1.56934 \times 10^{-3}$, learned energy $0.50001526$
- Impact: shows that an accuracy-oriented PINN can recover the exact harmonic-oscillator physics to near-analytic precision

### `notebooks/pinn_schrodinger.ipynb`

This notebook is the project's highest-accuracy time-dependent study. It should be read when the priority is complex-valued propagation and physical-consistency diagnostics.

- Focus: Gaussian wavepacket propagation, dual-output ComplexPINN, current and Ehrenfest diagnostics
- Strongest result: initial density rel-L2 $8.0 \times 10^{-8}$, mean rel-L2 $4.18\%$, final-time rel-L2 $6.71\%$
- Impact: shows that the project can move beyond stationary states and still preserve physically interpretable structure

### `notebooks/quantum_pinn_combined.ipynb`

This notebook is the integrative benchmark study. It should be read when the priority is breadth, transferability, and design guidance rather than single-problem optimization.

- Focus: four-problem benchmark, architecture grid, scaling study, noise robustness
- Strongest result: 5 layers x 64 units reaches rel-L2 $0.26585$ in the standard benchmark grid
- Impact: explains which parts of the framework transfer across qualitatively different quantum problems and which settings are most robust

---

## Project Structure

```text
QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/
├── README.md
├── requirements.txt
├── index.html
├── data/
│   └── schrodinger_sample.csv
├── notebooks/
│   ├── pinn_harmonic_oscillator.ipynb
│   ├── pinn_schrodinger.ipynb
│   └── quantum_pinn_combined.ipynb
├── outputs/
│   ├── combined_arch_grid.csv
│   ├── combined_noise_robustness.csv
│   ├── combined_scaling_matrix.csv
│   ├── combined_summary.csv
│   ├── harmonic_oscillator_loss.csv
│   ├── qho_activation_ablation.csv
│   ├── qho_collocation_ablation.csv
│   ├── qho_depth_ablation.csv
│   ├── qho_full_benchmark.csv
│   ├── schrodinger_benchmark.csv
│   └── schrodinger_predictions.csv
├── src/
│   ├── data.py
│   ├── physics.py
│   ├── pinn.py
│   ├── server.py
│   └── train.py
└── website/
    ├── index.html
    ├── style.css
    ├── demo.js
    └── README_SITE.md
```

---

## Reproducing the Results

```bash
pip install -r requirements.txt

# Execute the notebooks in the recommended order
jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_harmonic_oscillator.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/pinn_schrodinger.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/quantum_pinn_combined.ipynb

# Optional: run the training and serving modules directly
python -m src.train
python -m src.server
```

For the clearest reading path, start with the harmonic-oscillator notebook for the stationary-state accuracy result, continue to the time-dependent notebook for complex propagation, and then use the combined notebook to interpret transferability, ablations, and robustness across the full project.

---

## Generated Artifacts

| File | Purpose |
|---|---|
| `outputs/qho_full_benchmark.csv` | Harmonic-oscillator benchmark and multi-state results |
| `outputs/qho_activation_ablation.csv` | Activation-function comparison for the stationary-state branch |
| `outputs/qho_depth_ablation.csv` | Depth ablation for the stationary-state branch |
| `outputs/qho_collocation_ablation.csv` | Collocation-density ablation for the stationary-state branch |
| `outputs/schrodinger_benchmark.csv` | Time-resolved TDSE density and norm metrics |
| `outputs/schrodinger_predictions.csv` | Pointwise TDSE prediction samples |
| `outputs/combined_summary.csv` | Four-problem benchmark summary |
| `outputs/combined_arch_grid.csv` | Depth-width architecture grid |
| `outputs/combined_noise_robustness.csv` | Noise-robustness sweep |
| `outputs/combined_scaling_matrix.csv` | Collocation-versus-epochs scaling matrix |

---

## Interpretation

The project supports a clear overall conclusion. Physics-informed neural networks are not limited here to a single demonstration on an easy quantum system. In this project, they recover a canonical stationary-state benchmark to near-analytic precision, propagate a complex-valued wavepacket with strong conservation behavior, and remain informative under broader cross-problem comparisons and ablation studies. The specialist notebooks establish the strongest accuracy claims; the combined benchmark explains why those claims matter in the wider methodological picture.

---

## License

This project is released under the terms of the LICENSE file.
