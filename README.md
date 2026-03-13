# QuantumPINNs: Physics-Informed Neural Networks for Quantum-Relevant Physical Modeling

## Overview

This repository presents a research-oriented framework for applying Physics-Informed Neural Networks (PINNs) to quantum-relevant physical systems. The central objective is to develop machine-learning models that embed governing equations and physical constraints directly into the learning process, enabling accurate modeling of complex quantum systems even when observational data is limited, noisy, or expensive to obtain.

The framework targets physical systems governed by partial differential equations and Hamiltonians that arise in quantum simulation and related physics-constrained domains, including:

- Time-dependent and time-independent Schrödinger equations
- Quantum harmonic oscillator and anharmonic potential systems
- Hamiltonian-governed dynamical systems
- Density matrix evolution and open quantum system approximations

The workflow constructs loss functions that enforce physical residuals — such as PDE satisfaction, boundary conditions, and initial conditions — alongside data-fitting terms. This design allows models to leverage known physical structure while remaining responsive to available measurements.

---

## Motivation

Direct simulation of quantum systems is computationally expensive, and experimental data from quantum platforms is frequently sparse, noisy, or constrained by hardware limitations. These challenges motivate a learning-theoretic approach that incorporates physical structure as an inductive bias.

Physics-Informed Neural Networks provide a principled mechanism for this:

- **Governing equations** are enforced as soft constraints through residual terms in the training objective.
- **Automatic differentiation** enables exact computation of spatial and temporal derivatives required by PDEs.
- **Sparse data** is augmented by collocation points at which the physical residual is minimized, reducing dependence on dense measurements.
- **GPU acceleration** supports scalable training across high-dimensional parameter spaces.

This approach bridges statistical learning and physical simulation by constructing models that learn from incomplete information without discarding what is already known analytically about the system. For quantum-relevant problems, this is especially valuable because physical structure — symmetry, unitarity, energy conservation, Hamiltonian-governed dynamics — is often known precisely even when data is not.

The project examines the tradeoff between physical fidelity, data efficiency, and computational cost, and investigates when PINN-style training can match or surpass purely data-driven or purely numerical approaches.

---

## System Architecture

The pipeline contains three major components.

### 1. Physical Problem Formulation

Each target system is specified by:

- A governing PDE or ODE (e.g., the Schrödinger equation, quantum harmonic oscillator)
- Boundary and initial conditions
- Domain geometry and parameter ranges
- Optional measurement data

The formulation determines the structure of the physics residual and the data fidelity term in the composite loss.

### 2. Physics-Informed Neural Network

A deep neural network approximates the solution field (e.g., wavefunction ψ(x, t) or probability density |ψ|²):

- **Architecture**: Fully connected networks with tanh or sine activations, which reproduce smooth oscillatory solutions characteristic of quantum systems.
- **Collocation-based training**: The PDE residual is evaluated at a set of interior collocation points sampled from the domain, without requiring a mesh.
- **Residual-based loss**: The total loss combines the PDE residual, boundary condition violation, initial condition error, and optional data mismatch terms with tunable weights.
- **Automatic differentiation**: Spatial and temporal derivatives are computed exactly via reverse-mode autodiff (PyTorch), removing the need for finite-difference approximations.

### 3. Training and Evaluation

The training loop optimizes the composite loss using Adam or L-BFGS:

1. Sample collocation points from the domain interior and boundaries.
2. Evaluate the network and its derivatives at those points.
3. Compute the weighted sum of residual losses.
4. Backpropagate and update network parameters.
5. Evaluate on held-out test points and compare against reference solutions.

Metrics tracked include: PDE residual norm, relative L² error against analytic or numerical reference, convergence behavior under varying collocation density, and stability under observation noise.

---

## Example Applications

### Schrödinger Equation in 1D

Solve the time-dependent Schrödinger equation for a particle in a potential well:

```
iℏ ∂ψ/∂t = −(ℏ²/2m) ∂²ψ/∂x² + V(x) ψ
```

The PINN learns the wavefunction evolution over a specified time window, with boundary conditions enforced at the domain walls and an initial Gaussian wavepacket condition imposed at t = 0.

### Quantum Harmonic Oscillator

Approximate energy eigenstates and eigenvalues for the harmonic oscillator Hamiltonian:

```
H = −(ℏ²/2m) d²/dx² + (1/2) m ω² x²
```

The network is trained to satisfy the time-independent Schrödinger equation under orthonormality constraints, recovering ground-state and excited-state solutions.

### Anharmonic Potential Systems

Extend the harmonic oscillator to quartic and double-well potentials to study the impact of nonlinearity on PINN convergence and solution accuracy, and to compare collocation-based formulations with residual minimization on standard grids.

### Hamiltonian-Governed Dynamics

Learn trajectory solutions for classical and semiclassical Hamiltonian systems under energy conservation constraints, serving as a controlled benchmark for comparing physical fidelity across collocation densities and network sizes.

---

## Repository Structure

The repository separates core source code, data assets, reproducible notebooks, experimental outputs, and demonstration interfaces.

```
QuantumPINNs-Physics-Informed-Neural-Networks-for-Quantum-Relevant-Physical-Modeling/
├── README.md
├── LICENSE
├── requirements.txt
├── index.html                        ← GitHub Pages entry point
├── data/
│   └── schrodinger_sample.csv
├── notebooks/
│   ├── pinn_schrodinger.ipynb
│   ├── pinn_harmonic_oscillator.ipynb
│   └── quantum_pinn_combined.ipynb
├── outputs/
│   ├── schrodinger_predictions.csv
│   ├── harmonic_oscillator_loss.csv
│   └── collocation_residuals.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── pinn.py
│   ├── physics.py
│   ├── train.py
│   └── server.py
└── website/
    ├── README_SITE.md
    ├── demo.js
    ├── index.html
    └── style.css
```

---

## Project Summary

This project serves as a reproducible research artifact for studying physics-informed learning on quantum-relevant PDE systems. The implementation connects neural network training with explicit physical constraints — differential equation residuals, boundary conditions, and energy conservation — to investigate when and how embedding physical knowledge into the loss function improves solution accuracy, generalization under sparse data, and stability under measurement noise.

In addition to model code, the repository includes reproducible Jupyter notebooks, exported result tables, and an interactive web demo. The design is intended for interdisciplinary audiences in scientific machine learning, quantum simulation, and computational physics.

---

## Key Features

### Residual-Based Physics Enforcement

The training objective directly penalizes violations of governing differential equations at collocation points, without requiring a discretization mesh. This enables flexible domain sampling and supports adaptive refinement in regions of high residual error.

### Automatic Differentiation for Exact Derivative Computation

All spatial and temporal derivatives required by the physics residual are computed via PyTorch autograd. This eliminates finite-difference approximation error and enables seamless gradient flow through the entire computational graph.

### Collocation versus Data-Driven Comparison

The framework supports controlled experiments that vary the ratio of collocation points to observed data points. This allows direct empirical comparison of the tradeoff between physical fidelity and data efficiency across different system types.

### Stability Analysis under Sparse and Noisy Observations

Experiments systematically vary observation density, noise level, and domain coverage to characterize PINN stability. Metrics include PDE residual convergence, relative solution error, and qualitative visual comparison against reference solutions.

### GPU-Accelerated Scalable Training

The training pipeline supports GPU execution via PyTorch, enabling experiments at larger network sizes, finer collocation grids, and higher-dimensional parameter spaces without modification to the core codebase.

### Web Demo and API Interface

A Flask inference endpoint and static web interface provide an interactive environment for submitting physical parameters, running trained PINN models, and visualizing predicted solution fields and residuals.

---

## Quick Start

The following steps reproduce a baseline training run for the quantum harmonic oscillator.

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Train a PINN on the harmonic oscillator problem.

```bash
python -m src.train --problem harmonic_oscillator --epochs 5000 --collocation 2000 --model-path model.pt
```

3. Start the inference API.

```bash
python -m src.server
```

4. Optionally serve the web demo locally.

```bash
python -m http.server 8000
```

Then open `http://localhost:8000` to interact with the demo interface.

---

## Roadmap

### Near-Term

Expand the set of benchmark quantum systems, including multi-dimensional Schrödinger equations, Gross-Pitaevskii equations for Bose-Einstein condensates, and time-dependent perturbation problems. Improve convergence diagnostics and residual visualization tools.

### Mid-Term

Investigate adaptive collocation strategies that dynamically concentrate sampling in high-residual regions. Introduce hard constraint enforcement for boundary and initial conditions via network architecture (e.g., distance function multipliers) rather than soft penalty terms.

### Longer-Term

Extend the framework toward open quantum systems governed by Lindblad master equations, and explore integration with quantum hardware data for hybrid experimental-computational validation workflows.

---

## Contributing

Contributions are welcome in the form of new physical problem implementations, improved training procedures, ablation studies, notebook enhancements, and reproducibility improvements. Pull requests with clear technical motivation, documented assumptions, and validation against reference solutions are especially valuable for maintaining a rigorous and extensible research codebase.

---

## License

This project is released under the terms of the LICENSE file in this repository.

