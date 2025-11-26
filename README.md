# Neural Differential Equation Solvers (SciML)

This repository contains implementations of **Physics-Informed Neural Networks (PINNs)** and **Neural Ordinary/Partial Differential Equation (NODE/NPDE)** solvers. 

The primary research goal is to demonstrate the viability of deep learning architectures for solving boundary value problems (BVPs) and initial value problems (IVPs) in scenarios where traditional mesh-based methods (Finite Difference/Finite Element) are computationally prohibitive or limited by the "Curse of Dimensionality."

## Projects

### 1. Hard-Constraint Neural PDE Solver
**File:** `NNDE_PDE_Solvers.ipynb`

A custom implementation of a Neural Network solver for the 2D Poisson Equation and standard ODEs.

* **Methodology:** Utilizes the **Lagaris Ansatz** to mathematically encode boundary conditions directly into the trial solution structure.
    * *Trial Solution:* $\Psi(x,y) = A(x,y) + x(1-x)y(1-y)N(x,y,\theta)$
    * This formulation guarantees that the boundary conditions are satisfied by construction, allowing the optimizer to focus purely on minimizing the PDE residual.
* **Differentiation:** Implements manual backpropagation and automatic differentiation to compute higher-order derivatives (gradients and Hessians) of the neural network with respect to spatial coordinates.
* **Performance:** Achieves analytical precision ($10^{-5}$ error magnitude) on standard benchmarks, demonstrating the capability of mesh-free inference.

### 2. Advection-Diffusion Simulation
**File:** `08_Advection.ipynb`

A numerical study of transport dynamics in swirling flow fields.

* **Physics:** Solves the scalar advection equation $q_t + \mathbf{u} \cdot \nabla q = 0$.
* **Methodology:** Implements high-order flux-limiting schemes to maintain stability in convection-dominated regimes.
* **Application:** Models the transport of tracer fields (e.g., pollutants, chemical species) in complex velocity fields, a foundational problem in fluid dynamics and environmental engineering.

## Tech Stack
* **Core:** Python, NumPy, SciPy
* **Deep Learning:** TensorFlow / Keras
* **Symbolic Math:** SymPy (for exact derivative verification)
* **Visualization:** Matplotlib

## Theoretical Background
This work leverages the Universal Approximation Theorem to treat differential equations as optimization problems. By defining a custom loss function $L = ||\mathcal{N}[u] - f||^2 + \lambda||\mathcal{B}[u] - g||^2$ (where $\mathcal{N}$ is the differential operator and $\mathcal{B}$ is the boundary operator), we can train a network to approximate the solution surface $u(x,t)$.

## Author
**Travis Murphy**
*PhD Student, Applied Mathematics | University of Nevada, Reno*
*Research Focus: Control Theory & Scientific Machine Learning*
