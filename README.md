# Hamiltonian Simulation via Linear Combination of Unitaries (LCU)

This repository contains Python code for simulating Hamiltonian dynamics on quantum computers using various **Linear Combination of Unitaries (LCU)** techniques. It serves as a comprehensive framework for experimenting with single-ancilla, and analog LCU approaches, described in the paper "Implementing any Linear Combination of Unitaries on Intermediate-term Quantum Computers" by Shantanav Chakraborty, using Qiskit.

## Project Structure

The project is structured into distinct modules for optimization, simulation implementation, circuit visualization, and classical benchmarking:

### Core Modules

*   **`lcu_optimizer.py`**
    Contains logic for preprocessing and optimizing LCU deployments. Key features include techniques to reduce the $L_1$-norm of coefficients and minimize circuit depth by grouping mutually commuting Pauli operators as well as generating a Truncated Taylor series expansion of a Hamiltonian. It features the following clustering algorithm:
    *   **Fast Partitioning (PSFAM):** A high-performance sorting and greedy integration algorithm that uses symplectic representation and bitwise checks to optimally cluster Pauli strings described in "Fast Partitioning of Pauli Strings into Commuting Families for Optimal Expectation Value Measurements of Dense Operators" by Reggio et al.

*   **`single_ancilla_lcu.py`**
    Implements a Single-Ancilla Linear Combination of Unitaries protocol. It includes tools to sample from a Truncated Taylor series expansion of a Hamiltonian and construct corresponding Qiskit circuits, extracting expected values over multiple iterations using the `StatevectorEstimator`.

*   **`analog_lcu.py`**
    An implementation of the "Analog LCU" protocol using `bosonic_qiskit` (C2QA). It leverages continuous-variable (CV) harmonic oscillator modes (qumodes) interacting with standard qubits to enact operations without multiple ancilla qubits, efficiently encoding unitaries into multimode superpositions using Fock state encoding.

### Execution and Benchmarking

*   **`hamiltonian_simulation.py`**
    The main driver script meant for demonstration or scientific testing. It sets up Hamiltonians (like the 1D Heisenberg model or Transverse-field Ising model) and runs explicit Truncated Taylor series expansion LCU protocols. It compares the simulation estimates against the exact matrix exponentiation expected values.

*   **`stats.py`**
    Reads benchmarking results from `results.csv` and uses `matplotlib`/`pandas` to plot grouped bar charts representing the average errors and success rates, comparing unoptimized vs. optimized LCU simulation configurations.

### Utilities

*   **`draw_standard_lcu.py`**: A helper script utilizing `PennyLane` to create a standard algorithmic block diagram of an LCU circuit mapping.
*   **`endianness_diagram.py`**: A helper to generate diagrams or code relating to bitwise order/Qiskit conventions.

## Prerequisites

To run the implementations, ensure you have the following packages installed:
*   `qiskit`
*   `numpy`
*   `scipy`
*   `pandas`
*   `matplotlib`
*   `pennylane` (for circuit drawing utilities)
*   [`bosonic_qiskit` (C2QA)](https://github.com/C2QA/bosonic-qiskit) (for running the Analog LCU simulations)

## Usage

You can run the main simulation benchmark directly from the terminal:

```bash
python hamiltonian_simulation.py
```

This will run several demos comparing estimated vs. exact simulation outcomes under optimized and unoptimized conditions.

To evaluate algorithmic benchmarking figures:

```bash
python stats.py
```
