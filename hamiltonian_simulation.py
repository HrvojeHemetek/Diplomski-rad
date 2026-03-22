import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector, Operator
from scipy.linalg import expm

from single_ancilla_lcu import SingleAncillaLCU
from lcu_optimizer import (
    create_example_hamiltonian,
    generate_taylor_lcu,
    prepare_lcu_from_taylor,
    calculate_required_repetitions,
    LCUOptimizer
)


def exact_time_evolution_expectation(
    hamiltonian: SparsePauliOp,
    initial_state: QuantumCircuit,
    observable: SparsePauliOp,
    time: float
) -> float:

    H_matrix = hamiltonian.to_matrix()
    O_matrix = observable.to_matrix()
    
    U = expm(-1j * H_matrix * time)
    
    psi0 = Statevector.from_instruction(initial_state)
    psi0_vec = psi0.data
    
    psi_t = U @ psi0_vec
    
    expectation = np.real(np.conj(psi_t) @ O_matrix @ psi_t)
    
    return expectation


def run_simulation(
    num_qubits: int = 2,
    model: str = "heisenberg",
    time: float = 0.5,
    epsilon: float = 1e-3,
    num_shots: int = 1024,
    repetitions: int = None,
    optimize_lcu: bool = False,
    verbose: bool = True
) -> dict:

    if verbose:
        print(f"=" * 60)
        print(f"Hamiltonian Simulation via Single-Ancilla LCU")
        print(f"=" * 60)
        print(f"Model: {model}, Qubits: {num_qubits}, Total Time: {time}")
    
    H = create_example_hamiltonian(num_qubits, model)
    beta = np.sum(np.abs(H.coeffs))
    
    if verbose:
        print(f"Hamiltonian has {len(H.paulis)} Pauli terms")
        print(f"Hamiltonian 1-norm (beta): {beta:.4f}")

    qc_init = QuantumCircuit(num_qubits)
    observable = SparsePauliOp("Z" * num_qubits)
    
    lcu = None
    coeffs_norm = 0.0
    
    if verbose:
        print(f"Using Explicit Taylor Expansion.")
        
    taylor_op = generate_taylor_lcu(H, time, epsilon, np.abs(np.linalg.eigvalsh(observable.to_matrix())).max())
    
    if optimize_lcu:
        if verbose:
            print("Optimizing Taylor expansion via Pauli grouping...")
        opt = LCUOptimizer(taylor_op)
        coefficients, unitaries = opt.optimize_coefficients()
    else:
        coefficients, unitaries = prepare_lcu_from_taylor(taylor_op)
    
    lcu = SingleAncillaLCU(coefficients, unitaries)
    coeffs_norm = lcu.norm_c

    if verbose:
        print(f"LCU Coeff 1-norm: {coeffs_norm:.4f}")
        
    if repetitions is None:
        repetitions = calculate_required_repetitions(np.abs(np.linalg.eigvalsh(observable.to_matrix())).max(), epsilon)
        if verbose:
            print(f"Calculated repetitions for epsilon {epsilon}: {repetitions}")

    if verbose:
        print(f"Running estimation...")
        
    estimated_value = lcu.estimate_expectation(qc_init, observable, num_shots=num_shots, repetitions=repetitions, unitary=True)
    
    exact_value = exact_time_evolution_expectation(H, qc_init, observable, time)
    error = abs(estimated_value - exact_value)
    
    if verbose:
        print(f"Est: {estimated_value:.6f}, Exact: {exact_value:.6f}")
        print(f"Error: {error:.6f}")
        print("="*60)
        
    return {
        "estimated": estimated_value,
        "exact": exact_value,
        "absolute_error": error,
        "relative_error": error / abs(exact_value) if abs(exact_value) > 1e-10 else error
    }


def main():
    """Main entry point for demonstration."""
    
    OPTIMIZE = False

    print("=" * 60)
    print(f"Demo 1: Small Heisenberg Chain (2 qubits) [Opt={OPTIMIZE}]")
    print("=" * 60)
    result1 = run_simulation(
            num_qubits=2,
            model="heisenberg",
            time=0.3,
            epsilon=1e-1,
            optimize_lcu=OPTIMIZE,
            verbose=False
        )

    
    print("\n")
    print("=" * 60)
    print(f"Demo 2: Transverse-field Ising (3 qubits) [Opt={OPTIMIZE}]")
    print("=" * 60)
    result2 = run_simulation(
            num_qubits=3,
            model="ising",
            time=0.2,
            epsilon=1e-1,
            optimize_lcu=OPTIMIZE,
            verbose=False
        )

    
    print("\n")
    print("Summary:")
    print(f"  Heisenberg (2q): Error (Rel) = {result1['relative_error']*100:.2f}%")
    print(f"  Ising (3q): Error (Rel) = {result2['relative_error']*100:.2f}%")

    print("\n")
    print("=" * 60)
    print(f"Demo 3: Long Time (Force r > 1) [Opt={OPTIMIZE}]")
    print("=" * 60)
    result3 = run_simulation(
        num_qubits=2,
        model="heisenberg",
        time=1.0,
        epsilon=1e-1,
        optimize_lcu=OPTIMIZE,
        verbose=False
    )
    
    print(f"  Long Time (2q, t=1.0): Error (Rel) = {result3['relative_error']*100:.2f}%")


if __name__ == "__main__":
    main()
