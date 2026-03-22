
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Operator
from collections import defaultdict


class LCUOptimizer:
    
    def __init__(self, hamiltonian: SparsePauliOp):

        self.hamiltonian = hamiltonian.simplify()
        self.num_qubits = hamiltonian.num_qubits
        self._pauli_terms = self._extract_pauli_terms()
        
    def _extract_pauli_terms(self) -> List[Tuple[str, complex]]:
        
        terms = []
        for pauli_label, coeff in zip(self.hamiltonian.paulis.to_labels(), 
                                       self.hamiltonian.coeffs):
            terms.append((pauli_label, coeff))
        return terms
    
    def compute_one_norm(self, coefficients: Optional[np.ndarray] = None) -> float:

        if coefficients is None:
            coefficients = self.hamiltonian.coeffs
        return np.sum(np.abs(coefficients))
    
    def _pauli_to_symplectic(self, pauli_label: str) -> Tuple[int, int]:
        
        x_mask = 0
        z_mask = 0
        
        for i, char in enumerate(reversed(pauli_label)):
            if char == 'X':
                x_mask |= (1 << i)
            elif char == 'Z':
                z_mask |= (1 << i)
            elif char == 'Y':
                x_mask |= (1 << i)
                z_mask |= (1 << i)
                
        return x_mask, z_mask

    @staticmethod
    def _symplectic_commute(p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:

        x1, z1 = p1
        x2, z2 = p2
        
        term = (x1 & x2) ^ (z1 & z2)
        
        if hasattr(int, "bit_count"):
             return term.bit_count() % 2 == 0
        else:
             return bin(term).count('1') % 2 == 0

    def _check_commutes_with_family(self, term_symplectic: Tuple[int, int], family_symplectics: List[Tuple[int, int]]) -> bool:
        
        for member in family_symplectics:
            if not self._symplectic_commute(term_symplectic, member):
                return False
        return True

    def fast_pauli_grouping(self) -> List[SparsePauliOp]:
        
        sorted_terms = []
        for i, (label, coeff) in enumerate(self._pauli_terms):
            mag = abs(coeff)
            symp = self._pauli_to_symplectic(label)
            sorted_terms.append((mag, i, symp))
            
        sorted_terms.sort(key=lambda x: x[0], reverse=True)
        
        families: List[List[Tuple[int, Tuple[int, int]]]] = []
        
        for mag, idx, term_symp in sorted_terms:
            placed = False
            
            for family in families:
                family_symplectics = [item[1] for item in family]
                
                if self._check_commutes_with_family(term_symp, family_symplectics):
                    family.append((idx, term_symp))
                    placed = True
                    break
            
            if not placed:
                families.append([(idx, term_symp)])
                
        result = []
        for family in families:
            group_terms = []
            group_coeffs = []
            
            for idx, _ in family:
                label, coeff = self._pauli_terms[idx]
                group_terms.append(label)
                group_coeffs.append(coeff)
                
            result.append(SparsePauliOp(group_terms, group_coeffs))
            
        return result
    
    def optimize_coefficients(self) -> Tuple[List[float], List[SparsePauliOp]]:
        
        groups = self.fast_pauli_grouping()
        
        coefficients = []
        unitaries = []
        
        for group in groups:
            for label, coeff in zip(group.paulis.to_labels(), group.coeffs):
                mag = abs(coeff)
                if mag < 1e-10:
                    continue
                    
                coefficients.append(float(mag))
                
                phase = coeff / mag
                unitaries.append(SparsePauliOp(label, coeffs=[phase]))
        
        return coefficients, unitaries
    
    def get_optimization_stats(self) -> Dict[str, float]:

        original_norm = self.compute_one_norm()
        
        opt_coeffs, _ = self.optimize_coefficients()
        optimized_norm = np.sum(np.abs(opt_coeffs))
        
        original_cost = original_norm ** 4
        optimized_cost = optimized_norm ** 4
        
        return {
            "original_1_norm": original_norm,
            "optimized_1_norm": optimized_norm,
            "norm_reduction_ratio": optimized_norm / original_norm if original_norm > 0 else 1.0,
            "sampling_cost_reduction": optimized_cost / original_cost if original_cost > 0 else 1.0,
            "num_original_terms": len(self._pauli_terms),
            "num_optimized_groups": len(self.fast_pauli_grouping())
        }

def create_example_hamiltonian(num_qubits: int = 4, 
                               model: str = "heisenberg") -> SparsePauliOp:
    
    terms = []
    coeffs = []
    
    if model == "heisenberg":
        # 1D Heisenberg: H = sum_i (XX + YY + ZZ)
        for i in range(num_qubits - 1):
            for pauli in ['X', 'Y', 'Z']:
                # Create term like "IIXXII" for XX on sites i, i+1
                pauli_str = ['I'] * num_qubits
                pauli_str[i] = pauli
                pauli_str[i + 1] = pauli
                terms.append(''.join(reversed(pauli_str)))  # Qiskit uses LSB order
                coeffs.append(1.0)
                
    elif model == "ising":
        # Transverse-field Ising: H = -sum_i ZZ - h*sum_i X
        h = 0.5
        for i in range(num_qubits - 1):
            pauli_str = ['I'] * num_qubits
            pauli_str[i] = 'Z'
            pauli_str[i + 1] = 'Z'
            terms.append(''.join(reversed(pauli_str)))
            coeffs.append(-1.0)
        
        for i in range(num_qubits):
            pauli_str = ['I'] * num_qubits
            pauli_str[i] = 'X'
            terms.append(''.join(reversed(pauli_str)))
            coeffs.append(-h)
            
    elif model == "random":
        np.random.seed(42)
        num_terms = 3 * num_qubits
        paulis = ['I', 'X', 'Y', 'Z']
        
        for _ in range(num_terms):
            pauli_str = ''.join(np.random.choice(paulis, num_qubits))
            if pauli_str != 'I' * num_qubits:  # Skip identity
                terms.append(pauli_str)
                coeffs.append(np.random.randn())
    
    return SparsePauliOp(terms, coeffs).simplify()


def generate_taylor_lcu(hamiltonian: SparsePauliOp, time: float, 
                        epsilon: float = 1e-3, norm: float = 1.0) -> SparsePauliOp:

    from math import factorial
    
    # 1. Calculate segmentation parameters
    beta = np.sum(np.abs(hamiltonian.coeffs))

    r_val = (beta * time) ** 2
    r = int(np.ceil(r_val)) if r_val > 1 else 1
    
    tau = time / r
    
    # 2. Determine Taylor order K for one segment
    k = determine_taylor_parameters(hamiltonian, norm, time, epsilon=epsilon)
    
    # 3. Construct One Segment U_seg = sum (-i tau H)^k / k!
    num_qubits = hamiltonian.num_qubits
    identity = SparsePauliOp("I" * num_qubits, coeffs=[1.0])
    u_seg = identity
    
    h_power = identity
    
    for order_idx in range(1, k + 1):
        h_power = h_power.compose(hamiltonian).simplify()
        
        coeff = ((-1j * tau) ** order_idx) / factorial(order_idx)
        
        scaled_term = SparsePauliOp(
            h_power.paulis.to_labels(),
            h_power.coeffs * coeff
        )
        
        u_seg = u_seg + scaled_term
        u_seg = u_seg.simplify()
        
    # 4. Compose Segments: W = (U_seg)^r    
    final_op = u_seg
        
    for _ in range(r - 1):
        final_op = final_op.compose(u_seg).simplify()
            
    return final_op


def calculate_required_repetitions(norm: float, epsilon: float) -> int:
    
    if epsilon <= 0:
        return 1000
        
    t_val = np.e * (norm ** 2) / (epsilon ** 2)
    return int(np.ceil(t_val))


def prepare_lcu_from_taylor(taylor_op: SparsePauliOp) -> tuple:
    
    coefficients = []
    unitaries = []
    
    for label, coeff in zip(taylor_op.paulis.to_labels(), taylor_op.coeffs):
        magnitude = np.abs(coeff)
        if magnitude < 1e-15:
            continue 
            
        phase_angle = np.angle(coeff)
        phase = np.exp(1j * phase_angle)
            
        coefficients.append(float(magnitude))
        unitaries.append(SparsePauliOp(label, coeffs=[phase]))
        
        if abs(abs(phase) - 1.0) > 1e-10:
             print(f"DEBUG: Non-unitary phase detected! Label: {label}, Coeff: {coeff}, Mag: {magnitude}, Phase: {phase}, AbsPhase: {abs(phase)}")
    
    return coefficients, unitaries


def determine_taylor_parameters(hamiltonian: SparsePauliOp, norm: float, time: float, 
                                epsilon: float = 1e-3) -> int:
    beta = np.sum(np.abs(hamiltonian.coeffs))
    r_val = (beta * time) ** 2
    
    return int(np.ceil(np.log(6 * r_val * norm / epsilon) / np.log(np.log(6 * r_val * norm / epsilon))))



