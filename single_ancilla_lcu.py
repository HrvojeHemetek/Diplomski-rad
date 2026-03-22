import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Operator, SparsePauliOp
from typing import Optional, TYPE_CHECKING, List

if TYPE_CHECKING:
    from lcu_optimizer import LCUOptimizer

class SingleAncillaLCU:

    def __init__(self, lcu_coefficients, unitaries, sampler=None):

        self.sampler = sampler
        
        if self.sampler:
            self.norm_c = self.sampler.lcu_norm
            self.coefficients = None
            self.unitaries = None
            self.num_unitaries = None
            self.probs = None
        else:
            if len(lcu_coefficients) != len(unitaries):
                raise ValueError("Number of coefficients must match number of unitaries.")
            
            self.coefficients = np.array(lcu_coefficients)
            self.unitaries = unitaries
            self.norm_c = np.sum(np.abs(self.coefficients))
            self.probs = np.abs(self.coefficients) / self.norm_c
            self.num_unitaries = len(unitaries)

    def sample_unitaries(self):
        
        if self.sampler:
            U1 = self.sampler.sample_unitary()
            U2 = self.sampler.sample_unitary()
            return U1, U2
        else:
            idx1 = np.random.choice(self.num_unitaries, p=self.probs)
            idx2 = np.random.choice(self.num_unitaries, p=self.probs)
            return idx1, idx2

    def construct_circuit(self, u1, u2, initial_state_circuit, observable_circuit=None):

        num_system_qubits = initial_state_circuit.num_qubits
        
        ancilla = QuantumRegister(1, 'ancilla')
        system = QuantumRegister(num_system_qubits, 'system')
        
        qc = QuantumCircuit(ancilla, system)

        qc.h(ancilla[0])

        qc.compose(initial_state_circuit, qubits=system, inplace=True)

        def apply_controlled_pauli(qc, pauli_op, control_qubit, system_qubits, ctrl_state):
            
            for label, coeff in zip(pauli_op.paulis.to_labels(), pauli_op.coeffs):
            
                phase_angle = np.angle(coeff)
            
                if ctrl_state == 0:
                    qc.x(control_qubit)
                    
                qc.p(phase_angle, control_qubit)
                
                for i, p_char in enumerate(reversed(label)):
                    if i >= len(system_qubits): break
                    target = system_qubits[i]
                    if p_char == 'X':
                        qc.cx(control_qubit, target)
                    elif p_char == 'Y':
                        qc.cy(control_qubit, target)
                    elif p_char == 'Z':
                        qc.cz(control_qubit, target)
                
                if ctrl_state == 0:
                    qc.x(control_qubit)

        if self.sampler:
            op1 = u1
            op2 = u2
        else:
            op1 = self.unitaries[u1]
            op2 = self.unitaries[u2]

        apply_controlled_pauli(qc, op1, ancilla[0], system, ctrl_state=1)

        apply_controlled_pauli(qc, op2, ancilla[0], system, ctrl_state=0)

        qc.h(ancilla[0])
        
        if observable_circuit:
            qc.compose(observable_circuit, qubits=system, inplace=True)

        return qc

    def estimate_expectation(self, initial_state_circuit, observable, num_shots=1024, repetitions=100, estimator=None, unitary = False):

        if estimator is None:
            estimator = StatevectorEstimator()

        op_Z_ancilla = SparsePauliOp("Z")
        full_observable = observable ^ op_Z_ancilla
        
        pubs_mu = []
        for _ in range(repetitions):
            u1, u2 = self.sample_unitaries()
            qc = self.construct_circuit(u1, u2, initial_state_circuit)
            pubs_mu.append((qc, full_observable))
            
        job_mu = estimator.run(pubs_mu)
        result_mu = job_mu.result()
        
        mu_sum = sum([pub_result.data.evs for pub_result in result_mu])
        mu_avg = mu_sum / repetitions
        mu_estimate = (self.norm_c ** 2) * mu_avg

        if unitary:
            return mu_estimate
        
        num_system_qubits = initial_state_circuit.num_qubits
        op_I_system = SparsePauliOp("I" * num_system_qubits)
        full_observable_I = op_I_system ^ op_Z_ancilla 
        
        pubs_l = []
        for _ in range(repetitions):
            u1, u2 = self.sample_unitaries()
            qc = self.construct_circuit(u1, u2, initial_state_circuit)
            pubs_l.append((qc, full_observable_I))
            
        job_l = estimator.run(pubs_l)
        result_l = job_l.result()
        
        l_sum = sum([pub_result.data.evs for pub_result in result_l])
        l_avg = l_sum / repetitions
        l_estimate = (self.norm_c ** 2) * l_avg
        
        if abs(l_estimate) < 1e-9:
            return 0.0 
            
        return mu_estimate / l_estimate

    @classmethod
    def from_hamiltonian(cls, hamiltonian: SparsePauliOp, 
                         optimize: bool = True):

        if optimize:
            from lcu_optimizer import LCUOptimizer
            optimizer = LCUOptimizer(hamiltonian)
            coefficients, unitaries = optimizer.optimize_coefficients()
            return cls(coefficients, unitaries)
        else:
            coefficients = list(np.abs(hamiltonian.coeffs))
            unitaries = []
            for label, coeff in zip(hamiltonian.paulis.to_labels(), 
                                     hamiltonian.coeffs):
                phase = coeff / abs(coeff) if abs(coeff) > 0 else 1.0
                unitaries.append(SparsePauliOp(label, coeffs=[phase]))
            return cls(coefficients, unitaries)
