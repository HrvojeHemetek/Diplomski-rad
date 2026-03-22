import numpy as np
from typing import List, Optional, Union, Sequence
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit.util import simulate, stateread
import warnings
warnings.filterwarnings("ignore")


class AnalogLCU:
    
    def __init__(self, coefficients: Sequence[complex], unitaries: List[QuantumCircuit], 
                 num_qubits_per_qumode: int = 3):

        if len(coefficients) != len(unitaries):
            raise ValueError("Number of coefficients must match number of unitaries.")
            
        self.coefficients = np.array(coefficients)
        self.unitaries = unitaries
        self.num_unitaries = len(unitaries)
        self.num_qubits_per_qumode = num_qubits_per_qumode
        self.cutoff = 2 ** num_qubits_per_qumode
        
        if self.num_unitaries > self.cutoff:
             raise ValueError(f"Number of unitaries ({self.num_unitaries}) exceeds qumode cutoff ({self.cutoff}).")
             
        self.norm_c = np.linalg.norm(self.coefficients)
        self.amplitudes = self.coefficients / self.norm_c

    def construct_circuit(self, initial_state_circuit: Optional[QuantumCircuit] = None) -> CVCircuit:

        num_system_qubits = self.unitaries[0].num_qubits
        qr_system = QuantumRegister(num_system_qubits, name="sys")
        qmr_ancilla = QumodeRegister(1, num_qubits_per_qumode=self.num_qubits_per_qumode, name="anc")
        
        qc = CVCircuit(qmr_ancilla, qr_system)
        
        if initial_state_circuit:
            qc.append(initial_state_circuit, qr_system)
            
        cv_amps = np.zeros(self.cutoff, dtype=complex)
        cv_amps[:self.num_unitaries] = self.amplitudes
        qc.cv_initialize(cv_amps, qmr_ancilla[0])
        
        for j in range(self.num_unitaries):
            binary_str = format(j, f'0{self.num_qubits_per_qumode}b')[::-1]
            ctrl_state = binary_str
            
            u_j_gate = self.unitaries[j].to_gate().control(
                num_ctrl_qubits=self.num_qubits_per_qumode,
                ctrl_state=ctrl_state
            )
            
            qc.append(u_j_gate, qmr_ancilla[0] + list(qr_system))
            
        return qc

    def run_simulation(self, initial_state_circuit: Optional[QuantumCircuit] = None, shots: int = 1024):

        qc = self.construct_circuit(initial_state_circuit)
        
        state, result, fockcounts = simulate(qc, shots=shots)
        
        return state, result, fockcounts

if __name__ == "__main__":
    # Demo: Implement A = 0.6 * I + 0.8 * X
    
    coeffs = [0.6, 0.8]
    
    u0 = QuantumCircuit(1)
    
    u1 = QuantumCircuit(1)
    u1.x(0)
    
    analog_lcu = AnalogLCU(coeffs, [u0, u1], num_qubits_per_qumode=2)
    state, result, fockcounts = analog_lcu.run_simulation()
    
    print("Simulation finished.")
    if fockcounts:
        print("Fock State Counts:", {k: float(v) for k, v in fockcounts.items()} )
    
    print("\n" + "="*30)
    print("Dispersive Interaction")
    print("="*30)
    
    qr = QuantumRegister(1, "sys")
    qmr = QumodeRegister(1, num_qubits_per_qumode=3, name="anc")
    qc2 = CVCircuit(qmr, qr)
    
    qc2.cv_d(1.5, qmr[0]) 
    
    qc2.cv_c_r(np.pi/4, qmr[0], qr[0])
    
    state2, _, _ = simulate(qc2)
    print("Dispersive circuit simulation finished.")
    if state2:
        stateread(state2, numberofqubits=1, numberofmodes=1, cutoff=8)

