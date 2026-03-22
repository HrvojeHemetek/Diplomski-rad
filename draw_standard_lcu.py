import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

def draw_standard_lcu():
    k = 3 
    n = 4 
    
    ancilla_wires = list(range(k))
    system_wires = list(range(k, k+n))
    all_wires = ancilla_wires + system_wires
    
    dev = qml.device('default.qubit', wires=all_wires)
    
    @qml.qnode(dev)
    def circuit():
        qml.QubitUnitary(np.eye(2**k), wires=ancilla_wires, id="R")
        
        try:
            qml.Barrier(wires=all_wires)
        except AttributeError:
            pass
            
        qml.QubitUnitary(np.eye(2**(k+n)), wires=all_wires, id="Q")
        
        try:
            qml.Barrier(wires=all_wires)
        except AttributeError:
            pass
            
        qml.QubitUnitary(np.eye(2**k), wires=ancilla_wires, id="R†")
        
        return qml.state()

    print("Standard LCU Circuit Representation (PennyLane Big-Endian Convention):")
    drawer = qml.draw(circuit, show_all_wires=True)
    print(drawer())
    
    try:
        fig, ax = qml.draw_mpl(circuit, show_all_wires=True)( )
        
        fig.suptitle("Kvantni krug standardnog algoritamskog elementa LCU", fontsize=16)
        
        fig.savefig('standard_lcu_circuit.png')
        print("\nCircuit diagram saved to 'standard_lcu_circuit.png'")
    except Exception as e:
        print(f"\nCould not save image (matplotlib issues or missing dependencies): {e}")

if __name__ == "__main__":
    draw_standard_lcu()
