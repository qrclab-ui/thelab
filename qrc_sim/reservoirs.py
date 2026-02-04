import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class RandomReservoir:
    """
    A Quantum Reservoir with fixed random unitary dynamics.
    Structure: Layers of Entanglement (Cnot) + Random Rotations (Ry, Rz).
    """
    def __init__(self, n_qubits, depth, entanglement='ring', seed=42):
        self.n_qubits = n_qubits
        self.depth = depth
        self.entanglement = entanglement
        self.rng = np.random.RandomState(seed)
        
        # Generate fixed random weights for the reservoir
        # 2 params per qubit per depth layer (Ry, Rz)
        self.num_weights = self.n_qubits * self.depth * 2
        self.weights = self.rng.uniform(0, 2*np.pi, self.num_weights)
        
    def get_circuit(self):
        """
        Returns the reservoir QuantumCircuit (unitary U_res).
        This circuit is constant (parameters are fixed values).
        """
        qc = QuantumCircuit(self.n_qubits)
        
        param_idx = 0
        for d in range(self.depth):
            # 1. Entanglement Layer
            if self.entanglement == 'ring':
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)
            elif self.entanglement == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)
            
            # 2. Rotation Layer (Ry, Rz) with fixed random weights
            for i in range(self.n_qubits):
                theta_y = self.weights[param_idx]
                theta_z = self.weights[param_idx + 1]
                
                qc.ry(theta_y, i)
                qc.rz(theta_z, i)
                
                param_idx += 2
                
        return qc

class RandomCRotReservoir:
    """
    A stronger Reservoir using controlled rotations with random fixed angles.
    Structure: Layers of (CRX/CRY/CRZ) + Random Local Rotations.
    """
    def __init__(self, n_qubits, depth, seed=42):
        self.n_qubits = n_qubits
        self.depth = depth
        self.rng = np.random.RandomState(seed)
        
        # Random angles for both local and controlled rotations
        # Local: 2 per qubit per layer (Ry, Rz)
        # Controlled: 1 per adjacent pair per layer (CRy)
        self.num_local = self.n_qubits * self.depth * 2
        self.num_ctrl = self.n_qubits * self.depth
        
        self.local_weights = self.rng.uniform(0, 2*np.pi, self.num_local)
        self.ctrl_weights = self.rng.uniform(0, 2*np.pi, self.num_ctrl)
        
    def get_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        
        l_idx = 0
        c_idx = 0
        for d in range(self.depth):
            # 1. Randomized Entanglement (CRy)
            for i in range(self.n_qubits):
                target = (i + 1) % self.n_qubits
                angle = self.ctrl_weights[c_idx]
                qc.cry(angle, i, target)
                c_idx += 1
            
            # 2. Local Random Rotations
            for i in range(self.n_qubits):
                qc.ry(self.local_weights[l_idx], i)
                qc.rz(self.local_weights[l_idx + 1], i)
                l_idx += 2
                
        return qc
