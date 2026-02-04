import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

class AngleEncoder:
    """
    Encodes data into quantum states using Angle Encoding (Ry rotations).
    """
    def __init__(self, n_qubits, scale=1.0):
        self.n_qubits = n_qubits
        self.scale = scale
        
    def get_circuit(self, input_params):
        """
        Returns a QuantumCircuit that encodes the input_params.
        input_params: A ParameterVector or list of parameters.
        """
        qc = QuantumCircuit(self.n_qubits)
        for i in range(self.n_qubits):
             # Handle input dimension mismatch by cycling
             param_val = input_params[i % len(input_params)]
             qc.ry(param_val * self.scale, i)
             
        return qc

class ReuploadingEncoder:
    """
    A more expressive encoder that 're-uploads' features with interleaved rotations.
    This provides more non-linearity than single-layer angle encoding.
    """
    def __init__(self, n_qubits, layers=2, scale=np.pi):
        self.n_qubits = n_qubits
        self.layers = layers
        self.scale = scale
        # Fixed random rotations to interleave between data re-uploads
        self.rng = np.random.RandomState(42)
        self.rotations = self.rng.uniform(0, 2*np.pi, (layers - 1, n_qubits))

    def get_circuit(self, input_params):
        qc = QuantumCircuit(self.n_qubits)
        
        for l in range(self.layers):
            # 1. Input rotation
            for i in range(self.n_qubits):
                param_val = input_params[i % len(input_params)]
                qc.ry(param_val * self.scale, i)
            
            # 2. Interleaved fixed rotation (if not last layer)
            if l < self.layers - 1:
                for i in range(self.n_qubits):
                    qc.rz(self.rotations[l, i], i)
                    
        return qc
