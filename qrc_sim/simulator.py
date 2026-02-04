import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
# Try to import Aer, fallback if not present (though prompt implies using it)
try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except ImportError:
    AerSimulator = None
    NoiseModel = None

from .observables import ObservableEstimator

class QRCSimulator:
    """
    Orchestrates the QRC process:
    1. Angle Encoding
    2. Reservoir Dynamics
    3. Observable Estimation (ideal or shots)
    """
    @staticmethod
    def create_depolarizing_noise(p_error=0.01):
        """Helper to create a simple depolarizing noise model."""
        if NoiseModel is None:
            return None
        noise_model = NoiseModel()
        error = depolarizing_error(p_error, 1) # 1-qubit error
        noise_model.add_all_qubit_quantum_error(error, ['ry', 'rz', 'h'])
        return noise_model

    def __init__(self, encoder, reservoir, observable_list, 
                 backend_config='ideal', 
                 shots=1024,
                 state_update_mode='carry', # 'carry', 'reset_each_step', 'reupload_k'
                 reupload_k=3,
                 noise_model=None,
                 mode=None # Alias for state_update_mode
                 ):
        self.encoder = encoder
        self.reservoir = reservoir
        self.obs_estimator = ObservableEstimator(observable_list)
        self.backend_config = backend_config
        self.shots = shots
        self.state_update_mode = mode if mode is not None else state_update_mode
        self.reupload_k = reupload_k
        self.noise_model = noise_model
        self.n_qubits = encoder.n_qubits
        
        if backend_config == 'shots' and AerSimulator is None:
            print("Warning: qiskit_aer not found. Fallback to ideal simulator.")
            self.backend = None
        elif backend_config == 'shots':
            self.backend = AerSimulator()
    
    def _apply_basis_rotation(self, qc):
        """
        Adds rotations to the circuit to measure in X or Y basis if needed.
        Note: This assumes we want to measure different observables.
        If we have mixed Z, X, Y, we might need multiple circuits or 
        multiple measurement bases. For simplicity in teaching:
        If an observable is X, we rotate q to X basis (H).
        If multiple obs on SAME qubit in DIFFERENT bases are requested, 
        this simple approach will fail (needs separate runs).
        """
        # We'll do a simple multi-basis check:
        # If qubit i needs X, we apply H. If needs Y, we apply S† H.
        # This only works if each qubit has ONE basis requested in the list.
        bases = {}
        for obs in self.obs_estimator.observable_list:
            if obs[0] in ['X', 'Y', 'Z']:
                idx = obs[1]
                bases[idx] = obs[0]
        
        for idx, b in bases.items():
            if b == 'X':
                qc.h(idx)
            elif b == 'Y':
                qc.sdg(idx)
                qc.h(idx)
        return qc

    def run_sequence(self, sequence):
        """
        sequence: [T, input_dim]
        Returns features: [T, n_observables]
        """
        history_features = []
        current_state = Statevector.from_label('0' * self.n_qubits)
        prev_circuit = QuantumCircuit(self.n_qubits)
        
        # Buffer for re-uploading
        input_history = []
        
        for t, x_t in enumerate(sequence):
            input_history.append(x_t)
            
            if self.backend_config == 'ideal':
                # Ideal Statevector
                if self.state_update_mode == 'reupload_k':
                    # Single circuit with k inputs
                    qc = QuantumCircuit(self.n_qubits)
                    window = input_history[-self.reupload_k:]
                    for x_w in window:
                        qc.compose(self.encoder.get_circuit(x_w), inplace=True)
                        qc.compose(self.reservoir.get_circuit(), inplace=True)
                    state = Statevector.from_label('0' * self.n_qubits).evolve(qc)
                    history_features.append(self.obs_estimator.estimate_from_statevector(state))
                else:
                    if self.state_update_mode == 'reset_each_step':
                        current_state = Statevector.from_label('0' * self.n_qubits)
                    
                    # Normal Carry/Reset
                    step_qc = self.encoder.get_circuit(x_t).compose(self.reservoir.get_circuit())
                    current_state = current_state.evolve(step_qc)
                    history_features.append(self.obs_estimator.estimate_from_statevector(current_state))
                
            elif self.backend_config == 'shots':
                # Shot-based execution
                if self.state_update_mode == 'reupload_k':
                    run_qc = QuantumCircuit(self.n_qubits)
                    window = input_history[-self.reupload_k:]
                    for x_w in window:
                        run_qc.compose(self.encoder.get_circuit(x_w), inplace=True)
                        run_qc.compose(self.reservoir.get_circuit(), inplace=True)
                elif self.state_update_mode == 'carry':
                    step_qc = self.encoder.get_circuit(x_t).compose(self.reservoir.get_circuit())
                    prev_circuit.compose(step_qc, inplace=True)
                    run_qc = prev_circuit.copy()
                else: # reset_each_step
                    run_qc = self.encoder.get_circuit(x_t).compose(self.reservoir.get_circuit())
                
                # Basis Rotation
                self._apply_basis_rotation(run_qc)
                run_qc.measure_all()
                
                # Run with noise model if provided
                result = self.backend.run(run_qc, shots=self.shots, noise_model=self.noise_model).result()
                counts = result.get_counts()
                history_features.append(self.obs_estimator.estimate_from_counts(counts, self.shots))
                
        return np.array(history_features)
