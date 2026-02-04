import numpy as np
from qiskit.quantum_info import Statevector, Pauli

class ObservableEstimator:
    """
    Handles estimation of observables (features) from quantum states.
    Supports both ideal (statevector) and shot-based (counts) backends.
    """
    def __init__(self, observable_list):
        """
        observable_list: List of tuples.
           ('Z', i) -> Expectation of Z on qubit i
           ('ZZ', i, j) -> Correleation Z_i Z_j
        """
        self.observable_list = observable_list
        
    def estimate_from_statevector(self, state: Statevector):
        """
        Exact calculation using matrix multiplication.
        """
        results = []
        n_qubits = state.num_qubits
        
        for obs_spec in self.observable_list:
            op_type = obs_spec[0]
            
            if op_type == 'Z':
                idx = obs_spec[1]
                val = state.expectation_value(self._get_pauli_op(n_qubits, idx, 'Z')).real
                results.append(val)
            elif op_type == 'X':
                idx = obs_spec[1]
                val = state.expectation_value(self._get_pauli_op(n_qubits, idx, 'X')).real
                results.append(val)
            elif op_type == 'Y':
                idx = obs_spec[1]
                val = state.expectation_value(self._get_pauli_op(n_qubits, idx, 'Y')).real
                results.append(val)
            elif op_type == 'ZZ':
                idx1, idx2 = obs_spec[1], obs_spec[2]
                val = state.expectation_value(self._get_pauli_zz(n_qubits, idx1, idx2)).real
                results.append(val)
                
        return np.array(results)

    def estimate_from_counts(self, counts, total_shots, basis_info=None):
        """
        Statistical estimation from measurement counts.
        basis_info: Optional dict mapping observable index to basis used ('Z', 'X', 'Y').
                    If we measured everything in Z, we expect basis_info to reflect that
                    X/Y were handled by pre-rotation gates in the circuit.
        """
        results = []
        parsed_counts = {}
        for bstr, count in counts.items():
            bits = [int(c) for c in reversed(bstr)]
            parsed_counts[tuple(bits)] = count
            
        for i, obs_spec in enumerate(self.observable_list):
            op_type = obs_spec[0]
            val_accum = 0.0
            
            # For X and Y, if the circuit had the correct basis rotation, 
            # they are estimated just like Z.
            if op_type in ['Z', 'X', 'Y']:
                idx = obs_spec[1]
                for bits, count in parsed_counts.items():
                    sign = 1 - 2 * bits[idx]
                    val_accum += sign * count
                    
            elif op_type == 'ZZ':
                idx1, idx2 = obs_spec[1], obs_spec[2]
                for bits, count in parsed_counts.items():
                    sign = 1 - 2 * (bits[idx1] ^ bits[idx2])
                    val_accum += sign * count
            
            results.append(val_accum / total_shots)
            
        return np.array(results)

    def _get_pauli_op(self, n, idx, label):
        s = ['I'] * n
        s[-(idx+1)] = label
        return Pauli("".join(s))

    def _get_pauli_zz(self, n, idx1, idx2):
        s = ['I'] * n
        s[-(idx1+1)] = 'Z'
        if s[-(idx2+1)] == 'Z':
             s[-(idx2+1)] = 'I'
        else:
             s[-(idx2+1)] = 'Z'
        return Pauli("".join(s))
