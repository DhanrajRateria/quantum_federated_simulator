"""
Data encoding schemes for quantum machine learning.
This module provides methods to encode classical data into quantum states.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable

# Import quantum libraries based on configuration
try:
    import qiskit
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

class DataEncoder:
    """Class for encoding classical data into quantum states."""
    
    def __init__(self, n_qubits: int, encoding_type: str = 'angle'):
        """
        Initialize the data encoder.
        
        Args:
            n_qubits: Number of qubits to use for encoding
            encoding_type: Type of encoding strategy to use
        """
        self.n_qubits = n_qubits
        self.encoding_type = encoding_type
        
        if encoding_type not in ['angle', 'amplitude', 'basis', 'hybrid']:
            raise ValueError(f"Unsupported encoding type: {encoding_type}")
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize input data to appropriate range for quantum encoding.
        
        Args:
            data: Input data to normalize
            
        Returns:
            Normalized data
        """
        if self.encoding_type == 'angle':
            # Scale to [0, 2π] for rotation gates
            return 2 * np.pi * (data - np.min(data)) / (np.max(data) - np.min(data))
        elif self.encoding_type == 'amplitude':
            # Normalize to unit vector for amplitude encoding
            return data / np.linalg.norm(data)
        else:
            # Default normalization to [0, 1]
            return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess data to fit encoding requirements.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data
        """
        # Flatten if multi-dimensional
        flat_data = data.flatten()
        
        # Handle dimensionality for different encoding types
        if self.encoding_type == 'angle':
            # Pad or truncate to match qubit count
            if len(flat_data) < self.n_qubits:
                return np.pad(flat_data, (0, self.n_qubits - len(flat_data)))
            else:
                return flat_data[:self.n_qubits]
                
        elif self.encoding_type == 'amplitude':
            # For amplitude encoding, we need 2^n_qubits amplitudes
            required_dim = 2**self.n_qubits
            if len(flat_data) < required_dim:
                return np.pad(flat_data, (0, required_dim - len(flat_data)))
            else:
                return flat_data[:required_dim]
        
        # Default for other encoding types
        return flat_data
    
    def encode_qiskit(self, data: np.ndarray) -> QuantumCircuit:
        """
        Encode classical data into a Qiskit quantum circuit.
        
        Args:
            data: Input data to encode
            
        Returns:
            Quantum circuit with encoded data
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required but not installed.")
            
        # Preprocess and normalize
        processed_data = self.preprocess_data(data)
        normalized_data = self.normalize_data(processed_data)
        
        # Create circuit
        circuit = QuantumCircuit(self.n_qubits)
        
        if self.encoding_type == 'angle':
            # Angle encoding uses rotation gates
            for i, value in enumerate(normalized_data):
                if i < self.n_qubits:
                    circuit.rx(value, i)
                    
        elif self.encoding_type == 'amplitude':
            # Amplitude encoding requires initialization to specified amplitudes
            # This is a simplified version - full amplitude encoding is more complex
            from qiskit.extensions import Initialize
            init_gate = Initialize(normalized_data)
            circuit.append(init_gate, range(self.n_qubits))
            
        elif self.encoding_type == 'basis':
            # Basis encoding: binary representation (0->|0⟩, 1->|1⟩)
            binary_data = (normalized_data > 0.5).astype(int)
            for i, bit in enumerate(binary_data):
                if i < self.n_qubits and bit == 1:
                    circuit.x(i)
        
        return circuit
    
    def encode_pennylane(self, data: np.ndarray) -> Callable:
        """
        Create a PennyLane encoding function for classical data.
        
        Args:
            data: Input data to encode
            
        Returns:
            Function that applies the encoding in a PennyLane circuit
        """
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required but not installed.")
            
        # Preprocess and normalize
        processed_data = self.preprocess_data(data)
        normalized_data = self.normalize_data(processed_data)
        
        if self.encoding_type == 'angle':
            def encoding_function():
                # Angle encoding uses rotation gates
                for i, value in enumerate(normalized_data):
                    if i < self.n_qubits:
                        qml.RX(value, wires=i)
                        
        elif self.encoding_type == 'amplitude':
            def encoding_function():
                # Amplitude encoding in PennyLane
                qml.AmplitudeEmbedding(normalized_data, wires=range(self.n_qubits), normalize=True)
                
        elif self.encoding_type == 'basis':
            def encoding_function():
                # Basis encoding: binary representation
                binary_data = (normalized_data > 0.5).astype(int)
                for i, bit in enumerate(binary_data):
                    if i < self.n_qubits and bit == 1:
                        qml.PauliX(wires=i)
        
        return encoding_function
    
    def encode(self, data: np.ndarray, backend: str = 'qiskit') -> Union[object, Callable]:
        """
        Encode classical data for the specified backend.
        
        Args:
            data: Input data to encode
            backend: Quantum backend ('qiskit' or 'pennylane')
            
        Returns:
            Encoded circuit or function
        """
        if backend == 'qiskit':
            return self.encode_qiskit(data)
        elif backend == 'pennylane':
            return self.encode_pennylane(data)
        else:
            raise ValueError(f"Unsupported backend: {backend}")