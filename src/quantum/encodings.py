"""
Quantum data encoding strategies for embedding classical data into quantum states.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable, Dict, Any, Optional, Union, Tuple


def angle_encoding(features: np.ndarray, wires: List[int]) -> Callable:
    """
    Angle encoding: encode features as rotation angles in quantum states.
    
    Args:
        features: Input data to encode
        wires: Qubits to use for encoding
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        for i, wire in enumerate(wires):
            if i < len(features):
                qml.RX(features[i], wires=wire)
    
    return encoding


def amplitude_encoding(features: np.ndarray, wires: List[int], normalize: bool = True) -> Callable:
    """
    Amplitude encoding: encode features in the amplitudes of a quantum state.
    
    Args:
        features: Input data to encode
        wires: Qubits to use for encoding
        normalize: Whether to normalize the input data
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        # Pad features if needed
        n_qubits = len(wires)
        required_dim = 2**n_qubits
        
        padded_features = np.zeros(required_dim)
        padded_features[:min(len(features), required_dim)] = features[:min(len(features), required_dim)]
        
        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(padded_features)
            if norm > 0:
                padded_features = padded_features / norm
        
        qml.AmplitudeEmbedding(padded_features, wires=wires, normalize=False)
    
    return encoding


def basis_encoding(features: np.ndarray, wires: List[int]) -> Callable:
    """
    Basis encoding: encode binary features directly in the computational basis.
    
    Args:
        features: Binary input data to encode
        wires: Qubits to use for encoding
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        for i, wire in enumerate(wires):
            if i < len(features) and features[i] == 1:
                qml.PauliX(wire)
    
    return encoding


def iqp_feature_map(features: np.ndarray, wires: List[int], reps: int = 1) -> Callable:
    """
    IQP (Instantaneous Quantum Polynomial) feature map, similar to the one used in qiskit.
    
    Args:
        features: Input data to encode
        wires: Qubits to use for encoding
        reps: Number of repetitions of the encoding
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        n_qubits = len(wires)
        
        # Make sure we don't try to access more features than available
        features_to_use = features[:min(len(features), n_qubits)]
        
        # Apply Hadamard gates to all qubits
        for wire in wires:
            qml.Hadamard(wire)
        
        # Repeat the encoding block
        for _ in range(reps):
            # Phase rotations
            for i, wire in enumerate(wires):
                if i < len(features_to_use):
                    qml.RZ(features_to_use[i], wire)
            
            # Two-qubit ZZ rotations for all pairs
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    if i < len(features_to_use) and j < len(features_to_use):
                        qml.CNOT(wires=[wires[i], wires[j]])
                        qml.RZ(features_to_use[i] * features_to_use[j], wires[j])
                        qml.CNOT(wires=[wires[i], wires[j]])
            
            # Another Hadamard layer in between repetitions
            if _ < reps - 1:
                for wire in wires:
                    qml.Hadamard(wire)
    
    return encoding


def zz_feature_map(features: np.ndarray, wires: List[int], entanglement: str = 'linear', reps: int = 2) -> Callable:
    """
    ZZ feature map with configurable entanglement.
    
    Args:
        features: Input data to encode
        wires: Qubits to use for encoding
        entanglement: Entanglement strategy ('linear', 'circular', 'all_to_all')
        reps: Number of repetitions
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        n_qubits = len(wires)
        
        # Initial Hadamard layer
        for wire in wires:
            qml.Hadamard(wire)
        
        # Repeat the encoding block
        for _ in range(reps):
            # First rotation layer
            for i, wire in enumerate(wires):
                if i < len(features):
                    qml.RZ(features[i], wire)
            
            # Entanglement layer with ZZ rotations
            if entanglement == 'linear':
                for i in range(n_qubits - 1):
                    if i < len(features) and i+1 < len(features):
                        qml.CNOT(wires=[wires[i], wires[i+1]])
                        qml.RZ(features[i] * features[i+1], wires[i+1])
                        qml.CNOT(wires=[wires[i], wires[i+1]])
            
            elif entanglement == 'circular':
                for i in range(n_qubits):
                    next_i = (i + 1) % n_qubits
                    if i < len(features) and next_i < len(features):
                        qml.CNOT(wires=[wires[i], wires[next_i]])
                        qml.RZ(features[i] * features[next_i], wires[next_i])
                        qml.CNOT(wires=[wires[i], wires[next_i]])
            
            elif entanglement == 'all_to_all':
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        if i < len(features) and j < len(features):
                            qml.CNOT(wires=[wires[i], wires[j]])
                            qml.RZ(features[i] * features[j], wires[j])
                            qml.CNOT(wires=[wires[i], wires[j]])
    
    return encoding


def hybrid_encoding(features: np.ndarray, wires: List[int], strategy: str = 'angle_basis') -> Callable:
    """
    Hybrid encoding strategy combining multiple encoding methods.
    
    Args:
        features: Input data to encode
        wires: Qubits to use for encoding
        strategy: Which hybrid strategy to use ('angle_basis', 'amplitude_iqp')
        
    Returns:
        Encoding function that can be used in a circuit
    """
    def encoding():
        if strategy == 'angle_basis':
            # Use angle encoding for the first half of features
            # and basis encoding for the second half
            half_point = len(features) // 2
            first_half = features[:half_point]
            second_half = features[half_point:2*half_point]
            
            # Apply angle encoding
            for i, value in enumerate(first_half):
                if i < len(wires):
                    qml.RX(value, wires=wires[i])
            
            # Apply basis encoding
            for i, value in enumerate(second_half):
                if i < len(wires) and value > 0.5:  # Threshold for binary encoding
                    qml.PauliX(wires[i])
                    
        elif strategy == 'amplitude_iqp':
            # First apply amplitude encoding
            n_qubits = len(wires)
            required_dim = 2**n_qubits
            
            padded_features = np.zeros(required_dim)
            padded_features[:min(len(features), required_dim)] = features[:min(len(features), required_dim)]
            
            norm = np.linalg.norm(padded_features)
            if norm > 0:
                padded_features = padded_features / norm
            
            qml.AmplitudeEmbedding(padded_features, wires=wires, normalize=False)
            
            # Then apply a simplified IQP-like encoding
            for i, wire in enumerate(wires):
                if i < len(features):
                    qml.RZ(features[i], wire)
            
            # Apply entanglement
            for i in range(len(wires) - 1):
                qml.CNOT(wires=[wires[i], wires[i+1]])
    
    return encoding