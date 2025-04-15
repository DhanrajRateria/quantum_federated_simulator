"""
Quantum module for the Distributed Federated Quantum Machine Learning Simulator.

This module provides implementations of quantum circuits, data encodings,
and quantum machine learning models using PennyLane and Qiskit.
"""

from .utils import load_config, set_random_seed
from .circuits import (
    create_basic_circuit, 
    create_complex_circuit, 
    create_custom_circuit,
    create_noisy_circuit
)
from .encodings import (
    amplitude_encoding, 
    angle_encoding, 
    basis_encoding,
    iqp_feature_map
)
from .models import (
    QuantumNeuralNetwork,
    VariationalQuantumClassifier,
    QiskitVQC,
    HybridQuantumModel
)

__all__ = [
    'load_config',
    'save_circuit_diagram',
    'set_random_seed',
    'create_basic_circuit',
    'create_complex_circuit',
    'create_custom_circuit',
    'create_noisy_circuit',
    'amplitude_encoding',
    'angle_encoding',
    'basis_encoding',
    'iqp_feature_map',
    'QuantumNeuralNetwork',
    'VariationalQuantumClassifier',
    'QiskitVQC',
    'HybridQuantumModel'
]