"""
Utility functions for quantum module.
"""

import os
import yaml
import random
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Union, List, Tuple
import qiskit
from qiskit.visualization import circuit_drawer


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Integer seed for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    qiskit.utils.algorithm_globals.random_seed = seed
    

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dict containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_circuit_diagram(circuit: Union[qml.tape.QuantumTape, qiskit.QuantumCircuit], 
                         filename: str, 
                         format_type: str = 'png') -> None:
    """
    Save a visualization of the quantum circuit.
    
    Args:
        circuit: PennyLane or Qiskit quantum circuit
        filename: Output filename without extension
        format_type: Output format (png, pdf, etc.)
    """
    if isinstance(circuit, qml.tape.QuantumTape):
        # For PennyLane circuits
        fig, ax = qml.draw_mpl(circuit)
        fig.savefig(f"{filename}.{format_type}", bbox_inches='tight')
        plt.close(fig)
    elif isinstance(circuit, qiskit.QuantumCircuit):
        # For Qiskit circuits
        circuit_diagram = circuit_drawer(circuit, output='mpl')
        circuit_diagram.savefig(f"{filename}.{format_type}", bbox_inches='tight')
        plt.close(circuit_diagram)
    else:
        raise TypeError("Circuit must be a PennyLane tape or Qiskit QuantumCircuit")


def get_device(device_name: str, wires: int, shots: Optional[int] = None, **kwargs) -> qml.Device:
    """
    Get a PennyLane device with specific parameters.
    
    Args:
        device_name: Name of the device (default.qubit, lightning.qubit, qiskit.aer, etc.)
        wires: Number of qubits
        shots: Number of measurement shots (None for exact simulation)
        **kwargs: Additional device-specific parameters
        
    Returns:
        Initialized PennyLane device
    """
    return qml.device(device_name, wires=wires, shots=shots, **kwargs)


def circuit_to_qiskit(circuit: qml.tape.QuantumTape) -> qiskit.QuantumCircuit:
    """
    Convert a PennyLane circuit to a Qiskit QuantumCircuit.
    
    Args:
        circuit: PennyLane quantum circuit
        
    Returns:
        Equivalent Qiskit QuantumCircuit
    """
    from pennylane_qiskit import to_qiskit
    return to_qiskit(circuit)


def calculate_circuit_depth(circuit: Union[qml.tape.QuantumTape, qiskit.QuantumCircuit]) -> int:
    """
    Calculate the depth of a quantum circuit.
    
    Args:
        circuit: PennyLane or Qiskit quantum circuit
        
    Returns:
        Circuit depth
    """
    if isinstance(circuit, qml.tape.QuantumTape):
        # Convert to Qiskit to use its depth calculation
        qiskit_circuit = circuit_to_qiskit(circuit)
        return qiskit_circuit.depth()
    elif isinstance(circuit, qiskit.QuantumCircuit):
        return circuit.depth()
    else:
        raise TypeError("Circuit must be a PennyLane tape or Qiskit QuantumCircuit")