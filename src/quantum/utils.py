"""
Utility functions for quantum module.
"""

import os
import yaml
import random
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumScript, QuantumTape
# Optional imports with error handling
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
try:
    import qiskit
    from qiskit.visualization import circuit_drawer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
try:
    from pennylane_qiskit import to_qiskit
    PENNYLANE_QISKIT_AVAILABLE = True
except ImportError:
    PENNYLANE_QISKIT_AVAILABLE = False

from typing import Dict, Any, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across random, numpy, and PennyLane.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    qml.numpy.random.seed(seed) # PennyLane's numpy
    logger.debug(f"Set random, numpy, and PennyLane seeds to {seed}")
    # Removed outdated Qiskit global seed setting


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dict containing configuration parameters

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_path}: {e}")
        raise

def get_device(device_name: str, wires: int, shots: Optional[int] = None, **kwargs) -> qml.device:
    """
    Get a PennyLane device with specific parameters.

    Args:
        device_name: Name of the device (e.g., "default.qubit", "lightning.qubit").
        wires: Number of qubits.
        shots: Number of measurement shots (None for exact analytical results).
        **kwargs: Additional device-specific parameters.

    Returns:
        Initialized PennyLane device.

    Raises:
        qml.DeviceError: If the device cannot be loaded or initialized.
    """
    try:
        logger.info(f"Loading PennyLane device '{device_name}' with wires={wires}, shots={shots}, kwargs={kwargs}")
        device = qml.device(device_name, wires=wires, shots=shots, **kwargs)
        logger.info(f"Device '{device_name}' loaded successfully.")
        return device
    except qml.DeviceError as e:
        logger.error(f"Failed to load PennyLane device '{device_name}': {e}", exc_info=True)
        raise


def circuit_to_qiskit(tape: qml.tape.QuantumTape) -> qiskit.QuantumCircuit:
    """
    Convert a PennyLane QuantumTape to a Qiskit QuantumCircuit.

    Requires the pennylane-qiskit plugin to be installed.

    Args:
        tape: PennyLane quantum tape.

    Returns:
        Equivalent Qiskit QuantumCircuit.

    Raises:
        ImportError: If pennylane-qiskit is not installed.
        TypeError: If the input is not a PennyLane QuantumTape.
        Exception: For errors during conversion.
    """
    if not isinstance(tape, qml.tape.QuantumTape):
        raise TypeError(f"Input must be a PennyLane QuantumTape, got {type(tape)}")

    if not PENNYLANE_QISKIT_AVAILABLE:
        raise ImportError("pennylane-qiskit plugin is required for conversion. Please install it.")

    try:
        logger.info("Converting PennyLane tape to Qiskit circuit...")
        qiskit_circuit = to_qiskit(tape)
        logger.info("Conversion successful.")
        return qiskit_circuit
    except Exception as e:
        logger.error(f"Error converting PennyLane tape to Qiskit circuit: {e}", exc_info=True)
        raise


def calculate_circuit_depth(circuit: Union[qml.tape.QuantumTape, qiskit.QuantumCircuit]) -> int:
    """
    Calculate the depth of a quantum circuit using Qiskit's definition.

    For PennyLane tapes, it first converts them to Qiskit circuits.

    Args:
        circuit: PennyLane QuantumTape or Qiskit QuantumCircuit.

    Returns:
        Circuit depth.

    Raises:
        TypeError: If the input type is not supported.
        ImportError: If Qiskit (or pennylane-qiskit for tapes) is not installed.
        Exception: For errors during conversion or depth calculation.
    """
    logger.info(f"Calculating depth for circuit of type {type(circuit)}")
    if isinstance(circuit, qml.tape.QuantumTape):
        if not QISKIT_AVAILABLE:
             raise ImportError("Qiskit is required to calculate depth for PennyLane tapes (via conversion). Please install it.")
        try:
            qiskit_circuit = circuit_to_qiskit(circuit)
            # Use optimization_level=0 for depth calculation to avoid modification
            depth = qiskit_circuit.depth(optimization_level=0)
            logger.info(f"Calculated depth for converted PennyLane tape: {depth}")
            return depth
        except Exception as e:
            logger.error(f"Error calculating depth for PennyLane tape: {e}", exc_info=True)
            raise
    elif QISKIT_AVAILABLE and isinstance(circuit, qiskit.QuantumCircuit):
        try:
            # Use optimization_level=0 for depth calculation to avoid modification
            depth = circuit.depth(optimization_level=0)
            logger.info(f"Calculated depth for Qiskit circuit: {depth}")
            return depth
        except Exception as e:
            logger.error(f"Error calculating depth for Qiskit circuit: {e}", exc_info=True)
            raise
    else:
        if not QISKIT_AVAILABLE and type(circuit).__name__ == 'QuantumCircuit':
             raise ImportError("Qiskit is required to process Qiskit QuantumCircuit objects. Please install it.")
        raise TypeError(f"Unsupported circuit type for depth calculation: {type(circuit)}")

# --- Optional: Add utility to calculate expected parameters ---
def calculate_expected_params(n_qubits: int, n_layers: int, gate_set: List[str], entanglement: str) -> int:
    """Calculates the expected number of *trainable* parameters for a custom circuit."""
    # This calculation depends heavily on how create_custom_circuit assigns parameters.
    # Based on the current create_custom_circuit: only single-qubit RX, RY, RZ in the gate_set
    # contribute to trainable parameters within the layers loop.
    trainable_params_per_layer = 0
    parameterized_single_qubit_gates = [g.lower() for g in gate_set if g.lower() in ['rx', 'ry', 'rz']]

    # Each parameterized single-qubit gate is applied to each qubit in each layer
    trainable_params_per_layer = n_qubits * len(parameterized_single_qubit_gates)

    total_params = trainable_params_per_layer * n_layers
    logger.debug(f"Calculated expected params: {total_params} ({n_qubits} qubits, {n_layers} layers, "
                 f"param_gates={parameterized_single_qubit_gates})")
    return total_params