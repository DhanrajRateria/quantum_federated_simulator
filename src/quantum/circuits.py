"""
Quantum circuit implementations for the Federated Quantum ML Simulator.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from .utils import load_config, get_device


def create_basic_circuit(n_qubits: int, n_layers: int = 1) -> Callable:
    """
    Create a basic parameterized quantum circuit with alternating rotations and entanglement.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of repeated layers
        
    Returns:
        Circuit function that accepts parameters
    """
    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Basic circuit implementation with rotation and entanglement layers.
        
        Args:
            params: Circuit parameters
            features: Optional input features for data encoding
        """
        # Data encoding block (if features are provided)
        if features is not None:
            for i in range(min(n_qubits, len(features))):
                qml.RX(features[i], wires=i)
        
        # Compute the number of parameters per layer
        params_per_layer = n_qubits * 3  # 3 rotation gates per qubit
        
        # Apply layers of parameterized gates
        for layer in range(n_layers):
            # Parameter offset for current layer
            offset = layer * params_per_layer
            
            # Rotation layer
            for qubit in range(n_qubits):
                qml.RX(params[offset + qubit*3], wires=qubit)
                qml.RY(params[offset + qubit*3 + 1], wires=qubit)
                qml.RZ(params[offset + qubit*3 + 2], wires=qubit)
            
            # Entanglement layer (CZ gates in a ring topology)
            for qubit in range(n_qubits):
                qml.CZ(wires=[qubit, (qubit + 1) % n_qubits])
                
        # Measure all qubits in the computational basis
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit


def create_complex_circuit(n_qubits: int, n_layers: int = 2) -> Callable:
    """
    Create a more complex parameterized quantum circuit with different gate types
    and more sophisticated entanglement structure.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of repeated layers
        
    Returns:
        Circuit function that accepts parameters
    """
    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Complex circuit implementation with various gates and entanglement patterns.
        
        Args:
            params: Circuit parameters
            features: Optional input features for data encoding
        """
        # Data encoding block
        if features is not None:
            for i in range(min(n_qubits, len(features))):
                qml.RX(features[i], wires=i)
        
        # Calculate parameters per layer
        rotation_params_per_qubit = 3  # RX, RY, RZ
        entanglement_params_per_pair = 2  # For controlled rotation gates
        
        # Calculate pairs for entanglement
        pairs = []
        for i in range(n_qubits):
            # Connect each qubit to the next one (ring topology)
            pairs.append((i, (i + 1) % n_qubits))
            # Add some long-range connections for more complex entanglement
            if n_qubits > 3:
                pairs.append((i, (i + n_qubits // 2) % n_qubits))
        
        params_per_layer = (n_qubits * rotation_params_per_qubit) + (len(pairs) * entanglement_params_per_pair)
        
        # Apply layers
        for layer in range(n_layers):
            # Parameter offset for current layer
            offset = layer * params_per_layer
            
            # Rotation block
            for qubit in range(n_qubits):
                param_idx = offset + qubit * rotation_params_per_qubit
                qml.RX(params[param_idx], wires=qubit)
                qml.RY(params[param_idx + 1], wires=qubit)
                qml.RZ(params[param_idx + 2], wires=qubit)
            
            # Entanglement block
            entanglement_offset = offset + n_qubits * rotation_params_per_qubit
            for i, (control, target) in enumerate(pairs):
                param_idx = entanglement_offset + i * entanglement_params_per_pair
                # Use controlled rotations instead of just CZ
                qml.CRX(params[param_idx], wires=[control, target])
                qml.CRY(params[param_idx + 1], wires=[control, target])
            
            # Add non-parameterized gates for additional expressivity
            if layer < n_layers - 1:  # Skip in the last layer
                for i in range(n_qubits):
                    if i % 2 == 0:  # Alternate between qubits
                        qml.Hadamard(wires=i)
        
        # Measurement strategy - mix of single-qubit and two-qubit observables
        measurements = []
        
        # Single-qubit observables
        for i in range(n_qubits):
            measurements.append(qml.expval(qml.PauliZ(i)))
        
        # Two-qubit observables for the first few qubit pairs (if we have enough qubits)
        if n_qubits >= 4:
            measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
            measurements.append(qml.expval(qml.PauliX(0) @ qml.PauliX(2)))
        
        return measurements
    
    return circuit


def create_custom_circuit(config_path: str) -> Callable:
    """
    Create a quantum circuit based on a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Circuit function built according to the configuration
    """
    config = load_config(config_path)
    n_qubits = config.get('n_qubits', 4)
    n_layers = config.get('n_layers', 2)
    gate_set = config.get('gate_set', ['rx', 'ry', 'rz', 'cz'])
    entanglement = config.get('entanglement', 'linear')
    measurement = config.get('measurement', 'z')
    
    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Custom circuit implementation based on configuration.
        
        Args:
            params: Circuit parameters
            features: Optional input features for data encoding
        """
        # Data encoding block
        if features is not None:
            encoding_type = config.get('encoding', 'angle')
            if encoding_type == 'angle':
                for i in range(min(n_qubits, len(features))):
                    qml.RX(features[i], wires=i)
            elif encoding_type == 'amplitude':
                # Normalize features for amplitude encoding
                normalized_features = features / np.linalg.norm(features)
                qml.AmplitudeEmbedding(normalized_features, wires=range(n_qubits), pad_with=0.0, normalize=True)
        
        # Parameter counters
        param_idx = 0
        
        # Create the circuit architecture
        for layer in range(n_layers):
            # Rotation layer
            for qubit in range(n_qubits):
                for gate in gate_set:
                    if gate.lower() == 'rx':
                        qml.RX(params[param_idx], wires=qubit)
                        param_idx += 1
                    elif gate.lower() == 'ry':
                        qml.RY(params[param_idx], wires=qubit)
                        param_idx += 1
                    elif gate.lower() == 'rz':
                        qml.RZ(params[param_idx], wires=qubit)
                        param_idx += 1
                    elif gate.lower() == 'hadamard' or gate.lower() == 'h':
                        qml.Hadamard(wires=qubit)
            
            # Entanglement layer
            if entanglement == 'linear':
                for i in range(n_qubits - 1):
                    if 'cz' in gate_set:
                        qml.CZ(wires=[i, i+1])
                    else:
                        qml.CNOT(wires=[i, i+1])
            elif entanglement == 'circular':
                for i in range(n_qubits):
                    if 'cz' in gate_set:
                        qml.CZ(wires=[i, (i+1) % n_qubits])
                    else:
                        qml.CNOT(wires=[i, (i+1) % n_qubits])
            elif entanglement == 'all_to_all':
                for i in range(n_qubits):
                    for j in range(i+1, n_qubits):
                        if 'cz' in gate_set:
                            qml.CZ(wires=[i, j])
                        else:
                            qml.CNOT(wires=[i, j])
        
        # Measurement strategy
        if measurement == 'z':
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        elif measurement == 'x':
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        elif measurement == 'y':
            return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
        elif measurement == 'mixed':
            observables = []
            for i in range(n_qubits):
                if i % 3 == 0:
                    observables.append(qml.expval(qml.PauliX(i)))
                elif i % 3 == 1:
                    observables.append(qml.expval(qml.PauliY(i)))
                else:
                    observables.append(qml.expval(qml.PauliZ(i)))
            return observables
        else:
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    
    return circuit


def create_noisy_circuit(base_circuit: Callable, 
                         noise_model: Dict[str, Any]) -> Callable:
    """
    Wrap a circuit with a noise model to simulate realistic quantum hardware.
    
    Args:
        base_circuit: Base circuit function to add noise to
        noise_model: Noise parameters specification
        
    Returns:
        Noisy circuit function
    """
    noise_type = noise_model.get('type', 'depolarizing')
    noise_probability = noise_model.get('probability', 0.01)
    
    def noisy_circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Apply the base circuit with added noise model.
        
        Args:
            params: Circuit parameters
            features: Optional input features for data encoding
        """
        # Run the base circuit
        base_result = base_circuit(params, features)
        
        # For simulation with PennyLane, we'll need to use a noisy device
        # or explicitly add noise channels to the circuit
        # This is a dummy implementation to show the concept
        if noise_type == 'depolarizing':
            # In a real implementation, you would use a device with depolarizing channel
            # or explicitly add noise gates to the circuit
            pass
        
        return base_result
    
    return noisy_circuit


def parameter_shift_circuit(circuit_fn: Callable, 
                            params: np.ndarray, 
                            features: Optional[np.ndarray] = None) -> Callable:
    """
    Prepare a circuit for parameter-shift gradient calculation.
    
    Args:
        circuit_fn: Base circuit function
        params: Circuit parameters
        features: Optional input features for data encoding
        
    Returns:
        Function to compute shifted parameter values
    """
    def shifted_circuit(param_idx: int, shift: float = np.pi/2):
        """
        Create a circuit with one parameter shifted.
        
        Args:
            param_idx: Index of parameter to shift
            shift: Shift amount (default: π/2 for parameter-shift rule)
            
        Returns:
            Output of the circuit with shifted parameter
        """
        shifted_params = params.copy()
        shifted_params[param_idx] += shift
        
        return circuit_fn(shifted_params, features)
    
    return shifted_circuit