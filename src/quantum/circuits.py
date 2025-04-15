"""
Quantum circuit implementations for the Federated Quantum ML Simulator.
"""

import pennylane as qml
import numpy as np
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
from .utils import load_config, get_device, calculate_expected_params
import logging

logger = logging.getLogger(__name__)

def create_basic_circuit(n_qubits: int, n_layers: int = 1) -> Callable:
    """
    Create a basic parameterized quantum circuit with alternating rotations and entanglement.
    
    Args:
        n_qubits: Number of qubits in the circuit
        n_layers: Number of repeated layers
        
    Returns:
        Circuit function that accepts parameters
    """
    logger.info(f"Creating basic circuit with {n_qubits} qubits, {n_layers} layers.")
    # Calculate the number of trainable parameters required by this circuit structure
    expected_params_per_layer = n_qubits * 3 # RX, RY, RZ per qubit
    expected_total_params = expected_params_per_layer * n_layers
    logger.info(f"Basic circuit expects {expected_total_params} trainable parameters.")
    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Basic circuit implementation with rotation and entanglement layers.
        
        Args:
            params: Circuit parameters
            features: Optional input features for data encoding
        """

        if params.shape[0] != expected_total_params:
             logger.warning(f"Received params shape {params.shape} does not match expected shape ({expected_total_params},) for basic circuit.")

        # Data encoding block (if features are provided)
        if features is not None:
            for i in range(min(n_qubits, len(features))):
                qml.RX(features[i], wires=i)
        
        # Apply layers of parameterized gates
        for layer in range(n_layers):
            # Parameter offset for current layer
            offset = layer * expected_params_per_layer
            
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
    
    circuit.expected_params = expected_total_params
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
    logger.info(f"Creating complex circuit with {n_qubits} qubits, {n_layers} layers.")

    # Calculate expected trainable parameters
    rotation_params_per_qubit = 3  # RX, RY, RZ
    entanglement_params_per_pair = 2  # CRX, CRY

    pairs = []
    for i in range(n_qubits):
        pairs.append((i, (i + 1) % n_qubits)) # Ring
        if n_qubits > 3: # Long-range
            pairs.append((i, (i + n_qubits // 2) % n_qubits))
    # Remove duplicate pairs if any (e.g., (0, 2) and (2, 0) if n_qubits=4) - use set for uniqueness
    unique_pairs = sorted(list(set(tuple(sorted(p)) for p in pairs)))
    n_unique_pairs = len(unique_pairs)

    params_per_layer = (n_qubits * rotation_params_per_qubit) + (n_unique_pairs * entanglement_params_per_pair)
    expected_total_params = params_per_layer * n_layers
    logger.info(f"Complex circuit expects {expected_total_params} trainable parameters.")


    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Complex circuit implementation.

        Args:
            params: Trainable circuit parameters (shape: (expected_total_params,)).
            features: Optional input features for initial data encoding (shape: (<=n_qubits,)).
        """
        if params.shape[0] != expected_total_params:
             logger.warning(f"Received params shape {params.shape} does not match expected shape ({expected_total_params},) for complex circuit.")

        # Data encoding block
        if features is not None:
            logger.debug(f"Applying feature encoding (RX) with features shape {features.shape}")
            for i in range(min(n_qubits, len(features))):
                qml.RX(features[i], wires=i) # Not parameterized by 'params'

        # Ansatz Layers
        current_param_idx = 0
        for layer in range(n_layers):
            # Rotation block (using trainable params)
            for qubit in range(n_qubits):
                qml.RX(params[current_param_idx], wires=qubit); current_param_idx += 1
                qml.RY(params[current_param_idx], wires=qubit); current_param_idx += 1
                qml.RZ(params[current_param_idx], wires=qubit); current_param_idx += 1

            # Entanglement block (using trainable params)
            for control, target in unique_pairs:
                # Check if index exists before accessing
                if current_param_idx + 1 < len(params):
                    qml.CRX(params[current_param_idx], wires=[control, target]); current_param_idx += 1
                    qml.CRY(params[current_param_idx], wires=[control, target]); current_param_idx += 1
                else:
                     logger.error(f"Parameter index out of bounds during entanglement layer {layer+1}. Expected more parameters.")
                     # Handle error appropriately, e.g., raise IndexError

            # Non-parameterized gates for extra complexity (fixed)
            if layer < n_layers - 1:
                for i in range(n_qubits):
                    if i % 2 == 0:
                        qml.Hadamard(wires=i)

        # Measurement strategy (fixed)
        measurements = []
        for i in range(n_qubits):
            measurements.append(qml.expval(qml.PauliZ(i)))
        if n_qubits >= 4:
            measurements.append(qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)))
            measurements.append(qml.expval(qml.PauliX(0) @ qml.PauliX(2)))

        return measurements

    circuit.expected_params = expected_total_params
    return circuit


def create_custom_circuit(config_path: str) -> Callable:
    """
    Create a quantum circuit based on a YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A callable PennyLane circuit function built according to the config.
        Expects args: (params: np.ndarray, features: Optional[np.ndarray] = None).
    """
    logger.info(f"Creating custom circuit from config: {config_path}")
    config = load_config(config_path)
    n_qubits = config.get('n_qubits', 4)
    n_layers = config.get('n_layers', 2)
    gate_set = config.get('gate_set', ['rx', 'ry', 'rz', 'cz'])
    entanglement = config.get('entanglement', 'linear')
    measurement = config.get('measurement', 'z')
    encoding_type = config.get('encoding', 'angle') # Get encoding from config

    # Calculate expected parameters based *only* on parameterized gates in gate_set
    # This version assumes only single-qubit gates listed in gate_set are parameterized
    expected_total_params = calculate_expected_params(n_qubits, n_layers, gate_set, entanglement)
    config_n_params = config.get('n_params') # Get n_params from config if specified

    if config_n_params is not None and config_n_params != expected_total_params:
        logger.warning(f"Config file {config_path} specifies n_params={config_n_params}, "
                       f"but calculated expected parameters based on gate_set/layers/qubits is {expected_total_params}. "
                       f"Using calculated value {expected_total_params}.")
    elif config_n_params is None:
         logger.info(f"Custom circuit expects {expected_total_params} trainable parameters (calculated).")
    else:
         logger.info(f"Custom circuit expects {expected_total_params} trainable parameters (from config).")

    def circuit(params: np.ndarray, features: Optional[np.ndarray] = None):
        """
        Custom circuit implementation based on configuration.

        Args:
            params: Trainable circuit parameters (shape: (expected_total_params,)).
            features: Optional input features for data encoding.
        """
        if params.shape[0] != expected_total_params:
             logger.warning(f"Received params shape {params.shape} does not match expected shape ({expected_total_params},) for custom circuit from {config_path}.")

        # Data encoding block
        if features is not None:
            logger.debug(f"Applying feature encoding ({encoding_type}) with features shape {features.shape}")
            if encoding_type == 'angle':
                # Simple angle encoding using RX
                for i in range(min(n_qubits, len(features))):
                    qml.RX(features[i], wires=i)
            elif encoding_type == 'amplitude':
                # Amplitude encoding requires features length <= 2^n_qubits
                required_dim = 2**n_qubits
                if len(features) > required_dim:
                     logger.warning(f"Amplitude encoding requires len(features) <= 2**n_qubits ({required_dim}), got {len(features)}. Truncating features.")
                     features = features[:required_dim]

                # Pad if necessary
                if len(features) < required_dim:
                    padded_features = np.zeros(required_dim)
                    padded_features[:len(features)] = features
                    features = padded_features

                # Normalize for AmplitudeEmbedding (it can normalize itself, but explicit is clearer)
                norm = np.linalg.norm(features)
                if norm > 0:
                    normalized_features = features / norm
                else:
                    normalized_features = features # Avoid division by zero

                qml.AmplitudeEmbedding(normalized_features, wires=range(n_qubits), pad_with=0.0, normalize=False) # Already normalized
            # Add other encoding types from quantum_encodings.py here if needed
            else:
                 logger.warning(f"Unsupported encoding type '{encoding_type}' specified in config. Skipping encoding.")


        # Ansatz Layers
        param_idx = 0 # Index for the 'params' array
        for layer in range(n_layers):
            # Apply parameterized gates from gate_set
            for qubit in range(n_qubits):
                for gate in gate_set:
                    # Apply only parameterized gates using params
                    if gate.lower() == 'rx':
                        if param_idx < len(params): qml.RX(params[param_idx], wires=qubit); param_idx += 1
                    elif gate.lower() == 'ry':
                         if param_idx < len(params): qml.RY(params[param_idx], wires=qubit); param_idx += 1
                    elif gate.lower() == 'rz':
                         if param_idx < len(params): qml.RZ(params[param_idx], wires=qubit); param_idx += 1
                    # Apply non-parameterized gates directly
                    elif gate.lower() in ['h', 'hadamard']:
                        qml.Hadamard(wires=qubit)
                    elif gate.lower() == 'x':
                        qml.PauliX(wires=qubit)
                    # ... add other non-parameterized gates if needed ...

            # Entanglement layer (non-parameterized based on common gates like CNOT/CZ)
            entanglement_gates = [g.lower() for g in gate_set if g.lower() in ['cz', 'cnot']]
            ent_gate = qml.CZ if 'cz' in entanglement_gates else (qml.CNOT if 'cnot' in entanglement_gates else None)

            if ent_gate:
                if entanglement == 'linear':
                    for i in range(n_qubits - 1):
                        ent_gate(wires=[i, i+1])
                elif entanglement == 'circular':
                    for i in range(n_qubits):
                        ent_gate(wires=[i, (i+1) % n_qubits])
                elif entanglement == 'all_to_all':
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            ent_gate(wires=[i, j])
                else:
                     logger.warning(f"Unsupported entanglement type '{entanglement}' in config. Skipping entanglement.")
            else:
                 logger.debug("No entangling gate (CZ or CNOT) found in gate_set. Skipping entanglement layer.")

        # Final parameter index check
        if param_idx != expected_total_params:
            logger.warning(f"Parameter usage mismatch in custom circuit. Used {param_idx} params, expected {expected_total_params}.")

        # Measurement strategy
        logger.debug(f"Applying measurement strategy: {measurement}")
        if measurement == 'z':
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        elif measurement == 'x':
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
        elif measurement == 'y':
            return [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
        elif measurement == 'mixed':
            observables = []
            for i in range(n_qubits):
                if i % 3 == 0: observables.append(qml.expval(qml.PauliX(i)))
                elif i % 3 == 1: observables.append(qml.expval(qml.PauliY(i)))
                else: observables.append(qml.expval(qml.PauliZ(i)))
            return observables
        elif measurement == 'prob':
             # Return probability distribution over basis states
             return qml.probs(wires=range(n_qubits))
        else:
            logger.warning(f"Unsupported measurement type '{measurement}'. Defaulting to PauliZ expectation.")
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    circuit.expected_params = expected_total_params
    return circuit


def create_noisy_circuit(base_circuit: Callable,
                         noise_model: Dict[str, Any]) -> Callable:
    """
    Conceptual wrapper for adding noise.

    NOTE: This is a placeholder. True noise simulation in PennyLane typically
    requires using a noisy device (e.g., 'qiskit.aer' with a noise model)
    or manually inserting qml.NoiseChannel operations within the base_circuit
    definition itself. This wrapper doesn't modify the underlying circuit execution
    in standard PennyLane simulators like 'default.qubit'.

    Args:
        base_circuit: Base circuit function to wrap.
        noise_model: Dictionary specifying noise parameters (e.g., {'type': 'depolarizing', 'probability': 0.01}).

    Returns:
        The original circuit function (as this wrapper is conceptual).
    """
    noise_type = noise_model.get('type', 'depolarizing')
    noise_probability = noise_model.get('probability', 0.01)
    logger.warning(f"create_noisy_circuit is currently conceptual. It wraps the base circuit but doesn't actively apply noise type '{noise_type}' with probability {noise_probability} using standard simulators. Use a noisy device or modify the base circuit definition for actual noise simulation.")

    # This wrapper doesn't change the function's behavior with standard simulators
    # It just serves as a placeholder in the architecture design.
    def noisy_circuit_wrapper(params: np.ndarray, features: Optional[np.ndarray] = None):
        # In a real implementation with a noisy device, the noise would be applied
        # automatically by the device during execution.
        # Or, if manually adding channels, that logic would be within the base_circuit.
        return base_circuit(params, features)

    # Pass along expected params if the base circuit has it
    if hasattr(base_circuit, 'expected_params'):
         noisy_circuit_wrapper.expected_params = base_circuit.expected_params

    return noisy_circuit_wrapper


# --- parameter_shift_circuit ---
# (No changes needed, seems correct for its purpose)
def parameter_shift_circuit(circuit_fn: Callable,
                            params: np.ndarray,
                            features: Optional[np.ndarray] = None) -> Callable:
    """
    Prepare a circuit for parameter-shift gradient calculation.

    Args:
        circuit_fn: Base circuit function (QNode compatible).
        params: The parameter vector at which to calculate the shift.
        features: Optional input features (fixed during gradient calculation).

    Returns:
        A function that takes a parameter index and shift amount,
        and returns the output of the circuit with that parameter shifted.
    """
    logger.debug(f"Creating parameter shift function for circuit with {len(params)} params.")

    def shifted_circuit(param_idx: int, shift: float = np.pi/2):
        """
        Executes the circuit with one parameter shifted.

        Args:
            param_idx: Index of the parameter to shift.
            shift: Amount to shift the parameter by (default: pi/2).

        Returns:
            Output of the circuit with the shifted parameter.
        """
        if not 0 <= param_idx < len(params):
            raise ValueError(f"param_idx {param_idx} is out of bounds for params length {len(params)}")

        # Copy params to avoid modifying the original array
        shifted_params = params.copy()
        shifted_params[param_idx] += shift
        logger.debug(f"Executing shifted circuit: param index {param_idx}, shift {shift}")
        return circuit_fn(shifted_params, features)

    return shifted_circuit