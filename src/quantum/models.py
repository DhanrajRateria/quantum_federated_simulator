"""
Quantum machine learning models including Variational Quantum Classifiers (VQC)
and Quantum Neural Networks (QNN) integrated with PyTorch.
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
import logging

# Conditional Qiskit imports
try:
    import qiskit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# Imports from local modules
from .utils import load_config
from .circuits import create_basic_circuit, create_complex_circuit, create_custom_circuit

logger = logging.getLogger(__name__)

class VariationalQuantumClassifier(nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_layers: int = 2,
                 n_classes: int = 2,
                 circuit_type: str = 'basic',
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None,
                 config_path: Optional[str] = None,
                 diff_method: str = 'best'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.circuit_type = circuit_type
        self.device_name = device_name
        self.shots = shots
        self.config_path = config_path

        logger.info(f"Initializing VQC (TorchLayer): qubits={n_qubits}, layers={n_layers}, "
                    f"circuit='{circuit_type}', device='{device_name}', shots={shots}, diff_method='{diff_method}'")

        self.q_device = qml.device(device_name, wires=n_qubits, shots=shots)



        # Circuit definition function selection
        if circuit_type == 'basic':
            self._circuit_definition_func: Callable = create_basic_circuit(n_qubits, n_layers)
        elif circuit_type == 'complex':
            self._circuit_definition_func: Callable = create_complex_circuit(n_qubits, n_layers)
        elif circuit_type == 'custom':
            if config_path is None:
                raise ValueError("config_path must be provided for custom circuit type")
            self._circuit_definition_func: Callable = create_custom_circuit(config_path)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        if not hasattr(self._circuit_definition_func, 'expected_params'):
            raise AttributeError(f"Circuit definition function for VQC (type '{circuit_type}') must have 'expected_params' attribute.")
        self.n_params = self._circuit_definition_func.expected_params
        logger.info(f"VQC circuit requires {self.n_params} parameters.")

        # CRITICAL: The key in weight_shapes MUST match the QNode's trainable parameter name
        weight_shapes = {"circuit_weights": self.n_params} # Parameter name in QNode signature

        actual_diff_method = self.determine_diff_method(device_name, shots, diff_method)

        logger.info(f"{self.__class__.__name__} attempting to use differentiation method: {actual_diff_method}")


        # This is the QNode function that TorchLayer will directly wrap for SINGLE inputs
        @qml.qnode(self.q_device, interface='torch', diff_method=actual_diff_method)
        def qnode_for_layer(inputs, circuit_weights):
            return self._circuit_definition_func(params=circuit_weights, features=inputs)

        self.q_layer = qml.qnn.TorchLayer(qnode_for_layer, weight_shapes)

        # Dummy pass for output_dim inference
        try:
            dummy_input_dim = n_qubits # Default for angle encoding based circuits
            if self.circuit_type == 'custom' and self.config_path:
                custom_cfg = load_config(self.config_path)
                encoding_type = custom_cfg.get('encoding', 'angle')
                if encoding_type == 'amplitude':
                    dummy_input_dim = 2**n_qubits
                # Add other encoding types if they change input dim requirements
            
            dummy_input_len = max(1, dummy_input_dim) # Ensure at least 1 feature for dummy input
            # Ensure features are correctly shaped based on encoding in circuits.py
            # For 'basic' and 'complex', it's angle encoding so num_qubits features.
            # If a circuit expects fewer features than n_qubits, adjust dummy_input_len.
            # Assuming circuit always can take up to n_qubits features for encoding.
            if hasattr(self._circuit_definition_func, 'expected_features'):
                 dummy_input_len = self._circuit_definition_func.expected_features
            else: # Fallback to n_qubits for angle encoding
                 dummy_input_len = min(n_qubits, dummy_input_len) if dummy_input_len > n_qubits else dummy_input_len


            dummy_input = torch.zeros(dummy_input_len, dtype=torch.float64)
            dummy_q_output = self.q_layer(dummy_input)

            if isinstance(dummy_q_output, torch.Tensor):
                self.circuit_output_dim = dummy_q_output.shape[-1] if dummy_q_output.dim() > 0 else 1
            else: # scalar output
                self.circuit_output_dim = 1
            logger.info(f"VQC raw circuit output dimension: {self.circuit_output_dim}")

            # Add a classical head to map circuit_output_dim to n_classes logits
            if self.n_classes is not None and self.n_classes > 0:
                if self.circuit_output_dim == 1 and self.n_classes == 2: # Single logit for BCEWithLogitsLoss
                    self.final_head = nn.Identity()
                    self.model_output_dim = 1
                    logger.info("VQC: Circuit outputs 1 value, n_classes=2. Using as single logit for BCEWithLogitsLoss.")
                else: # General case: map circuit_output_dim to n_classes logits
                    self.final_head = nn.Linear(self.circuit_output_dim, self.n_classes)
                    self.model_output_dim = self.n_classes
                    logger.info(f"VQC: Added Linear head: {self.circuit_output_dim} -> {self.model_output_dim} logits for CrossEntropyLoss.")
            else:
                self.final_head = nn.Identity()
                self.model_output_dim = self.circuit_output_dim
                logger.warning(f"VQC: n_classes not suitable ({self.n_classes}). Model will output raw circuit values. Output dim: {self.model_output_dim}")
        except Exception as e:
            logger.error(f"Failed to initialize or infer output dim for VQC TorchLayer: {e}", exc_info=True)
            raise

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of VQC. Handles batched input and dtype conversions properly.
        
        Args:
            features: Input tensor, either [batch_size, n_features] or [n_features]
                
        Returns:
            Model output tensor on the same device as features with correct dtype
        """
        input_device = features.device
        
        # Get the dtype of the final_head weights to match our output to it
        if hasattr(self.final_head, 'weight'):
            head_dtype = self.final_head.weight.dtype
        else:
            head_dtype = torch.float32  # Default for Identity or when no weights
        
        # Handle batched input (multiple examples)
        if features.dim() > 1:
            batch_size = features.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Process one example at a time with float64 for quantum circuit
                single_output = self.q_layer(features[i].to(torch.float64))
                
                # Ensure output is properly shaped for stacking
                if not isinstance(single_output, torch.Tensor):
                    single_output = torch.tensor([single_output], dtype=torch.float64)
                elif single_output.dim() == 0:  # Scalar tensor
                    single_output = single_output.unsqueeze(0)
                
                outputs.append(single_output)
            
            # Stack all processed examples into a batch and convert to head's dtype
            quantum_output = torch.stack(outputs).to(head_dtype)
        else:
            # Single example (non-batched input)
            quantum_output = self.q_layer(features.to(torch.float64))
            
            # Ensure output is a proper tensor with right dimensions
            if not isinstance(quantum_output, torch.Tensor):
                quantum_output = torch.tensor([quantum_output], dtype=torch.float64)
            elif quantum_output.dim() == 0:
                quantum_output = quantum_output.unsqueeze(0)
                
            # Convert to head's dtype
            quantum_output = quantum_output.to(head_dtype)
        
        # Move to the correct device
        quantum_output = quantum_output.to(input_device)
        
        # Apply the final head
        output = self.final_head(quantum_output)
        return output

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            raw_output = self.forward(features)
            
            # Ensure batch dimension exists
            if raw_output.dim() == 1:
                raw_output = raw_output.unsqueeze(0)
                
        if self.output_dim == 1 and self.n_classes == 2:
            probs = torch.sigmoid(raw_output)
            return torch.cat((1 - probs, probs), dim=-1)
        elif self.output_dim == self.n_classes:
            return torch.softmax(raw_output, dim=-1)
        else:
            logger.warning(f"VQC predict_proba: output_dim ({self.output_dim}) != n_classes ({self.n_classes}). Adapting output.")
            if self.output_dim > self.n_classes:
                return torch.softmax(raw_output[..., :self.n_classes], dim=-1)
            else:
                padded_output = torch.zeros(*raw_output.shape[:-1], self.n_classes, device=raw_output.device, dtype=raw_output.dtype)
                padded_output[..., :self.output_dim] = raw_output
                return torch.softmax(padded_output, dim=-1)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        probas = self.predict_proba(features)
        return torch.argmax(probas, dim=-1)
    
    def determine_diff_method(self, device_name, shots, diff_method):
        """Helper method to safely determine differentiation method."""
        if shots is not None:
            logger.info(f"{self.__class__.__name__}: Finite shots ({shots}) detected. Forcing 'parameter-shift'.")
            return 'parameter-shift'
        
        if diff_method != 'best':
            # User explicitly requested a specific method - use it
            return diff_method
        
        # For 'best' option, use a safer approach with newer PennyLane API
        if device_name.startswith('lightning'):
            # For lightning devices, prefer adjoint when no shots
            actual_diff_method = 'adjoint'
            logger.info(f"{self.__class__.__name__}: Using '{device_name}' with no shots. Defaulting to 'adjoint'.")
        else:
            # For other devices, start with backprop then fall back to parameter-shift
            try:
                # This is a simple test - if it works, go with it, if not, use parameter-shift
                @qml.qnode(self.q_device, interface='torch', diff_method='backprop')
                def dummy_circuit(x):
                    return qml.expval(qml.PauliZ(0))
                
                # Just test if we can create it, no need to run it
                actual_diff_method = 'backprop'
                logger.info(f"{self.__class__.__name__}: Device '{device_name}' appears to support 'backprop'.")
            except Exception as e:
                logger.info(f"{self.__class__.__name__}: Device '{device_name}' backprop test failed: {str(e)}. Using 'parameter-shift'.")
                actual_diff_method = 'parameter-shift'
        
        return actual_diff_method


class QuantumNeuralNetwork(nn.Module):
    def __init__(self,
                 n_qubits: int,
                 n_layers: int = 2,
                 output_dim: Optional[int] = None,
                 circuit_type: str = 'basic',
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None,
                 config_path: Optional[str] = None,
                 diff_method: str = 'best'):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.device_name = device_name
        self.shots = shots
        self.config_path = config_path
        self.requested_output_dim_param = output_dim
        logger.info(f"Initializing QNN Layer: qubits={n_qubits}, layers={n_layers}, "
                    f"circuit='{circuit_type}', device='{device_name}', shots={shots}, diff_method='{diff_method}'")

        self.q_device = qml.device(device_name, wires=n_qubits, shots=shots)

        if circuit_type == 'basic':
            self._circuit_definition_func: Callable = create_basic_circuit(n_qubits, n_layers)
        elif circuit_type == 'complex':
            self._circuit_definition_func: Callable = create_complex_circuit(n_qubits, n_layers)
        elif circuit_type == 'custom':
            if config_path is None:
                raise ValueError("config_path must be provided for custom circuit type")
            self._circuit_definition_func: Callable = create_custom_circuit(config_path)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        if not hasattr(self._circuit_definition_func, 'expected_params'):
             raise AttributeError(f"Circuit definition function for QNN (type '{circuit_type}') must have 'expected_params' attribute.")
        self.n_params = self._circuit_definition_func.expected_params
        logger.info(f"QNN circuit requires {self.n_params} parameters.")

        weight_shapes = {"circuit_weights": self.n_params} # Key matches QNode param name

        actual_diff_method = self.determine_diff_method(device_name, shots, diff_method)

        logger.info(f"{self.__class__.__name__} attempting to use differentiation method: {actual_diff_method}")

        # QNode for processing SINGLE inputs (not batches)
        @qml.qnode(self.q_device, interface='torch', diff_method=actual_diff_method)
        def qnode_for_layer(inputs, circuit_weights):
            return self._circuit_definition_func(params=circuit_weights, features=inputs)

        self.q_layer = qml.qnn.TorchLayer(qnode_for_layer, weight_shapes)
        logger.debug(f"qml.qnn.TorchLayer created successfully for QNN.")

        # Infer actual circuit output dimension
        try:
            dummy_input_dim = n_qubits # Default for angle encoding based circuits
            if self.circuit_type == 'custom' and self.config_path:
                custom_cfg = load_config(self.config_path) # Ensure load_config is available
                encoding_type = custom_cfg.get('encoding', 'angle')
                if encoding_type == 'amplitude':
                    dummy_input_dim = 2**n_qubits
            
            dummy_input_len = max(1, dummy_input_dim)
            if hasattr(self._circuit_definition_func, 'expected_features'):
                 dummy_input_len = self._circuit_definition_func.expected_features
            else:
                 dummy_input_len = min(n_qubits, dummy_input_len) if dummy_input_len > n_qubits else dummy_input_len


            dummy_input = torch.zeros(dummy_input_len, dtype=torch.float64)
            dummy_q_output = self.q_layer(dummy_input)

            if isinstance(dummy_q_output, torch.Tensor):
                self.circuit_output_dim = dummy_q_output.shape[-1] if dummy_q_output.dim() > 0 else 1
            else: # scalar output
                self.circuit_output_dim = 1
            logger.info(f"QNN raw circuit output dimension: {self.circuit_output_dim}")

            # Add a classical head if user requested an output_dim different from circuit_output_dim
            if self.requested_output_dim_param is not None and self.requested_output_dim_param != self.circuit_output_dim:
                self.final_head = nn.Linear(self.circuit_output_dim, self.requested_output_dim_param)
                self.output_dim = self.requested_output_dim_param # This QNN module's final output dimension
                logger.info(f"QNN: Added Linear head: {self.circuit_output_dim} -> {self.output_dim}")
            else:
                self.final_head = nn.Identity()
                self.output_dim = self.circuit_output_dim # This QNN module's final output dimension
            logger.info(f"QNN module's final output dimension (after head): {self.output_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize TorchLayer or infer output dim for QNN: {e}", exc_info=True)
            raise

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass handling both batched and single inputs with proper dtype conversion.
        
        Args:
            features: Input tensor with shape [batch_size, n_features] or [n_features]
            
        Returns:
            Quantum circuit output tensor with appropriate batching and dtype
        """
        input_device = features.device
        
        # Get the dtype of the final_head weights to match our output to it
        if hasattr(self.final_head, 'weight'):
            head_dtype = self.final_head.weight.dtype
        else:
            head_dtype = torch.float32  # Default for Identity or when no weights
        
        # Handle batched input (multiple examples)
        if features.dim() > 1:
            batch_size = features.shape[0]
            outputs = []
            
            for i in range(batch_size):
                # Process one example at a time with float64 for quantum circuit
                single_output = self.q_layer(features[i].to(torch.float64))
                
                # Ensure output is properly shaped for stacking
                if not isinstance(single_output, torch.Tensor):
                    single_output = torch.tensor([single_output], dtype=torch.float64)
                elif single_output.dim() == 0:  # Scalar tensor
                    single_output = single_output.unsqueeze(0)
                
                outputs.append(single_output)
            
            # Stack all processed examples into a batch and convert to head's dtype
            quantum_output = torch.stack(outputs).to(head_dtype)
        else:
            # Single example (non-batched input)
            quantum_output = self.q_layer(features.to(torch.float64))
            
            # Ensure output is a proper tensor with right dimensions
            if not isinstance(quantum_output, torch.Tensor):
                quantum_output = torch.tensor([quantum_output], dtype=torch.float64)
            elif quantum_output.dim() == 0:
                quantum_output = quantum_output.unsqueeze(0)
                
            # Convert to head's dtype
            quantum_output = quantum_output.to(head_dtype)
        
        # Move to the correct device
        quantum_output = quantum_output.to(input_device)
        
        # Apply the final head
        output = self.final_head(quantum_output)
        return output
        
    def determine_diff_method(self, device_name, shots, diff_method):
        """Helper method to safely determine differentiation method."""
        if shots is not None:
            logger.info(f"{self.__class__.__name__}: Finite shots ({shots}) detected. Forcing 'parameter-shift'.")
            return 'parameter-shift'
        
        if diff_method != 'best':
            # User explicitly requested a specific method - use it
            return diff_method
        
        # For 'best' option, use a safer approach with newer PennyLane API
        if device_name.startswith('lightning'):
            # For lightning devices, prefer adjoint when no shots
            actual_diff_method = 'adjoint'
            logger.info(f"{self.__class__.__name__}: Using '{device_name}' with no shots. Defaulting to 'adjoint'.")
        else:
            # For other devices, start with backprop then fall back to parameter-shift
            try:
                # This is a simple test - if it works, go with it, if not, use parameter-shift
                @qml.qnode(self.q_device, interface='torch', diff_method='backprop')
                def dummy_circuit(x):
                    return qml.expval(qml.PauliZ(0))
                
                # Just test if we can create it, no need to run it
                actual_diff_method = 'backprop'
                logger.info(f"{self.__class__.__name__}: Device '{device_name}' appears to support 'backprop'.")
            except Exception as e:
                logger.info(f"{self.__class__.__name__}: Device '{device_name}' backprop test failed: {str(e)}. Using 'parameter-shift'.")
                actual_diff_method = 'parameter-shift'
        
        return actual_diff_method


# QiskitVQC implementation (unchanged)
if QISKIT_AVAILABLE:
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_aer import AerSimulator

    class QiskitVQC:
        """
        Variational Quantum Classifier implementation using Qiskit.
        Demonstrates interoperability concepts. Requires qiskit and qiskit-aer.
        """
        def __init__(self,
                     n_qubits: int,
                     n_layers: int = 2,
                     n_classes: int = 2,
                     feature_map_type: str = 'zz',
                     ansatz_type: str = 'real_amplitudes',
                     shots: int = 1024):
            """
            Initialize the Qiskit Variational Quantum Classifier.
            """
            self.n_qubits = n_qubits
            self.n_layers = n_layers
            self.n_classes = n_classes
            self.feature_map_type = feature_map_type
            self.ansatz_type = ansatz_type
            self.shots = shots

            logger.info(f"Initializing QiskitVQC: qubits={n_qubits}, layers={n_layers}, classes={n_classes}, "
                        f"feature_map='{feature_map_type}', ansatz='{ansatz_type}', shots={shots}")

            # Create feature map
            if feature_map_type.lower() == 'zz':
                self.feature_map = ZZFeatureMap(n_qubits, reps=min(n_layers, 2)) # ZZ reps often <= 2
            else:
                logger.warning(f"Unsupported Qiskit feature map type '{feature_map_type}'. Defaulting to ZZFeatureMap.")
                self.feature_map = ZZFeatureMap(n_qubits, reps=min(n_layers, 2))

            # Create variational form (ansatz)
            if ansatz_type.lower() == 'real_amplitudes':
                self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)
            else:
                logger.warning(f"Unsupported Qiskit ansatz type '{ansatz_type}'. Defaulting to RealAmplitudes.")
                self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)

            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.circuit.compose(self.feature_map, inplace=True)
            self.circuit.compose(self.ansatz, inplace=True)
            logger.debug("Qiskit circuit composed (FeatureMap + Ansatz).")

            self.n_params = self.ansatz.num_parameters
            self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
            logger.info(f"QiskitVQC initialized with {self.n_params} parameters.")

            try:
                 self.simulator = AerSimulator()
                 logger.debug("Qiskit AerSimulator initialized.")
            except Exception as e:
                 logger.error(f"Failed to initialize Qiskit AerSimulator: {e}", exc_info=True)
                 raise RuntimeError("Could not initialize AerSimulator.") from e

        def run_circuit(self, features: np.ndarray) -> np.ndarray:
            if len(features) != self.feature_map.num_parameters:
                 raise ValueError(f"Feature length ({len(features)}) must match feature map params ({self.feature_map.num_parameters}).")

            param_binding = {p: v for p, v in zip(self.ansatz.parameters, self.params)}
            feature_binding = {p: v for p, v in zip(self.feature_map.parameters, features)}
            try:
                bound_circuit = self.circuit.assign_parameters({**feature_binding, **param_binding})
            except Exception as e:
                 logger.error(f"Error binding parameters: {e}", exc_info=True)
                 logger.error(f"Feature Map Params ({len(self.feature_map.parameters)}): {self.feature_map.parameters}")
                 logger.error(f"Ansatz Params ({len(self.ansatz.parameters)}): {self.ansatz.parameters}")
                 logger.error(f"Feature values ({len(features)}): {features}")
                 logger.error(f"Param values ({len(self.params)}): {self.params}")
                 raise

            measured_circuit = bound_circuit.copy(name="meas_circ")
            measured_circuit.measure_all(inplace=True)

            try:
                transpiled_circuit = qiskit.transpile(measured_circuit, self.simulator)
                job = self.simulator.run(transpiled_circuit, shots=self.shots)
                result = job.result()
                counts = result.get_counts(0)
            except Exception as e:
                 logger.error(f"Error during Qiskit circuit execution: {e}", exc_info=True)
                 raise

            probabilities = np.zeros(self.n_classes)
            total_shots_val = sum(counts.values()) # Renamed to avoid conflict
            if total_shots_val == 0: return probabilities

            for bitstring, count in counts.items():
                try:
                    class_index = int(bitstring, 2) % self.n_classes
                    probabilities[class_index] += count / total_shots_val
                except ValueError:
                     logger.warning(f"Could not interpret bitstring '{bitstring}' as integer.")
                except IndexError:
                     logger.warning(f"Class index from '{bitstring}' out of bounds.")
            return probabilities

        def predict(self, features: np.ndarray) -> int:
            probabilities = self.run_circuit(features)
            return int(np.argmax(probabilities))

        def convert_to_pennylane(self) -> Optional[VariationalQuantumClassifier]:
            logger.info("Attempting conceptual conversion from QiskitVQC to PennyLane VQC...")
            try:
                pennylane_model = VariationalQuantumClassifier(
                    n_qubits=self.n_qubits, n_layers=self.n_layers, n_classes=self.n_classes,
                    circuit_type='basic', device_name='default.qubit', shots=self.shots
                )
                logger.info("Conceptual conversion successful (created new PennyLane VQC).")
                return pennylane_model
            except Exception as e:
                logger.error(f"Failed conceptual conversion: {e}", exc_info=True)
                return None
else:
    class QiskitVQC:
        def __init__(self, *args, **kwargs):
            logger.error("QiskitVQC class requires Qiskit installation.")
            raise ImportError("QiskitVQC requires qiskit and qiskit-aer.")


class HybridQuantumModel(nn.Module):
    def __init__(self,
                 n_qubits: int,
                 classical_input_dim: int,
                 classical_hidden_dim: int,
                 classical_output_dim: int, # This is n_classes for classification
                 q_layers: int = 2,
                 q_circuit_type: str = 'basic',
                 q_device_name: str = 'default.qubit',
                 q_shots: Optional[int] = None,
                 q_config_path: Optional[str] = None,
                 q_diff_method: str = 'best'): # Added diff_method for QNN
        super().__init__()
        self.n_qubits = n_qubits
        self.classical_input_dim = classical_input_dim
        self.classical_hidden_dim = classical_hidden_dim
        self.classical_output_dim = classical_output_dim # Final output (e.g. num_classes)

        logger.info(f"Initializing HybridQuantumModel: Input={classical_input_dim}, "
                    f"Hidden={classical_hidden_dim}, FinalOutput={classical_output_dim}, "
                    f"Qubits={n_qubits}, QCircuit='{q_circuit_type}', QDiff='{q_diff_method}'")

        # Classical pre-processing: maps input_dim to n_qubits (for angle/feature encoding)
        self.classical_pre = nn.Sequential(
            nn.Linear(classical_input_dim, classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim, n_qubits), # Output for quantum layer's feature input
            # Optional: nn.Tanh() if quantum encoding expects features in [-1, 1]
        )
        logger.debug(f"Classical Pre-processing layers initialized.")

        # Quantum layer
        self.quantum_layer = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=q_layers,
            # QNN's output_dim is inferred from its circuit. It doesn't need to match classical_output_dim here.
            output_dim=None, # Let QNN infer its circuit_output_dim
            circuit_type=q_circuit_type,
            device_name=q_device_name,
            shots=q_shots,
            config_path=q_config_path,
            diff_method=q_diff_method # Pass diff_method to QNN
        )
        # The actual number of features output by the quantum_layer
        self.quantum_feature_dim = self.quantum_layer.output_dim # QNN.output_dim is its circuit_output_dim
        logger.debug(f"Quantum layer initialized. Output features from quantum part: {self.quantum_feature_dim}")

        # Classical post-processing: maps quantum_feature_dim to final classical_output_dim
        self.classical_post = nn.Sequential(
            nn.Linear(self.quantum_feature_dim, classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim, classical_output_dim)
            # Optional: nn.LogSoftmax(dim=-1) if using NLLLoss for classification
        )
        logger.debug(f"Classical Post-processing layers initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the hybrid model, handles batched inputs properly.
        Ensures consistent device placement throughout the forward pass.
        
        Args:
            x: Input tensor with shape [batch_size, n_features] or [n_features]
            
        Returns:
            Model output tensor on the same device as input
        """
        input_device = x.device
        
        # Classical models and data loaders usually use float32
        x_classical = x.to(torch.float32)
        
        # Classical pre-processing (this handles batches natively)
        x_preprocessed = self.classical_pre(x_classical)
        
        # Quantum layer (this now handles batches properly in its forward method)
        x_quantum_out = self.quantum_layer(x_preprocessed)
        
        # Cast back to float32 for subsequent classical layers and move to input device
        x_to_post = x_quantum_out.to(torch.float32).to(input_device)
        
        # Classical post-processing (this handles batches natively)
        x_final = self.classical_post(x_to_post)
        return x_final.to(input_device)  # Final output on input device