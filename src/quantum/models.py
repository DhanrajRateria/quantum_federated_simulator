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
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    logging.warning("Qiskit or Qiskit Aer not found. QiskitVQC class will not be fully functional.")

# Imports from local modules
from .utils import get_device, load_config
from .circuits import create_basic_circuit, create_complex_circuit, create_custom_circuit
# Encodings are generally handled *within* the circuit functions now
# from .encodings import angle_encoding, amplitude_encoding, basis_encoding, iqp_feature_map

logger = logging.getLogger(__name__)

class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier implemented with PennyLane.
    Uses manual parameter-shift rule for training demonstration.
    """

    def __init__(self,
                 n_qubits: int,
                 n_layers: int = 2,
                 n_classes: int = 2,
                 circuit_type: str = 'basic',
                 # encoding_type: str = 'angle', # Encoding is now implicitly handled by circuit_fn
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Variational Quantum Classifier.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of circuit layers in the ansatz.
            n_classes: Number of output classes.
            circuit_type: Type of circuit ('basic', 'complex', 'custom').
                         The chosen circuit function should handle data encoding internally.
            device_name: PennyLane device name.
            shots: Number of measurement shots (None for analytic).
            config_path: Path to custom configuration file (required for 'custom' circuit_type).
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.circuit_type = circuit_type
        # self.encoding_type = encoding_type # Removed, handled by circuit_fn
        self.device_name = device_name
        self.shots = shots
        self.config_path = config_path
        self.params = None # Initialize later

        logger.info(f"Initializing VQC: qubits={n_qubits}, layers={n_layers}, classes={n_classes}, "
                    f"circuit='{circuit_type}', device='{device_name}', shots={shots}")

        # Initialize the device
        self.device = get_device(device_name, wires=n_qubits, shots=shots)

        # Get the right circuit function based on type
        if circuit_type == 'basic':
            self.circuit_fn = create_basic_circuit(n_qubits, n_layers)
        elif circuit_type == 'complex':
            self.circuit_fn = create_complex_circuit(n_qubits, n_layers)
        elif circuit_type == 'custom':
            if config_path is None:
                msg = "config_path must be provided for custom circuit type"
                logger.error(msg)
                raise ValueError(msg)
            # Need to load config to get circuit details if not passed explicitly
            try:
                # Custom circuit creator might need more info from config
                self.circuit_fn = create_custom_circuit(config_path)
            except Exception as e:
                logger.error(f"Failed to create custom circuit from {config_path}: {e}", exc_info=True)
                raise
        else:
            msg = f"Unknown circuit type: {circuit_type}"
            logger.error(msg)
            raise ValueError(msg)

        # Determine the number of parameters required by the circuit function
        if hasattr(self.circuit_fn, 'expected_params'):
            self.n_params = self.circuit_fn.expected_params
            logger.info(f"Circuit function requires {self.n_params} parameters.")
        else:
            # Fallback or error if expected_params is missing
            # This fallback is less reliable - should ideally always exist
            default_params_guess = n_qubits * 3 * n_layers # Simple guess
            self.n_params = default_params_guess
            logger.warning(f"Circuit function {circuit_type} missing 'expected_params' attribute. "
                           f"Guessing {self.n_params} parameters. Parameter checks may fail.")

        # Initialize parameters randomly
        # Use a small range initially, often helps training VQCs
        self.params = np.random.uniform(0, np.pi * 0.1, size=self.n_params) # Smaller initial range
        logger.info(f"Initialized {self.n_params} parameters randomly.")

        # Create the quantum node function
        # Wrap the circuit function; QNode handles features argument correctly
        @qml.qnode(self.device)
        def qnode_wrapper(params, features):
             # The circuit function itself handles params and features
             # Assumes circuit_fn measures expectation values needed for predict/loss
             return self.circuit_fn(params=params, features=features)

        self.qnode = qnode_wrapper

        self.circuit_output_dim = n_qubits
        if self.circuit_output_dim < n_classes:
             logger.warning(f"VQC WARNING: Circuit output dimension ({self.circuit_output_dim}) "
                            f"is less than target n_classes ({n_classes}). "
                            "Prediction/Loss may be suboptimal.")
             
        logger.debug("PennyLane QNode created.")

    # Removed get_encoding_function - encoding is internal to circuit_fn

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum circuit for a single sample.

        Args:
            features: Input data sample (1D array).

        Returns:
            Raw output from the quantum circuit (e.g., expectation values).
        """
        if self.params is None:
            raise RuntimeError("Model parameters have not been initialized.")
        if features.ndim != 1:
            logger.warning(f"VQC.forward expected 1D feature array, got {features.ndim}D. Attempting to proceed.")
            # Depending on the circuit's encoding, might need reshape or error
        try:
            # Ensure features are compatible type (e.g., float) if needed by device/gates
            features = np.asarray(features, dtype=float)
            result = self.qnode(self.params, features=features)
            # Convert to NumPy array for consistency
            return np.asarray(result)
        except Exception as e:
            logger.error(f"Error during QNode execution in forward pass: {e}", exc_info=True)
            raise

    def predict(self, features: np.ndarray) -> int:
        """Predict class based on circuit output."""
        output = self.forward(features) # Shape: (circuit_output_dim,)
        logger.debug(f"Raw output for prediction: {output}")

        # Use only the available outputs for prediction
        num_outputs_to_use = min(self.n_classes, self.circuit_output_dim)
        class_values = output[:num_outputs_to_use]

        if not np.any(class_values): # Handle all zeros case
             prediction = np.random.randint(0, self.n_classes) # Random guess
             logger.debug(f"Prediction from zero outputs -> random guess: {prediction}")
        elif num_outputs_to_use == 1 and self.n_classes == 2: # Specific binary case
            prediction = 1 if class_values[0] > 0 else 0
            logger.debug(f"Binary prediction: decision_value={class_values[0]:.4f} -> class={prediction}")
        else: # General case: argmax over available outputs
            prediction = np.argmax(class_values)
            # This will only predict classes 0 to circuit_output_dim-1
            logger.debug(f"Multi-class prediction (limited by output dim): values={class_values} -> class={prediction}")

        return int(prediction)

    def _compute_loss_single_sample(self, prediction_output: np.ndarray, label: int) -> float:
         """Compute loss, adapting to output dimension."""
         # Use only available outputs, scale to [0,1] approx probabilities
         num_outputs_to_use = min(self.n_classes, self.circuit_output_dim)
         pred_values = prediction_output[:num_outputs_to_use]
         pred_probs_approx = (pred_values + 1.0) / 2.0

         # Create target vector based on ACTUAL classes
         y_one_hot = np.zeros(self.n_classes)
         if 0 <= label < self.n_classes: y_one_hot[label] = 1.0
         else: logger.warning(f"Invalid label {label} for n_classes={self.n_classes}.")

         # Compare available predicted probabilities to corresponding target entries
         loss = np.mean((pred_probs_approx - y_one_hot[:num_outputs_to_use]) ** 2)
         # Add penalty for mismatch in dimensions? Optional.
         # For classes > circuit_output_dim, the target is 1 but prediction is implicitly 0.
         # Could add: loss += np.sum(y_one_hot[num_outputs_to_use:]) # Penalize missed classes

         return loss

    def loss(self, features_batch: np.ndarray, labels_batch: np.ndarray, current_params: Optional[np.ndarray] = None) -> float:
        """
        Compute the average loss for a batch of data, optionally using specific parameters.

        Args:
            features_batch: Input data batch (2D array, shape: [batch_size, n_features]).
            labels_batch: Target labels (1D array, shape: [batch_size]).
            current_params: Optional NumPy array of parameters to use for calculating
                             the loss. If None, uses self.params.

        Returns:
            Average loss value over the batch.
        """
        if features_batch.ndim != 2 or labels_batch.ndim != 1 or features_batch.shape[0] != labels_batch.shape[0]:
             raise ValueError("Invalid shapes for features_batch or labels_batch.")
        batch_size = features_batch.shape[0]; total_loss = 0.0
        if batch_size == 0: return 0.0

        # Use provided params if given, else use model's current params
        params_to_use = current_params if current_params is not None else self.params
        original_params = self.params # Store original to restore later if needed

        # Temporarily set model params ONLY if different params were provided
        if current_params is not None:
            self.params = params_to_use

        try:
             # Calculate loss using params_to_use (which might be self.params or current_params)
             for i in range(batch_size):
                 # The forward pass uses self.params, which we just set
                 prediction_output = self.forward(features_batch[i])
                 total_loss += self._compute_loss_single_sample(prediction_output, labels_batch[i])
        finally:
             # IMPORTANT: Restore original parameters if they were temporarily changed
             if current_params is not None:
                  self.params = original_params

        average_loss = total_loss / batch_size
        logger.debug(f"Calculated batch loss: {average_loss:.6f}") # Keep less verbose
        return average_loss

    def train_step(self,
                   features_batch: np.ndarray,
                   labels_batch: np.ndarray,
                   learning_rate: float = 0.1,
                   proximal_term: float = 0.0,
                   global_params_numpy: Optional[np.ndarray] = None
                   ) -> float:
        """
        Perform one training step using the parameter-shift rule,
        optionally including the FedProx proximal term.

        Args:
            features_batch: Input data batch.
            labels_batch: Target labels batch.
            learning_rate: Learning rate for gradient descent.
            proximal_term: Coefficient mu for the FedProx proximal term. If > 0,
                           global_params_numpy must be provided.
            global_params_numpy: NumPy array of the global model parameters
                                 required for FedProx calculation.

        Returns:
            Loss value *before* the parameter update.
        """
        if proximal_term > 0 and global_params_numpy is None:
            raise ValueError("global_params_numpy must be provided when proximal_term > 0 for FedProx.")
        if proximal_term > 0 and global_params_numpy.shape != self.params.shape:
             raise ValueError("Shape mismatch between local and global parameters for FedProx.")

        # Store original params and calculate initial loss
        original_params = np.copy(self.params)
        # Loss before update (without prox term for reporting)
        initial_loss = self.loss(features_batch, labels_batch, current_params=original_params)

        # Compute gradients using parameter-shift rule manually
        gradients = np.zeros_like(self.params)
        param_shift = np.pi / 2

        for i in range(len(self.params)):
            # Define loss function including potential FedProx term *for gradient calculation*
            def loss_for_grad(p):
                 base_loss = self.loss(features_batch, labels_batch, current_params=p)
                 if proximal_term > 0:
                      prox_loss = (proximal_term / 2.0) * np.sum((p - global_params_numpy) ** 2)
                      return base_loss + prox_loss
                 return base_loss

            # Shift up
            params_plus = original_params.copy(); params_plus[i] += param_shift
            loss_plus = loss_for_grad(params_plus)

            # Shift down
            params_minus = original_params.copy(); params_minus[i] -= param_shift
            loss_minus = loss_for_grad(params_minus)

            # Calculate gradient for parameter i using central difference formula
            gradients[i] = (loss_plus - loss_minus) / (2.0 * np.sin(param_shift))

        # Update parameters using gradients (apply update to original_params)
        self.params = original_params - learning_rate * gradients
        # self.params = np.mod(self.params, 2 * np.pi) # Optional clipping/wrapping

        logger.debug(f"VQC Train step completed. Initial Loss: {initial_loss:.6f}. Gradient norm: {np.linalg.norm(gradients):.4f}")
        return initial_loss


    def fit(self,
            features: np.ndarray,
            labels: np.ndarray,
            epochs: int = 100,
            batch_size: int = 10,
            learning_rate: float = 0.1,
            verbose: bool = True) -> List[float]:
        """
        Train the model on the given dataset using manual gradient descent.

        Args:
            features: Training data (2D array).
            labels: Target labels (1D array).
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            learning_rate: Learning rate for optimization.
            verbose: Whether to print progress.

        Returns:
            List of average loss values per epoch.
        """
        n_samples = features.shape[0]
        if n_samples == 0:
            logger.warning("Training dataset is empty. Skipping fit.")
            return []

        loss_history = []
        logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")

        for epoch in range(epochs):
            # Shuffle data indices
            shuffle_indices = np.random.permutation(n_samples)

            epoch_total_loss = 0.0
            num_batches = 0

            # Process batches
            for i in range(0, n_samples, batch_size):
                batch_indices = shuffle_indices[i:min(i + batch_size, n_samples)]
                if len(batch_indices) == 0: continue # Skip empty trailing batch

                batch_features = features[batch_indices]
                batch_labels = labels[batch_indices]

                # Perform one training step
                batch_loss = self.train_step(batch_features, batch_labels, learning_rate)
                epoch_total_loss += batch_loss * len(batch_indices) # Weight loss by actual batch size
                num_batches += 1

            # Calculate average loss for the epoch
            if num_batches > 0 :
                 avg_epoch_loss = epoch_total_loss / n_samples
                 loss_history.append(avg_epoch_loss)
                 if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                      print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.6f}")
            else:
                 logger.warning(f"Epoch {epoch+1} had no batches.")
                 loss_history.append(np.nan) # Indicate missing loss

        logger.info("Training finished.")
        return loss_history

    def save_params(self, filename: str) -> None:
        """Save the trained parameters to a file."""
        try:
            np.save(filename, self.params)
            logger.info(f"Parameters saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save parameters to {filename}: {e}", exc_info=True)
            raise

    def load_params(self, filename: str) -> None:
        """Load parameters from a file."""
        try:
            loaded_params = np.load(filename)
            if loaded_params.shape == self.params.shape:
                self.params = loaded_params
                logger.info(f"Parameters loaded from {filename}")
            else:
                logger.error(f"Shape mismatch loading parameters from {filename}. "
                             f"Expected {self.params.shape}, got {loaded_params.shape}.")
                raise ValueError("Parameter shape mismatch during loading.")
        except FileNotFoundError:
            logger.error(f"Parameter file not found: {filename}")
            raise
        except Exception as e:
            logger.error(f"Failed to load parameters from {filename}: {e}", exc_info=True)
            raise


class QuantumNeuralNetwork(nn.Module):
    """
    Quantum Neural Network layer implemented as a PyTorch nn.Module.
    Uses qml.qnn.TorchLayer for seamless integration with PyTorch autograd.
    """

    def __init__(self,
                 n_qubits: int,
                 n_layers: int = 2,
                 output_dim: int = None, # Output dim determined by circuit measurements
                 circuit_type: str = 'basic',
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Quantum Neural Network layer.

        Args:
            n_qubits: Number of qubits.
            n_layers: Number of circuit layers in the ansatz.
            output_dim: Expected dimension of the quantum layer's output. If None,
                        it's inferred from the circuit's measurement settings.
            circuit_type: Type of circuit ('basic', 'complex', 'custom').
                          The circuit function should handle data encoding internally based on features.
            device_name: PennyLane device name.
            shots: Number of measurement shots (None for analytic). Affects gradients if not None.
            config_path: Path to custom configuration file (required for 'custom' circuit_type).
        """
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        self.device_name = device_name
        self.shots = shots
        self.config_path = config_path

        logger.info(f"Initializing QNN Layer: qubits={n_qubits}, layers={n_layers}, "
                    f"circuit='{circuit_type}', device='{device_name}', shots={shots}")

        # Initialize the device (shared or specific instance)
        self.q_device = get_device(device_name, wires=n_qubits, shots=shots)

        # Get the right circuit function
        if circuit_type == 'basic':
            self.circuit_fn = create_basic_circuit(n_qubits, n_layers)
        elif circuit_type == 'complex':
            self.circuit_fn = create_complex_circuit(n_qubits, n_layers)
        elif circuit_type == 'custom':
            if config_path is None:
                raise ValueError("config_path must be provided for custom circuit type")
            self.circuit_fn = create_custom_circuit(config_path)
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")

        # Get expected parameters from circuit function
        if not hasattr(self.circuit_fn, 'expected_params'):
             raise AttributeError(f"Circuit function for type '{circuit_type}' must have 'expected_params' attribute.")
        self.n_params = self.circuit_fn.expected_params
        logger.info(f"QNN circuit requires {self.n_params} parameters.")

        # Define weight shape for TorchLayer (needs to match circuit_fn's params argument)
        weight_shapes = {"params": self.n_params}

        # Create the TorchLayer
        # The TorchLayer handles the interface between PyTorch Tensors and the QNode execution
        # It expects the circuit function to take weights (params) and inputs (features)
        try:
            # Define the QNode - Use 'backprop' if supported by the device (like default.qubit)
            diff_method_to_use = 'backprop' if self.q_device.supports_derivatives else 'parameter-shift'
            logger.info(f"Using differentiation method: {diff_method_to_use}")

            # Define the circuit function separately for clarity
            _circuit_fn = self.circuit_fn

            @qml.qnode(self.q_device, interface='torch', diff_method=diff_method_to_use)
            def qnode_for_torchlayer(inputs, params):
                 return self.circuit_fn(params=params, features=inputs)

            self.q_layer = qml.qnn.TorchLayer(qnode_for_torchlayer, weight_shapes)
            logger.debug(f"qml.qnn.TorchLayer created successfully with diff_method='{diff_method_to_use}'.")

            # Determine output dimension if not provided
            dummy_weights = {k: torch.randn(*shp if isinstance(shp, tuple) else (shp,), dtype=torch.float64)
                             for k, shp in weight_shapes.items()}
            # Use parameter setting method of TorchLayer
            self.q_layer.weights = dummy_weights['params'] # Directly assign if 'params' is the only weight key
            # self.q_layer.load_state_dict(dummy_weights, strict=False) # Alternative

            self.q_layer.double()

            dummy_input = torch.zeros(n_qubits, dtype=torch.float64)
            dummy_output = self.q_layer(dummy_input)
            self.output_dim = dummy_output.shape[-1]
            logger.info(f"Inferred QNN output dimension: {self.output_dim}")

        except Exception as e:
            logger.error(f"Failed to initialize TorchLayer: {e}", exc_info=True)
            raise

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the quantum layer.

        Args:
            features: Input tensor (batch_size, n_features) or (n_features,).
                    The number of features should match the expectation of the
                    encoding within the circuit function (often n_qubits).

        Returns:
            Output tensor (batch_size, self.output_dim) or (self.output_dim,).
        """
        logger.debug(f"QNN Layer forward pass with input shape: {features.shape}")
        
        # Handle both batched and single inputs
        is_batched = features.dim() > 1
        if is_batched:
            # Process batch one at a time and stack results
            batch_size = features.shape[0]
            results = []
            for i in range(batch_size):
                single_result = self.q_layer(features[i])
                results.append(single_result)
            output = torch.stack(results)
        else:
            # Process a single input
            output = self.q_layer(features)
        
        logger.debug(f"QNN Layer output shape: {output.shape}")
        return output


# Optional QiskitVQC class (conditionally defined)
if QISKIT_AVAILABLE:
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
            # Add other Qiskit feature maps if needed (e.g., PauliFeatureMap)
            # elif feature_map_type.lower() == 'pauli':
            #    self.feature_map = PauliFeatureMap(n_qubits, reps=min(n_layers, 2))
            else:
                logger.warning(f"Unsupported Qiskit feature map type '{feature_map_type}'. Defaulting to ZZFeatureMap.")
                self.feature_map = ZZFeatureMap(n_qubits, reps=min(n_layers, 2))

            # Create variational form (ansatz)
            if ansatz_type.lower() == 'real_amplitudes':
                self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)
            # Add other Qiskit ansatzes if needed (e.g., EfficientSU2)
            # elif ansatz_type.lower() == 'efficientsu2':
            #    self.ansatz = EfficientSU2(n_qubits, reps=n_layers)
            else:
                logger.warning(f"Unsupported Qiskit ansatz type '{ansatz_type}'. Defaulting to RealAmplitudes.")
                self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)

            # Combine feature map and ansatz
            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.circuit.compose(self.feature_map, inplace=True)
            self.circuit.compose(self.ansatz, inplace=True)
            logger.debug("Qiskit circuit composed (FeatureMap + Ansatz).")

            # Create parameters
            self.n_params = self.ansatz.num_parameters
            self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
            logger.info(f"QiskitVQC initialized with {self.n_params} parameters.")

            # Create Aer simulator backend
            try:
                 self.simulator = AerSimulator()
                 logger.debug("Qiskit AerSimulator initialized.")
            except Exception as e:
                 logger.error(f"Failed to initialize Qiskit AerSimulator: {e}", exc_info=True)
                 raise RuntimeError("Could not initialize AerSimulator.") from e

        def run_circuit(self, features: np.ndarray) -> np.ndarray:
            """
            Run the quantum circuit with features and current parameters.

            Args:
                features: Input data sample (1D array, size matching feature map parameters).

            Returns:
                Array of estimated class probabilities based on measurement counts.
            """
            if len(features) != self.feature_map.num_parameters:
                 raise ValueError(f"Feature length ({len(features)}) must match feature map parameters "
                                  f"({self.feature_map.num_parameters}).")

            # Bind parameters
            param_binding = {p: v for p, v in zip(self.ansatz.parameters, self.params)}
            feature_binding = {p: v for p, v in zip(self.feature_map.parameters, features)}
            try:
                bound_circuit = self.circuit.assign_parameters({**feature_binding, **param_binding})
            except Exception as e:
                 logger.error(f"Error binding parameters: {e}", exc_info=True)
                 # Log parameter details for debugging
                 logger.error(f"Feature Map Params ({len(self.feature_map.parameters)}): {self.feature_map.parameters}")
                 logger.error(f"Ansatz Params ({len(self.ansatz.parameters)}): {self.ansatz.parameters}")
                 logger.error(f"Feature values ({len(features)}): {features}")
                 logger.error(f"Param values ({len(self.params)}): {self.params}")
                 raise

            # Add measurement
            measured_circuit = bound_circuit.copy(name="meas_circ") # Give copy a name
            measured_circuit.measure_all(inplace=True)

            # Run simulation
            try:
                # Use execute function for compatibility
                transpiled_circuit = qiskit.transpile(measured_circuit, self.simulator)
                job = self.simulator.run(transpiled_circuit, shots=self.shots)
                result = job.result()
                counts = result.get_counts(0) # Get counts for the first (only) circuit
            except Exception as e:
                 logger.error(f"Error during Qiskit circuit execution: {e}", exc_info=True)
                 raise

            # Convert counts to probabilities
            probabilities = np.zeros(self.n_classes)
            total_shots = sum(counts.values())
            if total_shots == 0: return probabilities # Avoid division by zero

            for bitstring, count in counts.items():
                # Simple mapping: Interpret bitstring as int, modulo n_classes
                # Assumes standard Qiskit bitstring order (rightmost is qubit 0)
                try:
                    class_index = int(bitstring, 2) % self.n_classes
                    probabilities[class_index] += count / total_shots
                except ValueError:
                     logger.warning(f"Could not interpret bitstring '{bitstring}' as integer. Skipping.")
                except IndexError:
                     logger.warning(f"Calculated class index from bitstring '{bitstring}' is out of bounds "
                                    f"for n_classes={self.n_classes}. Skipping.")

            logger.debug(f"Input features: {features}, Output probabilities: {probabilities}")
            return probabilities

        def predict(self, features: np.ndarray) -> int:
            """Make a class prediction for a single data sample."""
            probabilities = self.run_circuit(features)
            prediction = int(np.argmax(probabilities))
            logger.debug(f"QiskitVQC prediction: {prediction} (from probabilities {probabilities})")
            return prediction

        def convert_to_pennylane(self) -> Optional[VariationalQuantumClassifier]:
            """
            Conceptually convert this Qiskit model to a PennyLane VQC.
            Creates a *new* PennyLane model with similar parameters.
            Note: The underlying circuits will differ.

            Returns:
                A new PennyLane VariationalQuantumClassifier instance, or None if conversion fails.
            """
            logger.info("Attempting conceptual conversion from QiskitVQC to PennyLane VQC...")
            try:
                # Create a PennyLane model approximating the structure
                pennylane_model = VariationalQuantumClassifier(
                    n_qubits=self.n_qubits,
                    n_layers=self.n_layers, # Use original layers
                    n_classes=self.n_classes,
                    # Choose PennyLane equivalents - this mapping is approximate!
                    circuit_type='basic', # Or 'complex' - needs mapping logic
                    # encoding_type='angle', # Encoding now internal to circuit
                    device_name='default.qubit', # Use a default device
                    shots=self.shots
                )
                # Attempt to set parameter count if possible, otherwise VQC init handles it
                # pennylane_model.params = np.random.uniform(0, 2*np.pi, size=pennylane_model.n_params) # Re-initialize
                logger.info("Conceptual conversion successful (created new PennyLane VQC).")
                return pennylane_model
            except Exception as e:
                logger.error(f"Failed during conceptual conversion to PennyLane: {e}", exc_info=True)
                return None

else:
    # Define a placeholder if Qiskit is not available
    class QiskitVQC:
        def __init__(self, *args, **kwargs):
            logger.error("QiskitVQC class requires Qiskit installation.")
            raise ImportError("QiskitVQC requires qiskit and qiskit-aer.")

# --- Hybrid Model ---

class HybridQuantumModel(nn.Module):
    """
    Hybrid Quantum-Classical Model using PyTorch nn.Module.
    Combines classical pre/post-processing layers with a QuantumNeuralNetwork layer.
    This model is trained using standard PyTorch optimizers and loss functions.
    """

    def __init__(self,
                 n_qubits: int,
                 classical_input_dim: int,
                 classical_hidden_dim: int,
                 classical_output_dim: int,
                 q_layers: int = 2,
                 q_circuit_type: str = 'basic',
                 q_device_name: str = 'default.qubit',
                 q_shots: Optional[int] = None,
                 q_config_path: Optional[str] = None):
        """
        Initialize the hybrid quantum-classical model.

        Args:
            n_qubits: Number of qubits for the quantum layer.
            classical_input_dim: Dimension of the input features to the entire hybrid model.
            classical_hidden_dim: Dimension of the hidden layers in classical parts.
            classical_output_dim: Dimension of the final output of the hybrid model.
            q_layers: Number of layers in the quantum circuit ansatz.
            q_circuit_type: Type of quantum circuit ('basic', 'complex', 'custom').
            q_device_name: PennyLane device for the quantum layer.
            q_shots: Number of shots for the quantum layer device.
            q_config_path: Path to config for custom quantum circuit.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.classical_input_dim = classical_input_dim
        self.classical_hidden_dim = classical_hidden_dim
        self.classical_output_dim = classical_output_dim

        logger.info(f"Initializing HybridQuantumModel: Input={classical_input_dim}, "
                    f"Hidden={classical_hidden_dim}, Output={classical_output_dim}, "
                    f"Qubits={n_qubits}, QCircuit='{q_circuit_type}'")

        # Classical pre-processing layer(s)
        # Maps input features to the number of features expected by the quantum layer's encoding (typically n_qubits)
        self.classical_pre = nn.Sequential(
            nn.Linear(classical_input_dim, classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim, n_qubits), # Output dim must match quantum layer input expectation
            # Optional: Add activation like Tanh if quantum encoding expects values in a specific range, e.g., [-1, 1]
            # nn.Tanh()
        )
        logger.debug(f"Classical Pre-processing layers initialized: {self.classical_pre}")

        # Quantum layer (using the corrected QuantumNeuralNetwork)
        # The output dimension of the QNN layer is inferred or passed if known
        self.quantum_layer = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=q_layers,
            output_dim=None, # Let QNN infer its output dim
            circuit_type=q_circuit_type,
            device_name=q_device_name,
            shots=q_shots,
            config_path=q_config_path
        )
        # Get the actual output dimension from the quantum layer after it's initialized
        self.quantum_output_dim = self.quantum_layer.output_dim
        logger.debug(f"Quantum layer initialized. Output dimension: {self.quantum_output_dim}")


        # Classical post-processing layer(s)
        # Maps the quantum layer's output to the final desired output dimension
        self.classical_post = nn.Sequential(
            nn.Linear(self.quantum_output_dim, classical_hidden_dim),
            nn.ReLU(),
            nn.Linear(classical_hidden_dim, classical_output_dim)
            # Optional: Add final activation (e.g., LogSoftmax for classification) depending on the task
            # nn.LogSoftmax(dim=1)
        )
        logger.debug(f"Classical Post-processing layers initialized: {self.classical_post}")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete hybrid model.

        Args:
            x: Input tensor (batch_size, classical_input_dim).

        Returns:
            Output tensor (batch_size, classical_output_dim).
        """
        logger.debug(f"Hybrid forward pass started. Input shape: {x.shape}")

        # 1. Classical Pre-processing
        x = self.classical_pre(x)
        logger.debug(f"After classical_pre. Shape: {x.shape}")

        # 2. Quantum Layer
        # The input 'x' now has shape (batch_size, n_qubits) - or whatever classical_pre outputs
        x = self.quantum_layer(x)
        logger.debug(f"After quantum_layer. Shape: {x.shape}") # Should be (batch_size, quantum_output_dim)

        # 3. Classical Post-processing
        x = self.classical_post(x)
        logger.debug(f"After classical_post. Final Output shape: {x.shape}")

        return x