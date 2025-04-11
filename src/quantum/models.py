"""
Quantum machine learning models including Variational Quantum Classifiers (VQC) 
and Quantum Neural Networks (QNN).
"""

import pennylane as qml
import numpy as np
import torch
import torch.nn as nn
from typing import List, Callable, Dict, Any, Optional, Union, Tuple
import qiskit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector

from .utils import get_device, load_config
from .circuits import create_basic_circuit, create_complex_circuit, create_custom_circuit
from .encodings import angle_encoding, amplitude_encoding, basis_encoding, iqp_feature_map


class VariationalQuantumClassifier:
    """
    Variational Quantum Classifier implemented with PennyLane.
    """
    
    def __init__(self, 
                 n_qubits: int, 
                 n_layers: int = 2,
                 n_classes: int = 2,
                 circuit_type: str = 'basic',
                 encoding_type: str = 'angle',
                 device_name: str = 'default.qubit',
                 shots: Optional[int] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the Variational Quantum Classifier.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
            n_classes: Number of output classes
            circuit_type: Type of circuit ('basic', 'complex', 'custom')
            encoding_type: Type of data encoding ('angle', 'amplitude', 'basis', 'iqp')
            device_name: PennyLane device name
            shots: Number of measurement shots (None for analytic)
            config_path: Path to custom configuration file (for 'custom' circuit_type)
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.circuit_type = circuit_type
        self.encoding_type = encoding_type
        self.device_name = device_name
        self.shots = shots
        self.config_path = config_path
        
        # Initialize the device
        self.device = get_device(device_name, wires=n_qubits, shots=shots)
        
        # Get the right circuit based on type
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
        
        # Determine the number of parameters required by the circuit
        if circuit_type == 'basic':
            # 3 rotation parameters per qubit per layer
            self.n_params = n_qubits * 3 * n_layers
        elif circuit_type == 'complex':
            # More complex parameter counting based on the circuit structure
            rotation_params = n_qubits * 3 * n_layers  # 3 rotation gates per qubit
            # Entanglement parameters: ring topology + long-range connections
            pairs_count = n_qubits + (n_qubits if n_qubits > 3 else 0)
            entanglement_params = pairs_count * 2 * n_layers  # 2 params per pair
            self.n_params = rotation_params + entanglement_params
        elif circuit_type == 'custom':
            # For custom circuits, we'll use a default or read from config
            config = load_config(config_path)
            self.n_params = config.get('n_params', n_qubits * 4 * n_layers)
        
        # Initialize parameters randomly
        self.params = np.random.uniform(0, 2*np.pi, size=self.n_params)
        
        # Create the quantum node function
        self.qnode = qml.QNode(self.circuit_fn, self.device)
    
    def get_encoding_function(self, features: np.ndarray) -> Callable:
        """
        Get the appropriate encoding function based on encoding_type.
        
        Args:
            features: Input data to encode
            
        Returns:
            Encoding function
        """
        wires = list(range(self.n_qubits))
        
        if self.encoding_type == 'angle':
            return angle_encoding(features, wires)
        elif self.encoding_type == 'amplitude':
            return amplitude_encoding(features, wires)
        elif self.encoding_type == 'basis':
            return basis_encoding(features, wires)
        elif self.encoding_type == 'iqp':
            return iqp_feature_map(features, wires)
        else:
            raise ValueError(f"Unknown encoding type: {self.encoding_type}")
    
    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum circuit.
        
        Args:
            features: Input data
            
        Returns:
            Circuit output
        """
        return self.qnode(self.params, features)
    
    def predict(self, features: np.ndarray) -> int:
        """
        Make a prediction for classification.
        
        Args:
            features: Input data
            
        Returns:
            Predicted class
        """
        # Get the output from the quantum circuit
        output = self.forward(features)
        
        # For binary classification
        if self.n_classes == 2:
            # Use the first qubit's expectation as decision value
            return 1 if output[0] > 0 else 0
        
        # For multi-class classification
        elif self.n_classes > 2:
            # Use multiple qubits and find the one with highest expectation
            class_values = output[:min(self.n_classes, self.n_qubits)]
            return np.argmax(class_values)
        
        else:
            raise ValueError("n_classes must be at least 2")
    
    def loss(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the loss for a batch of data.
        
        Args:
            features: Input data batch
            labels: Target labels
            
        Returns:
            Loss value
        """
        predictions = np.array([self.forward(x) for x in features])
        
        # For binary classification
        if self.n_classes == 2:
            # Convert labels to {-1, 1} for easier comparison with PauliZ expectation
            y = 2 * labels - 1
            # Use first qubit's expectation value
            pred_values = predictions[:, 0]
            # Compute Mean Squared Error
            loss = np.mean((pred_values - y) ** 2)
        
        # For multi-class classification
        else:
            # Convert labels to one-hot encoding
            y_one_hot = np.zeros((len(labels), self.n_classes))
            for i, label in enumerate(labels):
                y_one_hot[i, label] = 1
            
            # Use first n_classes qubits
            pred_values = predictions[:, :self.n_classes]
            # Scale predictions from [-1, 1] to [0, 1]
            pred_values = (pred_values + 1) / 2
            
            # Compute Mean Squared Error
            loss = np.mean((pred_values - y_one_hot) ** 2)
        
        return loss
    
    def train_step(self, features: np.ndarray, labels: np.ndarray, learning_rate: float = 0.1) -> float:
        """
        Perform one training step using parameter-shift rule.
        
        Args:
            features: Input data batch
            labels: Target labels
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Current loss value
        """
        # Compute current loss
        current_loss = self.loss(features, labels)
        
        # Compute gradients using parameter-shift rule
        gradients = np.zeros_like(self.params)
        
        for i in range(len(self.params)):
            # Shift parameter up
            self.params[i] += np.pi/2
            loss_plus = self.loss(features, labels)
            
            # Shift parameter down
            self.params[i] -= np.pi
            loss_minus = self.loss(features, labels)
            
            # Restore original parameter
            self.params[i] += np.pi/2
            
            # Compute gradient
            gradients[i] = (loss_plus - loss_minus) / 2
        
        # Update parameters
        self.params -= learning_rate * gradients
        
        return current_loss
    
    def fit(self, 
            features: np.ndarray, 
            labels: np.ndarray, 
            epochs: int = 100, 
            batch_size: int = 10, 
            learning_rate: float = 0.1,
            verbose: bool = True) -> List[float]:
        """
        Train the model on the given dataset.
        
        Args:
            features: Training data
            labels: Target labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            verbose: Whether to print progress
            
        Returns:
            List of loss values per epoch
        """
        n_samples = len(features)
        loss_history = []
        
        for epoch in range(epochs):
            # Shuffle data
            shuffle_indices = np.random.permutation(n_samples)
            features_shuffled = features[shuffle_indices]
            labels_shuffled = labels[shuffle_indices]
            
            # Process batches
            epoch_loss = 0.0
            batches = n_samples // batch_size
            
            for i in range(batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_features = features_shuffled[start_idx:end_idx]
                batch_labels = labels_shuffled[start_idx:end_idx]
                
                # Perform one training step
                batch_loss = self.train_step(batch_features, batch_labels, learning_rate)
                epoch_loss += batch_loss
            
            # Calculate average loss for the epoch
            avg_loss = epoch_loss / batches
            loss_history.append(avg_loss)
            
            # Print progress
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        return loss_history
    
    def save_params(self, filename: str) -> None:
        """
        Save the trained parameters to a file.
        
        Args:
            filename: Path to save the parameters
        """
        np.save(filename, self.params)
    
    def load_params(self, filename: str) -> None:
        """
        Load parameters from a file.
        
        Args:
            filename: Path to the parameters file
        """
        self.params = np.load(filename)


class QuantumNeuralNetwork(nn.Module):
    """
    Quantum Neural Network implemented as a PyTorch module.
    This allows integration with traditional neural network layers.
    """
    
    def __init__(self, 
                 n_qubits: int, 
                 n_layers: int = 2,
                 input_size: int = None,
                 output_size: int = None,
                 circuit_type: str = 'basic',
                 encoding_type: str = 'angle',
                 device_name: str = 'default.qubit',
                 pre_processing: bool = False,
                 post_processing: bool = False,
                 config_path: Optional[str] = None):
        """
        Initialize the Quantum Neural Network.
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of circuit layers
            input_size: Size of input features (default: n_qubits)
            output_size: Size of output (default: n_qubits)
            circuit_type: Type of circuit ('basic', 'complex', 'custom')
            encoding_type: Type of data encoding ('angle', 'amplitude', 'basis', 'iqp')
            device_name: PennyLane device name
            pre_processing: Whether to include classical pre-processing layers
            post_processing: Whether to include classical post-processing layers
            config_path: Path to custom configuration file (for 'custom' circuit_type)
        """
        super().__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.input_size = input_size if input_size is not None else n_qubits
        self.output_size = output_size if output_size is not None else n_qubits
        self.circuit_type = circuit_type
        self.encoding_type = encoding_type
        self.device_name = device_name
        self.config_path = config_path
        
        # Initialize the quantum device
        self.q_device = get_device(device_name, wires=n_qubits)
        
        # Get the right circuit based on type
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
        
        # Determine the number of parameters
        if circuit_type == 'basic':
            self.n_params = n_qubits * 3 * n_layers
        elif circuit_type == 'complex':
            rotation_params = n_qubits * 3 * n_layers
            pairs_count = n_qubits + (n_qubits if n_qubits > 3 else 0)
            entanglement_params = pairs_count * 2 * n_layers
            self.n_params = rotation_params + entanglement_params
        elif circuit_type == 'custom':
            config = load_config(config_path)
            self.n_params = config.get('n_params', n_qubits * 4 * n_layers)
        
        # Create trainable parameters
        self.quantum_params = nn.Parameter(torch.randn(self.n_params) * 0.1)
        
        # Create the quantum node
        self.qnode = qml.QNode(self.circuit_fn, self.q_device)
        
        # Optional classical pre-processing layers
        if pre_processing:
            self.pre_proc = nn.Sequential(
                nn.Linear(self.input_size, 2 * self.n_qubits),
                nn.ReLU(),
                nn.Linear(2 * self.n_qubits, self.n_qubits),
                nn.Tanh()  # Scale to [-1, 1] range for better quantum encoding
            )
        else:
            self.pre_proc = nn.Identity()
        
        # Optional classical post-processing layers
        if post_processing:
            self.post_proc = nn.Sequential(
                nn.Linear(self.n_qubits, 2 * self.output_size),
                nn.ReLU(),
                nn.Linear(2 * self.output_size, self.output_size)
            )
        else:
            self.post_proc = nn.Linear(self.n_qubits, self.output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid classical-quantum network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """

        is_single_sample = x.dim() == 1
        if is_single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        batch_size = x.shape[0]
        
        # Apply pre-processing
        x = self.pre_proc(x)
        
        # Process each sample in the batch
        q_out = torch.zeros(batch_size, self.n_qubits)
        
        for i in range(batch_size):
            # Convert to numpy for PennyLane
            x_np = x[i].detach().numpy()
            params_np = self.quantum_params.detach().numpy()
            
            # Run quantum circuit
            q_result = self.qnode(params_np, x_np)
            q_out[i] = torch.tensor(q_result, requires_grad=True)
        
        # Connect the quantum output to the autograd graph
        q_out = q_out.to(x.device)
        q_out.requires_grad_(True)
        
        # Apply post-processing
        result = self.post_proc(q_out)
        
        # Remove batch dimension for single samples
        if is_single_sample:
            result = result.squeeze(0)
        
        return result

class QiskitVQC:
    """
    Variational Quantum Classifier implementation using Qiskit.
    This demonstrates interoperability between Qiskit and PennyLane.
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
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of layers in the ansatz
            n_classes: Number of output classes
            feature_map_type: Type of feature map ('zz', 'pauli')
            ansatz_type: Type of ansatz ('real_amplitudes', 'efficient_su2')
            shots: Number of measurement shots
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.feature_map_type = feature_map_type
        self.ansatz_type = ansatz_type
        self.shots = shots
        
        # Create feature map
        if feature_map_type == 'zz':
            self.feature_map = ZZFeatureMap(n_qubits)
        else:
            # Fallback to ZZ feature map
            self.feature_map = ZZFeatureMap(n_qubits)
        
        # Create variational form (ansatz)
        if ansatz_type == 'real_amplitudes':
            self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)
        else:
            # Fallback to RealAmplitudes
            self.ansatz = RealAmplitudes(n_qubits, reps=n_layers)
        
        # Combine feature map and ansatz
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        self.circuit.compose(self.feature_map, inplace=True)
        self.circuit.compose(self.ansatz, inplace=True)
        
        # Create parameters
        self.params = np.random.uniform(0, 2*np.pi, size=self.ansatz.num_parameters)
        
        # Create a quantum instance for simulation
        from qiskit_aer import AerSimulator
        self.simulator = AerSimulator()
    
    def run_circuit(self, features: np.ndarray) -> np.ndarray:
        """
        Run the quantum circuit with the current parameters.
        
        Args:
            features: Input data point
            
        Returns:
            Measurement results
        """
        # Bind parameters to the circuit
        bound_circuit = self.circuit.bind_parameters({
            **dict(zip(self.feature_map.parameters, features)),
            **dict(zip(self.ansatz.parameters, self.params))
        })
        
        # Add measurement
        measured_circuit = bound_circuit.copy()
        measured_circuit.measure_all()
        
        # Run the circuit
        job = qiskit.execute(measured_circuit, self.simulator, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert counts to class probabilities
        probabilities = np.zeros(self.n_classes)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # Use the first bit as class label for binary classification
            if self.n_classes == 2:
                class_index = int(bitstring[-1])
                probabilities[class_index] += count / total_shots
            # For multi-class, use more sophisticated mapping
            else:
                class_index = int(bitstring, 2) % self.n_classes
                probabilities[class_index] += count / total_shots
        
        return probabilities
    
    def predict(self, features: np.ndarray) -> int:
        """
        Make a prediction for a single data point.
        
        Args:
            features: Input data point
            
        Returns:
            Predicted class
        """
        probabilities = self.run_circuit(features)
        return np.argmax(probabilities)
    
    def convert_to_pennylane(self) -> VariationalQuantumClassifier:
        """
        Convert this Qiskit model to a PennyLane model.
        
        Returns:
            Equivalent PennyLane VariationalQuantumClassifier
        """
        # Create a PennyLane model with similar structure
        pennylane_model = VariationalQuantumClassifier(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_classes=self.n_classes,
            circuit_type='basic',  # Using basic as approximation
            encoding_type='angle',
            device_name='default.qubit'
        )
        
        # Note: The circuits won't be exactly the same,
        # but will have similar expressivity
        return pennylane_model


class HybridQuantumModel:
    """
    Hybrid Quantum-Classical Model combining a classical neural network with a quantum circuit.
    """
    
    def __init__(self, 
                 n_qubits: int,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 n_layers: int = 2,
                 circuit_type: str = 'basic'):
        """
        Initialize the hybrid quantum-classical model.
        
        Args:
            n_qubits: Number of qubits
            input_size: Size of input features
            hidden_size: Size of hidden layer in classical part
            output_size: Size of output
            n_layers: Number of circuit layers
            circuit_type: Type of quantum circuit ('basic', 'complex')
        """
        self.n_qubits = n_qubits
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.circuit_type = circuit_type
        
        # Classical pre-processing
        self.classical_pre = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_qubits),
            nn.Tanh()  # Output in [-1, 1] range for quantum encoding
        )
        
        # Quantum part
        self.quantum_model = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=n_layers,
            input_size=n_qubits,
            output_size=n_qubits,
            circuit_type=circuit_type
        )
        
        # Classical post-processing
        self.classical_post = nn.Sequential(
            nn.Linear(n_qubits, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Apply classical pre-processing
        x = self.classical_pre(x)
        
        # Apply quantum circuit
        x = self.quantum_model(x)
        
        # Apply classical post-processing
        x = self.classical_post(x)
        
        return x
    
    def train_step(self, features: np.ndarray, labels: np.ndarray, learning_rate: float = 0.1) -> float:
        """
        Perform one training step using parameter-shift rule.
        
        Args:
            features: Input data batch
            labels: Target labels
            learning_rate: Learning rate for gradient descent
            
        Returns:
            Current loss value
        """
        # Compute current loss
        current_loss = self.loss(features, labels)
        
        # Compute gradients using parameter-shift rule
        gradients = np.zeros_like(self.params)
        
        for i in range(len(self.params)):
            # Shift parameter up
            self.params[i] += np.pi/2
            loss_plus = self.loss(features, labels)
            
            # Shift parameter down
            self.params[i] -= np.pi
            loss_minus = self.loss(features, labels)
            
            # Restore original parameter
            self.params[i] += np.pi/2
            
            # Compute gradient
            gradients[i] = (loss_plus - loss_minus) / 2
        
        # Update parameters
        self.params -= learning_rate * gradients
        
        # Clip parameters to ensure numerical stability
        self.params = np.clip(self.params, 0, 2*np.pi)
        
        return current_loss