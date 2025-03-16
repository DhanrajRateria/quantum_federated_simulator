"""
Quantum machine learning models for the Quantum Federated Learning Simulator.
This module implements various quantum neural network models using parameterized
quantum circuits defined in the circuits module.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union, Callable
import logging

# Import project components
from .circuits import CircuitFactory
from .encodings import DataEncoder

# Import quantum libraries based on configuration
try:
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.circuit import Parameter
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

logger = logging.getLogger(__name__)

class QuantumModel:
    """Base class for quantum machine learning models."""
    
    def __init__(self, config: Dict):
        """
        Initialize a quantum model.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.backend_name = config.get('backend', 'qiskit')
        self.n_qubits = config.get('n_qubits', 4)
        self.shots = config.get('shots', 1024)
        self.circuit_type = config.get('circuit_type', 'vqc')
        self.encoding_type = config.get('encoding_type', 'angle')
        
        # Initialize circuit factory
        self.circuit_factory = CircuitFactory(config)
        
        # Initialize data encoder
        self.data_encoder = DataEncoder(self.n_qubits, self.encoding_type)
        
        # Initialize parameters
        self.parameters = None
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """Initialize model parameters."""
        self.parameters = self.circuit_factory.initialize_random_parameters(self.circuit_type)
        logger.info(f"Initialized parameters with shape: {self.parameters.shape if hasattr(self.parameters, 'shape') else len(self.parameters)}")
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the quantum model.
        
        Args:
            x: Input data
            
        Returns:
            Model output
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def compute_loss(self, outputs: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute loss based on model outputs and targets.
        
        Args:
            outputs: Model outputs
            targets: Target values
            
        Returns:
            Loss value
        """
        # Default: Mean Squared Error
        return np.mean((outputs - targets) ** 2)
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """
        Update parameters based on gradients.
        
        Args:
            x: Input data
            y: Target values
            learning_rate: Learning rate for parameter updates
        """
        raise NotImplementedError("Subclasses must implement backward method")
    
    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 4, 
            learning_rate: float = 0.01, verbose: bool = True) -> Dict:
        """
        Train the quantum model.
        
        Args:
            x: Training data
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for parameter updates
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = len(x)
        history = {'loss': []}
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Process in batches
            for batch_idx in range(0, n_samples, batch_size):
                batch_x = x[batch_idx:batch_idx + batch_size]
                batch_y = y[batch_idx:batch_idx + batch_size]
                
                # Forward pass
                batch_outputs = np.array([self.forward(sample) for sample in batch_x])
                batch_loss = self.compute_loss(batch_outputs, batch_y)
                epoch_loss += batch_loss * len(batch_x)
                
                # Backward pass
                self.backward(batch_x, batch_y, learning_rate)
            
            # Average loss for epoch
            avg_epoch_loss = epoch_loss / n_samples
            history['loss'].append(avg_epoch_loss)
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        return history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions with the quantum model.
        
        Args:
            x: Input data
            
        Returns:
            Predictions
        """
        return np.array([self.forward(sample) for sample in x])
    
    def get_parameters(self) -> np.ndarray:
        """
        Get model parameters.
        
        Returns:
            Model parameters
        """
        return self.parameters
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """
        Set model parameters.
        
        Args:
            parameters: New parameters
        """
        self.parameters = parameters


class QiskitModel(QuantumModel):
    """Quantum model implementation using Qiskit."""
    
    def __init__(self, config: Dict):
        """
        Initialize a Qiskit quantum model.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__(config)
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required but not installed.")
        
        # Create quantum circuit
        self.circuit = self.circuit_factory.create_circuit(self.circuit_type)
        
        # Setup simulator
        self.simulator = Aer.get_backend('qasm_simulator')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the Qiskit quantum model.
        
        Args:
            x: Input data
            
        Returns:
            Model output (expectation values)
        """
        # Encode data into quantum states
        encoding_circuit = self.data_encoder.encode(x, backend='qiskit')
        
        # Combine encoding and variational circuits
        full_circuit = encoding_circuit.compose(self.circuit)
        
        # Bind parameters
        parameter_dict = {param: value for param, value in zip(self.circuit.parameters, self.parameters)}
        bound_circuit = full_circuit.bind_parameters(parameter_dict)
        
        # Add measurements
        measurement_circuit = bound_circuit.copy()
        measurement_circuit.measure_all()
        
        # Execute circuit
        job = execute(measurement_circuit, self.simulator, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to expectation values
        expectations = self._counts_to_expectations(counts)
        return expectations
    
    def _counts_to_expectations(self, counts: Dict[str, int]) -> np.ndarray:
        """
        Convert measurement counts to expectation values.
        
        Args:
            counts: Dictionary of measurement results
            
        Returns:
            Array of expectation values
        """
        # Initialize expectations for each qubit (Z measurement)
        expectations = np.zeros(self.n_qubits)
        
        total_shots = sum(counts.values())
        
        # Calculate expectation value for each qubit
        for bitstring, count in counts.items():
            # Reverse bitstring to match qubit ordering
            bitstring = bitstring[::-1]
            
            # For each qubit, if measured 0 contribute +1, if 1 contribute -1
            for i, bit in enumerate(bitstring):
                if i < self.n_qubits:
                    # Convert bit to +1 (for |0>) or -1 (for |1>)
                    value = 1 if bit == '0' else -1
                    expectations[i] += value * count
        
        # Normalize by total shots
        expectations /= total_shots
        return expectations
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """
        Update parameters using parameter shift rule for gradients.
        
        Args:
            x: Input data
            y: Target values
            learning_rate: Learning rate for parameter updates
        """
        n_params = len(self.parameters)
        gradients = np.zeros(n_params)
        
        # For each parameter, compute gradient using parameter shift rule
        for p in range(n_params):
            # Shift parameter up
            params_plus = self.parameters.copy()
            params_plus[p] += np.pi/2
            
            # Shift parameter down
            params_minus = self.parameters.copy()
            params_minus[p] -= np.pi/2
            
            # Store original parameters
            original_params = self.parameters.copy()
            
            # Compute expectation with positive shift
            self.parameters = params_plus
            outputs_plus = self.predict(x)
            loss_plus = self.compute_loss(outputs_plus, y)
            
            # Compute expectation with negative shift
            self.parameters = params_minus
            outputs_minus = self.predict(x)
            loss_minus = self.compute_loss(outputs_minus, y)
            
            # Compute gradient
            gradients[p] = 0.5 * (loss_plus - loss_minus)
            
            # Restore original parameters
            self.parameters = original_params
        
        # Update parameters
        self.parameters -= learning_rate * gradients


class PennyLaneModel(QuantumModel):
    """Quantum model implementation using PennyLane."""
    
    def __init__(self, config: Dict):
        """
        Initialize a PennyLane quantum model.
        
        Args:
            config: Dictionary containing model configuration
        """
        super().__init__(config)
        
        if not PENNYLANE_AVAILABLE:
            raise ImportError("PennyLane is required but not installed.")
        
        # Device selection
        self.device_name = config.get('pennylane_device', 'default.qubit')
        self.device = qml.device(self.device_name, wires=self.n_qubits, shots=self.shots)
        
        # Create quantum circuit
        self.circuit_template = self.circuit_factory.create_circuit(self.circuit_type)
        
        # Define the QNode (quantum function)
        @qml.qnode(self.device)
        def quantum_circuit(inputs, weights):
            # Apply data encoding
            encoding_fn = self.data_encoder.encode(inputs, backend='pennylane')
            encoding_fn()
            
            # Apply variational circuit
            self.circuit_template(inputs, weights)
            
            # Return expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the PennyLane quantum model.
        
        Args:
            x: Input data
            
        Returns:
            Model output (expectation values)
        """
        return self.quantum_circuit(x, self.parameters)
    
    def backward(self, x: np.ndarray, y: np.ndarray, learning_rate: float = 0.01):
        """
        Update parameters using automatic differentiation.
        
        Args:
            x: Input data
            y: Target values
            learning_rate: Learning rate for parameter updates
        """
        # Define cost function for this batch
        def cost_fn(params):
            outputs = np.array([self.quantum_circuit(sample, params) for sample in x])
            return self.compute_loss(outputs, y)
        
        # Compute gradient using autograd
        gradient = qml.grad(cost_fn)(self.parameters)
        
        # Update parameters
        self.parameters -= learning_rate * gradient


def create_quantum_model(config: Dict) -> QuantumModel:
    """
    Factory function to create a quantum model based on configuration.
    
    Args:
        config: Dictionary containing model configuration
        
    Returns:
        Instantiated quantum model
    """
    backend = config.get('backend', 'qiskit')
    
    if backend == 'qiskit':
        return QiskitModel(config)
    elif backend == 'pennylane':
        return PennyLaneModel(config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")