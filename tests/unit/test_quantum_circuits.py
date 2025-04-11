"""
Unit tests for quantum circuit implementations.
"""

import os
import sys
import unittest
import numpy as np
import pennylane as qml
import tempfile

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.quantum.utils import load_config, set_random_seed
from src.quantum.circuits import (
    create_basic_circuit, 
    create_complex_circuit, 
    create_custom_circuit,
    create_noisy_circuit
)


class TestBasicCircuit(unittest.TestCase):
    """Test basic circuit implementation."""
    
    def setUp(self):
        """Set up test environment."""
        set_random_seed(42)
        self.n_qubits = 4
        self.n_layers = 2
        self.device = qml.device("default.qubit", wires=self.n_qubits)
    
    def test_creation(self):
        """Test circuit creation."""
        circuit_fn = create_basic_circuit(self.n_qubits, self.n_layers)
        self.assertTrue(callable(circuit_fn))
    
    def test_execution(self):
        """Test circuit execution."""
        circuit_fn = create_basic_circuit(self.n_qubits, self.n_layers)
        qnode = qml.QNode(circuit_fn, self.device)
        
        n_params = self.n_qubits * 3 * self.n_layers  # 3 rotation gates per qubit per layer
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)
        
        result = qnode(params, features)
        self.assertEqual(len(result), self.n_qubits)
        for val in result:
            self.assertTrue(-1 <= val <= 1)  # PauliZ expectation values
    
    def test_parameter_count(self):
        """Test parameter count in circuit."""
        circuit_fn = create_basic_circuit(self.n_qubits, self.n_layers)
        
        # Expected parameter count - now including data encoding parameters
        data_encoding_params = self.n_qubits  # One RX gate per qubit for data encoding
        circuit_params = self.n_qubits * 3 * self.n_layers  # Actual circuit parameters
        expected_param_count = circuit_params + data_encoding_params
        
        # Create a QNode to inspect the circuit
        dev = qml.device("default.qubit", wires=self.n_qubits)
        qnode = qml.QNode(circuit_fn, dev)
        
        # Execute the QNode to build the circuit
        params = np.random.uniform(0, 2*np.pi, size=circuit_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)
        qnode(params, features)
        
        # Count parameters in the circuit (including data encoding)
        param_count = 0
        for op in qnode.qtape.operations:
            if op.name in ['RX', 'RY', 'RZ'] and not isinstance(op.parameters[0], np.ndarray):
                param_count += 1
                
        self.assertEqual(param_count, expected_param_count)


class TestComplexCircuit(unittest.TestCase):
    """Test complex circuit implementation."""
    
    def setUp(self):
        """Set up test environment."""
        set_random_seed(42)
        self.n_qubits = 4
        self.n_layers = 2
        self.device = qml.device("default.qubit", wires=self.n_qubits)
    
    def test_creation(self):
        """Test circuit creation."""
        circuit_fn = create_complex_circuit(self.n_qubits, self.n_layers)
        self.assertTrue(callable(circuit_fn))
    
    def test_execution(self):
        """Test circuit execution."""
        circuit_fn = create_complex_circuit(self.n_qubits, self.n_layers)
        qnode = qml.QNode(circuit_fn, self.device)
        
        # Parameter count for complex circuit
        rotation_params = self.n_qubits * 3 * self.n_layers
        pairs_count = self.n_qubits + (self.n_qubits if self.n_qubits > 3 else 0)
        entanglement_params = pairs_count * 2 * self.n_layers
        n_params = rotation_params + entanglement_params
        
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)
        
        result = qnode(params, features)
        self.assertTrue(len(result) >= self.n_qubits)  # May include two-qubit observables
        for val in result:
            self.assertTrue(-1 <= val <= 1)  # Expectation values
    
    def test_parameter_count(self):
        """Test parameter count in complex circuit."""
        circuit_fn = create_complex_circuit(self.n_qubits, self.n_layers)
        
        # Expected parameter count - now including data encoding parameters
        data_encoding_params = self.n_qubits  # One RX gate per qubit for data encoding
        rotation_params = self.n_qubits * 3 * self.n_layers
        pairs_count = self.n_qubits + (self.n_qubits if self.n_qubits > 3 else 0)
        entanglement_params = pairs_count * 2 * self.n_layers
        expected_param_count = rotation_params + entanglement_params + data_encoding_params
        
        # Create a QNode to inspect the circuit
        dev = qml.device("default.qubit", wires=self.n_qubits)
        qnode = qml.QNode(circuit_fn, dev)
        
        # Execute the QNode to build the circuit
        circuit_params = rotation_params + entanglement_params
        params = np.random.uniform(0, 2*np.pi, size=circuit_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)
        qnode(params, features)
        
        # Count parameters in the circuit (including data encoding)
        param_count = 0
        for op in qnode.qtape.operations:
            if op.name in ['RX', 'RY', 'RZ', 'CRX', 'CRY'] and not isinstance(op.parameters[0], np.ndarray):
                param_count += 1
                
        self.assertEqual(param_count, expected_param_count)


class TestCustomCircuit(unittest.TestCase):
    """Test custom circuit from configuration."""
    
    def setUp(self):
        """Set up test environment."""
        set_random_seed(42)
        # Create a temporary config file
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        
        config_content = """
        n_qubits: 4
        n_layers: 2
        n_params: 24
        gate_set:
          - rx
          - ry
          - rz
          - cz
        entanglement: linear
        encoding: angle
        measurement: z
        """
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_creation(self):
        """Test custom circuit creation."""
        circuit_fn = create_custom_circuit(self.config_path)
        self.assertTrue(callable(circuit_fn))
    
    def test_execution(self):
        """Test custom circuit execution."""
        circuit_fn = create_custom_circuit(self.config_path)
        
        config = load_config(self.config_path)
        n_qubits = config.get('n_qubits', 4)
        n_params = config.get('n_params', 24)
        
        device = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(circuit_fn, device)
        
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=n_qubits)
        
        result = qnode(params, features)
        self.assertEqual(len(result), n_qubits)  # Default measurement returns one value per qubit
        for val in result:
            self.assertTrue(-1 <= val <= 1)  # Expectation values


class TestNoisyCircuit(unittest.TestCase):
    """Test noisy circuit implementation."""
    
    def setUp(self):
        """Set up test environment."""
        set_random_seed(42)
        self.n_qubits = 4
        self.n_layers = 2
        self.device = qml.device("default.qubit", wires=self.n_qubits)
    
    def test_creation(self):
        """Test noisy circuit creation."""
        base_circuit = create_basic_circuit(self.n_qubits, self.n_layers)
        noise_model = {'type': 'depolarizing', 'probability': 0.01}
        
        noisy_circuit = create_noisy_circuit(base_circuit, noise_model)
        self.assertTrue(callable(noisy_circuit))
    
    def test_execution(self):
        """Test noisy circuit execution."""
        base_circuit = create_basic_circuit(self.n_qubits, self.n_layers)
        noise_model = {'type': 'depolarizing', 'probability': 0.01}
        
        noisy_circuit = create_noisy_circuit(base_circuit, noise_model)
        qnode = qml.QNode(noisy_circuit, self.device)
        
        n_params = self.n_qubits * 3 * self.n_layers
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)
        
        # Should run without errors
        result = qnode(params, features)
        self.assertEqual(len(result), self.n_qubits)


if __name__ == "__main__":
    unittest.main()