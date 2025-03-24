"""
Unit tests for quantum circuit implementations.
"""

import os
import sys
import pytest
import numpy as np
import pennylane as qml

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ...src.quantum.utils import load_config, set_random_seed
from ...src.quantum.circuits import (
    create_basic_circuit, 
    create_complex_circuit, 
    create_custom_circuit,
    create_noisy_circuit
)


# Set random seed for reproducibility
@pytest.fixture(scope="module")
def seed():
    """Set random seed for tests."""
    set_random_seed(42)


@pytest.fixture(scope="module")
def basic_setup():
    """Basic test setup."""
    n_qubits = 4
    n_layers = 2
    device = qml.device("default.qubit", wires=n_qubits)
    return n_qubits, n_layers, device


class TestBasicCircuit:
    """Test basic circuit implementation."""
    
    def test_creation(self, seed, basic_setup):
        """Test circuit creation."""
        n_qubits, n_layers, _ = basic_setup
        circuit_fn = create_basic_circuit(n_qubits, n_layers)
        assert callable(circuit_fn)
    
    def test_execution(self, seed, basic_setup):
        """Test circuit execution."""
        n_qubits, n_layers, device = basic_setup
        circuit_fn = create_basic_circuit(n_qubits, n_layers)
        qnode = qml.QNode(circuit_fn, device)
        
        n_params = n_qubits * 3 * n_layers  # 3 rotation gates per qubit per layer
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=n_qubits)
        
        result = qnode(params, features)
        assert len(result) == n_qubits
        assert all(-1 <= val <= 1 for val in result)  # PauliZ expectation values
    
    def test_parameter_count(self, seed, basic_setup):
        """Test parameter count in circuit."""
        n_qubits, n_layers, _ = basic_setup
        circuit_fn = create_basic_circuit(n_qubits, n_layers)
        
        # Expected parameter count
        expected_param_count = n_qubits * 3 * n_layers
        
        # Create a QNode to inspect the circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(circuit_fn, dev)
        
        # Execute the QNode to build the circuit
        params = np.random.uniform(0, 2*np.pi, size=expected_param_count)
        features = np.random.uniform(-1, 1, size=n_qubits)
        qnode(params, features)
        
        # Count parameters in the circuit (only the trainable ones)
        param_count = 0
        for op in qnode.qtape.operations:
            if op.name in ['RX', 'RY', 'RZ'] and not isinstance(op.parameters[0], np.ndarray):
                param_count += 1
                
        assert param_count == expected_param_count


class TestComplexCircuit:
    """Test complex circuit implementation."""
    
    def test_creation(self, seed, basic_setup):
        """Test circuit creation."""
        n_qubits, n_layers, _ = basic_setup
        circuit_fn = create_complex_circuit(n_qubits, n_layers)
        assert callable(circuit_fn)
    
    def test_execution(self, seed, basic_setup):
        """Test circuit execution."""
        n_qubits, n_layers, device = basic_setup
        circuit_fn = create_complex_circuit(n_qubits, n_layers)
        qnode = qml.QNode(circuit_fn, device)
        
        # Parameter count for complex circuit
        rotation_params = n_qubits * 3 * n_layers
        pairs_count = n_qubits + (n_qubits if n_qubits > 3 else 0)
        entanglement_params = pairs_count * 2 * n_layers
        n_params = rotation_params + entanglement_params
        
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=n_qubits)
        
        result = qnode(params, features)
        assert len(result) >= n_qubits  # May include two-qubit observables
        assert all(-1 <= val <= 1 for val in result)  # Expectation values
    
    def test_parameter_count(self, seed, basic_setup):
        """Test parameter count in complex circuit."""
        n_qubits, n_layers, _ = basic_setup
        circuit_fn = create_complex_circuit(n_qubits, n_layers)
        
        # Expected parameter count
        rotation_params = n_qubits * 3 * n_layers
        pairs_count = n_qubits + (n_qubits if n_qubits > 3 else 0)
        entanglement_params = pairs_count * 2 * n_layers
        expected_param_count = rotation_params + entanglement_params
        
        # Create a QNode to inspect the circuit
        dev = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(circuit_fn, dev)
        
        # Execute the QNode to build the circuit
        params = np.random.uniform(0, 2*np.pi, size=expected_param_count)
        features = np.random.uniform(-1, 1, size=n_qubits)
        qnode(params, features)
        
        # Count parameters in the circuit (only the trainable ones)
        param_count = 0
        for op in qnode.qtape.operations:
            if op.name in ['RX', 'RY', 'RZ', 'CRX', 'CRY'] and not isinstance(op.parameters[0], np.ndarray):
                param_count += 1
                
        assert param_count == expected_param_count


class TestCustomCircuit:
    """Test custom circuit from configuration."""
    
    @pytest.fixture
    def config_path(self, tmpdir):
        """Create a temporary config file."""
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
        config_file = tmpdir.join("test_config.yaml")
        config_file.write(config_content)
        return str(config_file)
    
    def test_creation(self, seed, config_path):
        """Test custom circuit creation."""
        circuit_fn = create_custom_circuit(config_path)
        assert callable(circuit_fn)
    
    def test_execution(self, seed, config_path):
        """Test custom circuit execution."""
        circuit_fn = create_custom_circuit(config_path)
        
        config = load_config(config_path)
        n_qubits = config.get('n_qubits', 4)
        n_params = config.get('n_params', 24)
        
        device = qml.device("default.qubit", wires=n_qubits)
        qnode = qml.QNode(circuit_fn, device)
        
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=n_qubits)
        
        result = qnode(params, features)
        assert len(result) == n_qubits  # Default measurement returns one value per qubit
        assert all(-1 <= val <= 1 for val in result)  # Expectation values


class TestNoisyCircuit:
    """Test noisy circuit implementation."""
    
    def test_creation(self, seed, basic_setup):
        """Test noisy circuit creation."""
        n_qubits, n_layers, _ = basic_setup
        base_circuit = create_basic_circuit(n_qubits, n_layers)
        noise_model = {'type': 'depolarizing', 'probability': 0.01}
        
        noisy_circuit = create_noisy_circuit(base_circuit, noise_model)
        assert callable(noisy_circuit)
    
    def test_execution(self, seed, basic_setup):
        """Test noisy circuit execution."""
        n_qubits, n_layers, device = basic_setup
        base_circuit = create_basic_circuit(n_qubits, n_layers)
        noise_model = {'type': 'depolarizing', 'probability': 0.01}
        
        noisy_circuit = create_noisy_circuit(base_circuit, noise_model)
        qnode = qml.QNode(noisy_circuit, device)
        
        n_params = n_qubits * 3 * n_layers
        params = np.random.uniform(0, 2*np.pi, size=n_params)
        features = np.random.uniform(-1, 1, size=n_qubits)
        
        # Should run without errors
        result = qnode(params, features)
        assert len(result) == n_qubits


if __name__ == "__main__":
    pytest.main(["-v"])