"""
Unit tests for quantum machine learning models.
"""

import os
import sys
import pytest
import numpy as np
import torch
import pennylane as qml

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.quantum.utils import set_random_seed
from src.quantum.models import (
    VariationalQuantumClassifier,
    QuantumNeuralNetwork,
    QiskitVQC,
    HybridQuantumModel
)
from src.quantum.encodings import (
    angle_encoding,
    amplitude_encoding,
    basis_encoding,
    iqp_feature_map
)


# Set random seed for reproducibility
@pytest.fixture(scope="module")
def seed():
    """Set random seed for tests."""
    set_random_seed(42)


class TestVariationalQuantumClassifier:
    """Test Variational Quantum Classifier implementation."""
    
    @pytest.fixture
    def vqc_model(self, seed):
        """Create a VQC model for testing."""
        return VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            circuit_type='basic',
            encoding_type='angle',
            device_name='default.qubit'
        )
    
    def test_initialization(self, vqc_model):
        """Test model initialization."""
        assert vqc_model.n_qubits == 4
        assert vqc_model.n_layers == 2
        assert vqc_model.n_classes == 2
        assert vqc_model.circuit_type == 'basic'
        assert vqc_model.encoding_type == 'angle'
        assert vqc_model.n_params == 24  # 4 qubits * 3 params * 2 layers
        assert vqc_model.params.shape == (24,)
        assert isinstance(vqc_model.qnode, qml.QNode)
    
    def test_forward(self, vqc_model):
        """Test forward pass."""
        features = np.random.uniform(-1, 1, size=4)
        output = vqc_model.forward(features)
        assert len(output) == 4
        assert all(-1 <= val <= 1 for val in output)
    
    def test_predict(self, vqc_model):
        """Test prediction."""
        features = np.random.uniform(-1, 1, size=4)
        prediction = vqc_model.predict(features)
        assert prediction in [0, 1]  # Binary classification
    
    def test_loss(self, vqc_model):
        """Test loss calculation."""
        features = np.random.uniform(-1, 1, size=(5, 4))  # 5 samples
        labels = np.random.randint(0, 2, size=5)  # Binary labels
        loss = vqc_model.loss(features, labels)
        assert isinstance(loss, float)
        assert 0 <= loss <= 2.0  # MSE loss should be non-negative
    
    def test_train_step(self, vqc_model):
        """Test a single training step."""
        features = np.random.uniform(-1, 1, size=(5, 4))
        labels = np.random.randint(0, 2, size=5)
        initial_params = vqc_model.params.copy()
        
        loss = vqc_model.train_step(features, labels, learning_rate=0.1)
        
        # Parameters should have changed
        assert not np.array_equal(vqc_model.params, initial_params)
        assert isinstance(loss, float)
    
    def test_fit(self, vqc_model):
        """Test model training."""
        features = np.random.uniform(-1, 1, size=(10, 4))
        labels = np.random.randint(0, 2, size=10)
        
        # Use only 2 epochs for testing
        loss_history = vqc_model.fit(
            features, labels, epochs=2, batch_size=5, learning_rate=0.1, verbose=False
        )
        
        assert len(loss_history) == 2
        assert all(isinstance(loss, float) for loss in loss_history)
    
    def test_save_load_params(self, vqc_model, tmpdir):
        """Test parameter saving and loading."""
        # Save parameters
        params_file = os.path.join(tmpdir, "vqc_params.npy")
        vqc_model.save_params(params_file)
        
        # Modify parameters
        original_params = vqc_model.params.copy()
        vqc_model.params = np.random.uniform(0, 2*np.pi, size=vqc_model.n_params)
        
        # Load parameters
        vqc_model.load_params(params_file)
        assert np.array_equal(vqc_model.params, original_params)


class TestQuantumNeuralNetwork:
    """Test Quantum Neural Network implementation."""
    
    @pytest.fixture
    def qnn_model(self, seed):
        """Create a QNN model for testing."""
        return QuantumNeuralNetwork(
            n_qubits=4,
            n_layers=2,
            input_size=4,
            output_size=2,
            circuit_type='basic',
            encoding_type='angle',
            device_name='default.qubit',
            pre_processing=True,
            post_processing=True
        )
    
    def test_initialization(self, qnn_model):
        """Test model initialization."""
        assert qnn_model.n_qubits == 4
        assert qnn_model.n_layers == 2
        assert qnn_model.input_size == 4
        assert qnn_model.output_size == 2
        assert qnn_model.circuit_type == 'basic'
        assert qnn_model.encoding_type == 'angle'
        assert qnn_model.n_params == 24  # 4 qubits * 3 params * 2 layers
        assert qnn_model.quantum_params.shape == (24,)
        assert isinstance(qnn_model.qnode, qml.QNode)
        
        # Check PyTorch integration
        assert isinstance(qnn_model, torch.nn.Module)
        assert isinstance(qnn_model.pre_proc, torch.nn.Sequential)
        assert isinstance(qnn_model.post_proc, torch.nn.Sequential)
    
    def test_forward_single_sample(self, qnn_model):
        """Test forward pass with a single sample."""
        x = torch.rand(4)  # Single input sample
        output = qnn_model(x)
        assert output.shape == (2,)  # Output size should be 2
    
    def test_forward_batch(self, qnn_model):
        """Test forward pass with a batch of samples."""
        x = torch.rand(3, 4)  # 3 samples, 4 features each
        output = qnn_model(x)
        assert output.shape == (3, 2)  # 3 samples, 2 outputs each
    
    def test_integration_with_optimizer(self, qnn_model):
        """Test integration with PyTorch optimizer."""
        # Create a simple classification task
        x = torch.rand(5, 4)
        y = torch.randint(0, 2, (5,))
        
        # Setup optimizer
        optimizer = torch.optim.Adam(qnn_model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initial forward pass
        output = qnn_model(x)
        initial_loss = loss_fn(output, y)
        
        # Single optimization step
        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        # New forward pass
        new_output = qnn_model(x)
        new_loss = loss_fn(new_output, y)
        
        # Parameters should have been updated
        # Note: Loss might not decrease in a single step
        assert new_loss != initial_loss


class TestQiskitVQC:
    """Test Qiskit Variational Quantum Classifier implementation."""
    
    @pytest.fixture
    def qiskit_vqc(self, seed):
        """Create a Qiskit VQC model for testing."""
        try:
            # Skip if qiskit is not available
            import qiskit
            return QiskitVQC(
                n_qubits=2,  # Using fewer qubits for faster tests
                n_layers=1,
                n_classes=2,
                feature_map_type='zz',
                ansatz_type='real_amplitudes',
                shots=100  # Using fewer shots for faster tests
            )
        except ImportError:
            pytest.skip("Qiskit not available")
    
    def test_initialization(self, qiskit_vqc):
        """Test model initialization."""
        assert qiskit_vqc.n_qubits == 2
        assert qiskit_vqc.n_layers == 1
        assert qiskit_vqc.n_classes == 2
        assert qiskit_vqc.feature_map_type == 'zz'
        assert qiskit_vqc.ansatz_type == 'real_amplitudes'
        assert qiskit_vqc.shots == 100
    
    def test_run_circuit(self, qiskit_vqc):
        """Test running the quantum circuit."""
        features = np.random.uniform(0, 1, size=2)
        probabilities = qiskit_vqc.run_circuit(features)
        
        assert len(probabilities) == 2  # Binary classification
        assert np.isclose(sum(probabilities), 1.0, atol=1e-5)  # Should sum to 1
        assert all(0 <= p <= 1 for p in probabilities)  # Valid probabilities
    
    def test_predict(self, qiskit_vqc):
        """Test making predictions."""
        features = np.random.uniform(0, 1, size=2)
        prediction = qiskit_vqc.predict(features)
        assert prediction in [0, 1]  # Binary classification
    
    def test_convert_to_pennylane(self, qiskit_vqc):
        """Test conversion to PennyLane model."""
        pennylane_model = qiskit_vqc.convert_to_pennylane()
        assert isinstance(pennylane_model, VariationalQuantumClassifier)
        assert pennylane_model.n_qubits == qiskit_vqc.n_qubits
        assert pennylane_model.n_layers == qiskit_vqc.n_layers
        assert pennylane_model.n_classes == qiskit_vqc.n_classes


class TestHybridQuantumModel:
    """Test Hybrid Quantum-Classical Model implementation."""
    
    @pytest.fixture
    def hybrid_model(self, seed):
        """Create a hybrid model for testing."""
        return HybridQuantumModel(
            n_qubits=2,  # Using fewer qubits for faster tests
            input_size=8,
            hidden_size=16,
            output_size=3,
            n_layers=1,
            circuit_type='basic'
        )
    
    def test_initialization(self, hybrid_model):
        """Test model initialization."""
        assert hybrid_model.n_qubits == 2
        assert hybrid_model.input_size == 8
        assert hybrid_model.hidden_size == 16
        assert hybrid_model.output_size == 3
        assert hybrid_model.n_layers == 1
        
        # Check model components
        assert isinstance(hybrid_model.classical_pre, torch.nn.Sequential)
        assert isinstance(hybrid_model.quantum_model, QuantumNeuralNetwork)
        assert isinstance(hybrid_model.classical_post, torch.nn.Sequential)
    
    def test_forward(self, hybrid_model):
        """Test forward pass."""
        x = torch.rand(5, 8)  # 5 samples, 8 features
        output = hybrid_model.forward(x)
        assert output.shape == (5, 3)  # 5 samples, 3 classes
    
    def test_train_step(self, hybrid_model):
        """Test training step."""
        x = torch.rand(5, 8)
        y = torch.randint(0, 3, (5,))
        optimizer = torch.optim.Adam(hybrid_model.quantum_model.parameters(), lr=0.01)
        
        loss = hybrid_model.train_step(x, y, optimizer)
        assert isinstance(loss, float)
        assert loss > 0


if __name__ == "__main__":
    pytest.main(["-v"])