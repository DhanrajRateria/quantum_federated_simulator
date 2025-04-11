"""
Unit tests for quantum machine learning models.
"""

import os
import sys
import unittest
import tempfile
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


class TestVariationalQuantumClassifier(unittest.TestCase):
    """Test Variational Quantum Classifier implementation."""
    
    def setUp(self):
        """Set up test environment."""
        set_random_seed(42)
        self.vqc_model = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            circuit_type='basic',
            encoding_type='angle',
            device_name='default.qubit'
        )
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.vqc_model.n_qubits, 4)
        self.assertEqual(self.vqc_model.n_layers, 2)
        self.assertEqual(self.vqc_model.n_classes, 2)
        self.assertEqual(self.vqc_model.circuit_type, 'basic')
        self.assertEqual(self.vqc_model.encoding_type, 'angle')
        self.assertEqual(self.vqc_model.n_params, 24)  # 4 qubits * 3 params * 2 layers
        self.assertEqual(self.vqc_model.params.shape, (24,))
        self.assertIsInstance(self.vqc_model.qnode, qml.QNode)
    
    def test_forward(self):
        """Test forward pass."""
        features = np.random.uniform(-1, 1, size=4)
        output = self.vqc_model.forward(features)
        self.assertEqual(len(output), 4)
        for val in output:
            self.assertTrue(-1 <= val <= 1)
    
    def test_predict(self):
        """Test prediction."""
        features = np.random.uniform(-1, 1, size=4)
        prediction = self.vqc_model.predict(features)
        self.assertIn(prediction, [0, 1])  # Binary classification
    
    def test_loss(self):
        """Test loss calculation."""
        features = np.random.uniform(-1, 1, size=(5, 4))  # 5 samples
        labels = np.random.randint(0, 2, size=5)  # Binary labels
        loss = self.vqc_model.loss(features, labels)
        self.assertIsInstance(loss, float)
        self.assertTrue(0 <= loss <= 2.0)  # MSE loss should be non-negative
    
    def test_train_step(self):
        """Test a single training step."""
        features = np.random.uniform(-1, 1, size=(5, 4))
        labels = np.random.randint(0, 2, size=5)
        initial_params = self.vqc_model.params.copy()
        
        loss = self.vqc_model.train_step(features, labels, learning_rate=0.1)
        
        # Parameters should have changed
        self.assertFalse(np.array_equal(self.vqc_model.params, initial_params))
        self.assertIsInstance(loss, float)
    
    def test_fit(self):
        """Test model training."""
        features = np.random.uniform(-1, 1, size=(10, 4))
        labels = np.random.randint(0, 2, size=10)
        
        # Use only 2 epochs for testing
        loss_history = self.vqc_model.fit(
            features, labels, epochs=2, batch_size=5, learning_rate=0.1, verbose=False
        )
        
        self.assertEqual(len(loss_history), 2)
        for loss in loss_history:
            self.assertIsInstance(loss, float)
    
    def test_save_load_params(self):
        """Test parameter saving and loading."""
        # Create temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save parameters
            params_file = os.path.join(tmpdir, "vqc_params.npy")
            self.vqc_model.save_params(params_file)
            
            # Modify parameters
            original_params = self.vqc_model.params.copy()
            self.vqc_model.params = np.random.uniform(0, 2*np.pi, size=self.vqc_model.n_params)
            
            # Load parameters
            self.vqc_model.load_params(params_file)
            self.assertTrue(np.array_equal(self.vqc_model.params, original_params))


class TestQuantumNeuralNetwork(unittest.TestCase):
    """Test Quantum Neural Network implementation."""
    
    def setUp(self):
        """Create a QNN model for testing."""
        set_random_seed(42)
        self.qnn_model = QuantumNeuralNetwork(
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
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.qnn_model.n_qubits, 4)
        self.assertEqual(self.qnn_model.n_layers, 2)
        self.assertEqual(self.qnn_model.input_size, 4)
        self.assertEqual(self.qnn_model.output_size, 2)
        self.assertEqual(self.qnn_model.circuit_type, 'basic')
        self.assertEqual(self.qnn_model.encoding_type, 'angle')
        self.assertEqual(self.qnn_model.n_params, 24)  # 4 qubits * 3 params * 2 layers
        self.assertEqual(self.qnn_model.quantum_params.shape, (24,))
        self.assertIsInstance(self.qnn_model.qnode, qml.QNode)
        
        # Check PyTorch integration
        self.assertIsInstance(self.qnn_model, torch.nn.Module)
        self.assertIsInstance(self.qnn_model.pre_proc, torch.nn.Sequential)
        self.assertIsInstance(self.qnn_model.post_proc, torch.nn.Sequential)
    
    def test_forward_single_sample(self):
        """Test forward pass with a single sample."""
        x = torch.rand(4)  # Single input sample
        output = self.qnn_model(x)
        self.assertEqual(output.shape, (2,))  # Output size should be 2
    
    def test_forward_batch(self):
        """Test forward pass with a batch of samples."""
        x = torch.rand(3, 4)  # 3 samples, 4 features each
        output = self.qnn_model(x)
        self.assertEqual(output.shape, (3, 2))  # 3 samples, 2 outputs each
    
    def test_integration_with_optimizer(self):
        """Test integration with PyTorch optimizer."""
        # Create a simple classification task
        x = torch.rand(5, 4)
        y = torch.randint(0, 2, (5,))
        
        # Setup optimizer
        optimizer = torch.optim.Adam(self.qnn_model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Initial forward pass
        output = self.qnn_model(x)
        initial_loss = loss_fn(output, y)
        
        # Single optimization step
        optimizer.zero_grad()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        # New forward pass
        new_output = self.qnn_model(x)
        new_loss = loss_fn(new_output, y)
        
        # Parameters should have been updated
        # Note: Loss might not decrease in a single step
        self.assertNotEqual(float(new_loss), float(initial_loss))


@unittest.skipIf(not hasattr(sys.modules, 'qiskit'), "Qiskit not available")
class TestQiskitVQC(unittest.TestCase):
    """Test Qiskit Variational Quantum Classifier implementation."""
    
    def setUp(self):
        """Create a Qiskit VQC model for testing."""
        set_random_seed(42)
        try:
            import qiskit
            self.qiskit_vqc = QiskitVQC(
                n_qubits=2,  # Using fewer qubits for faster tests
                n_layers=1,
                n_classes=2,
                feature_map_type='zz',
                ansatz_type='real_amplitudes',
                shots=100  # Using fewer shots for faster tests
            )
        except ImportError:
            self.skipTest("Qiskit not available")
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.qiskit_vqc.n_qubits, 2)
        self.assertEqual(self.qiskit_vqc.n_layers, 1)
        self.assertEqual(self.qiskit_vqc.n_classes, 2)
        self.assertEqual(self.qiskit_vqc.feature_map_type, 'zz')
        self.assertEqual(self.qiskit_vqc.ansatz_type, 'real_amplitudes')
        self.assertEqual(self.qiskit_vqc.shots, 100)
    
    def test_run_circuit(self):
        """Test running the quantum circuit."""
        features = np.random.uniform(0, 1, size=2)
        probabilities = self.qiskit_vqc.run_circuit(features)
        
        self.assertEqual(len(probabilities), 2)  # Binary classification
        self.assertTrue(np.isclose(sum(probabilities), 1.0, atol=1e-5))  # Should sum to 1
        for p in probabilities:
            self.assertTrue(0 <= p <= 1)  # Valid probabilities
    
    def test_predict(self):
        """Test making predictions."""
        features = np.random.uniform(0, 1, size=2)
        prediction = self.qiskit_vqc.predict(features)
        self.assertIn(prediction, [0, 1])  # Binary classification
    
    def test_convert_to_pennylane(self):
        """Test conversion to PennyLane model."""
        pennylane_model = self.qiskit_vqc.convert_to_pennylane()
        self.assertIsInstance(pennylane_model, VariationalQuantumClassifier)
        self.assertEqual(pennylane_model.n_qubits, self.qiskit_vqc.n_qubits)
        self.assertEqual(pennylane_model.n_layers, self.qiskit_vqc.n_layers)
        self.assertEqual(pennylane_model.n_classes, self.qiskit_vqc.n_classes)


class TestHybridQuantumModel(unittest.TestCase):
    """Test Hybrid Quantum-Classical Model implementation."""
    
    def setUp(self):
        """Create a hybrid model for testing."""
        set_random_seed(42)
        self.hybrid_model = HybridQuantumModel(
            n_qubits=2,  # Using fewer qubits for faster tests
            input_size=8,
            hidden_size=16,
            output_size=3,
            n_layers=1,
            circuit_type='basic'
        )
    
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.hybrid_model.n_qubits, 2)
        self.assertEqual(self.hybrid_model.input_size, 8)
        self.assertEqual(self.hybrid_model.hidden_size, 16)
        self.assertEqual(self.hybrid_model.output_size, 3)
        self.assertEqual(self.hybrid_model.n_layers, 1)
        
        # Check model components
        self.assertIsInstance(self.hybrid_model.classical_pre, torch.nn.Sequential)
        self.assertIsInstance(self.hybrid_model.quantum_model, QuantumNeuralNetwork)
        self.assertIsInstance(self.hybrid_model.classical_post, torch.nn.Sequential)
    
    def test_forward(self):
        """Test forward pass."""
        x = torch.rand(5, 8)  # 5 samples, 8 features
        output = self.hybrid_model.forward(x)
        self.assertEqual(output.shape, (5, 3))  # 5 samples, 3 classes
    
    def test_train_step(self):
        """Test training step."""
        x = torch.rand(5, 8)
        y = torch.randint(0, 3, (5,))
        optimizer = torch.optim.Adam(self.hybrid_model.quantum_model.parameters(), lr=0.01)
        
        loss = self.hybrid_model.train_step(x, y, optimizer)
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)


if __name__ == "__main__":
    unittest.main()