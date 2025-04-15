"""
Unit tests for quantum machine learning models.
Includes logging for execution details.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import pennylane as qml
import logging
from pennylane import PennyLaneDeprecationWarning
import warnings

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs', 'quantum')

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Conditional Qiskit import for skipping tests
try:
    import qiskit
    QISKIT_AVAILABLE_TEST = True
except ImportError:
    QISKIT_AVAILABLE_TEST = False

# --- Import project modules ---
from src.quantum.utils import set_random_seed, load_config
from src.quantum.models import (
    VariationalQuantumClassifier,
    QuantumNeuralNetwork,
    QiskitVQC,
    HybridQuantumModel
)

# --- Test Base Class ---
class BaseTestCase(unittest.TestCase):
    def setUp(self):
        logger.info(f"\n--- Starting test: {self.id()} ---")
        set_random_seed(42)

    def tearDown(self):
        logger.info(f"--- Finished test: {self.id()} ---")

# --- Test Cases ---

class TestVariationalQuantumClassifier(BaseTestCase):
    """Test Variational Quantum Classifier implementation."""

    def setUp(self):
        super().setUp()
        # *** CORRECTED CALL: Removed encoding_type ***
        self.vqc_model = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            n_classes=2,
            circuit_type='basic',
            device_name='default.qubit',
            shots=None
        )
        # *** CORRECTED CALL: Removed encoding_type ***
        self.vqc_multi = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=1,
            n_classes=3,
            circuit_type='basic',
            device_name='default.qubit',
            shots=None
        )

    def test_initialization(self):
        """Test model initialization."""
        logger.info("Testing VQC Initialization")
        self.assertEqual(self.vqc_model.n_qubits, 4)
        self.assertEqual(self.vqc_model.n_layers, 2)
        self.assertEqual(self.vqc_model.n_classes, 2)
        self.assertEqual(self.vqc_model.circuit_type, 'basic')
        self.assertEqual(self.vqc_model.n_params, self.vqc_model.circuit_fn.expected_params)
        self.assertEqual(self.vqc_model.params.shape, (self.vqc_model.n_params,))
        self.assertIsInstance(self.vqc_model.qnode, qml.QNode)
        logger.info("VQC Initialization Test Passed")

    def test_forward(self):
        """Test forward pass."""
        logger.info("Testing VQC Forward Pass")
        features = np.random.uniform(-1, 1, size=self.vqc_model.n_qubits)
        output = self.vqc_model.forward(features)
        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(len(output), self.vqc_model.n_qubits)
        for val in output:
            self.assertTrue(-1.001 <= val <= 1.001)
        logger.info(f"VQC Forward Pass Test Passed. Output shape: {output.shape}")

    def test_predict_binary(self):
        """Test prediction for binary classification."""
        logger.info("Testing VQC Predict (Binary)")
        features = np.random.uniform(-1, 1, size=self.vqc_model.n_qubits)
        prediction = self.vqc_model.predict(features)
        self.assertIn(prediction, [0, 1])
        logger.info(f"VQC Predict (Binary) Test Passed. Prediction: {prediction}")

    def test_predict_multi(self):
        """Test prediction for multi-class classification."""
        logger.info("Testing VQC Predict (Multi-class)")
        features = np.random.uniform(-1, 1, size=self.vqc_multi.n_qubits)
        prediction = self.vqc_multi.predict(features)
        self.assertIn(prediction, [0, 1, 2])
        logger.info(f"VQC Predict (Multi-class) Test Passed. Prediction: {prediction}")

    def test_loss_binary(self):
        """Test loss calculation for binary classification."""
        logger.info("Testing VQC Loss (Binary)")
        features = np.random.uniform(-1, 1, size=(5, self.vqc_model.n_qubits))
        labels = np.random.randint(0, 2, size=5)
        loss = self.vqc_model.loss(features, labels)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)
        logger.info(f"VQC Loss (Binary) Test Passed. Loss: {loss:.6f}")

    def test_loss_multi(self):
        """Test loss calculation for multi-class classification."""
        logger.info("Testing VQC Loss (Multi-class)")
        features = np.random.uniform(-1, 1, size=(5, self.vqc_multi.n_qubits))
        labels = np.random.randint(0, self.vqc_multi.n_classes, size=5)
        loss = self.vqc_multi.loss(features, labels)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0.0)
        logger.info(f"VQC Loss (Multi-class) Test Passed. Loss: {loss:.6f}")

    def test_train_step(self):
        """Test a single training step."""
        logger.info("Testing VQC Train Step")
        features = np.random.uniform(-1, 1, size=(5, self.vqc_model.n_qubits))
        labels = np.random.randint(0, 2, size=5)
        initial_params = self.vqc_model.params.copy()

        loss_before = self.vqc_model.train_step(features, labels, learning_rate=0.1)

        self.assertFalse(np.array_equal(self.vqc_model.params, initial_params))
        self.assertIsInstance(loss_before, float)
        logger.info(f"VQC Train Step Test Passed. Loss before step: {loss_before:.6f}")

    def test_fit(self):
        """Test model training over a few epochs."""
        logger.info("Testing VQC Fit")
        features = np.random.uniform(-1, 1, size=(20, self.vqc_model.n_qubits))
        labels = np.random.randint(0, 2, size=20)
        initial_params = self.vqc_model.params.copy()

        loss_history = self.vqc_model.fit(
            features, labels, epochs=3, batch_size=5, learning_rate=0.1, verbose=True
        )

        self.assertEqual(len(loss_history), 3)
        self.assertFalse(np.array_equal(self.vqc_model.params, initial_params))
        logger.info(f"VQC Fit Test Passed. Final loss: {loss_history[-1]:.6f}")

    def test_save_load_params(self):
        """Test parameter saving and loading."""
        logger.info("Testing VQC Save/Load Parameters")
        with tempfile.TemporaryDirectory() as tmpdir:
            params_file = os.path.join(tmpdir, "vqc_params.npy")
            original_params = self.vqc_model.params.copy()
            self.vqc_model.save_params(params_file)

            self.vqc_model.params = np.random.rand(*original_params.shape) * 0.01
            self.assertFalse(np.array_equal(self.vqc_model.params, original_params))

            self.vqc_model.load_params(params_file)
            np.testing.assert_array_almost_equal(self.vqc_model.params, original_params)
            logger.info("VQC Save/Load Parameters Test Passed")


class TestQuantumNeuralNetwork(BaseTestCase):
    """Test Quantum Neural Network TorchLayer implementation."""

    def setUp(self):
        """Create a QNN model for testing."""
        super().setUp()
        self.n_qubits = 2
        self.n_layers = 1
        self.input_features = self.n_qubits
        # *** CORRECTED CALL: Removed unused/implicit args ***
        self.qnn_layer = QuantumNeuralNetwork(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            output_dim=None, # Let it infer
            circuit_type='basic',
            device_name='default.qubit',
            shots=None
        )
        self.output_dim = self.qnn_layer.output_dim

    def test_initialization(self):
        """Test model initialization."""
        logger.info("Testing QNN Initialization")
        self.assertEqual(self.qnn_layer.n_qubits, self.n_qubits)
        self.assertEqual(self.qnn_layer.n_layers, self.n_layers)
        self.assertEqual(self.qnn_layer.output_dim, self.n_qubits)
        self.assertEqual(self.qnn_layer.circuit_type, 'basic')
        self.assertIsInstance(self.qnn_layer.q_layer, qml.qnn.TorchLayer)
        self.assertTrue(hasattr(self.qnn_layer.q_layer, 'params'))
        self.assertEqual(self.qnn_layer.q_layer.params.shape, (self.qnn_layer.n_params,))
        logger.info(f"QNN Initialization Test Passed. Output dim: {self.output_dim}")

    def test_forward_single_sample(self):
        """Test forward pass with a single sample."""
        logger.info("Testing QNN Forward (Single)")
        x = torch.rand(self.input_features, dtype=torch.float64)
        self.qnn_layer.double()
        output = self.qnn_layer(x)
        self.assertEqual(output.shape, (self.output_dim,))
        logger.info(f"QNN Forward (Single) Test Passed. Output shape: {output.shape}")

    def test_forward_batch(self):
        """Test forward pass with a batch of samples."""
        logger.info("Testing QNN Forward (Batch)")
        batch_size = 3
        x = torch.rand(batch_size, self.input_features, dtype=torch.float64)
        self.qnn_layer.double()
        output = self.qnn_layer(x)
        self.assertEqual(output.shape, (batch_size, self.output_dim,))
        logger.info(f"QNN Forward (Batch) Test Passed. Output shape: {output.shape}")

    def test_backward_pass(self):
        """Test if gradients can be computed (backward pass)."""
        logger.info("Testing QNN Backward Pass")
        batch_size = 2
        x = torch.rand(batch_size, self.input_features, dtype=torch.float64, requires_grad=False)
        self.qnn_layer.double()

        for param in self.qnn_layer.parameters():
            self.assertTrue(param.requires_grad)

        output = self.qnn_layer(x)
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        loss = output.sum()

        try:
            loss.backward()
            logger.info("Backward pass executed successfully.")
        except Exception as e:
            self.fail(f"Backward pass failed: {e}")

        grad_exists = False
        for param in self.qnn_layer.parameters():
            self.assertIsNotNone(param.grad, "Gradient should exist for quantum parameters.")
            if param.grad is not None:
                 logger.debug(f"Param grad norm: {torch.linalg.norm(param.grad)}")
                 if torch.linalg.norm(param.grad) > 1e-9:
                     grad_exists = True
        self.assertTrue(grad_exists, "Gradients seem to be zero for all parameters.")
        logger.info("QNN Backward Pass Test Passed.")


@unittest.skipIf(not QISKIT_AVAILABLE_TEST, "Qiskit/Aer not available")
class TestQiskitVQC(BaseTestCase):
    """Test Qiskit Variational Quantum Classifier implementation."""
    # --- This class remains unchanged as its setUp was correct ---
    def setUp(self):
        super().setUp()
        self.n_qubits = 2
        self.qiskit_vqc = QiskitVQC(
            n_qubits=self.n_qubits,
            n_layers=1,
            n_classes=2,
            feature_map_type='zz',
            ansatz_type='real_amplitudes',
            shots=1024
        )
    # ... tests remain the same ...
    def test_initialization(self):
        logger.info("Testing QiskitVQC Initialization")
        self.assertEqual(self.qiskit_vqc.n_qubits, self.n_qubits)
        self.assertEqual(self.qiskit_vqc.feature_map.num_qubits, self.n_qubits)
        self.assertEqual(self.qiskit_vqc.ansatz.num_qubits, self.n_qubits)
        self.assertEqual(self.qiskit_vqc.n_params, self.qiskit_vqc.ansatz.num_parameters)
        logger.info("QiskitVQC Initialization Test Passed")

    def test_run_circuit(self):
        logger.info("Testing QiskitVQC Run Circuit")
        n_features = self.qiskit_vqc.feature_map.num_parameters
        features = np.random.uniform(0, np.pi, size=n_features)
        probabilities = self.qiskit_vqc.run_circuit(features)
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), self.qiskit_vqc.n_classes)
        self.assertTrue(np.isclose(sum(probabilities), 1.0, atol=1e-5))
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
        logger.info(f"QiskitVQC Run Circuit Test Passed. Probabilities: {probabilities}")

    def test_predict(self):
        logger.info("Testing QiskitVQC Predict")
        n_features = self.qiskit_vqc.feature_map.num_parameters
        features = np.random.uniform(0, np.pi, size=n_features)
        prediction = self.qiskit_vqc.predict(features)
        self.assertIn(prediction, [0, 1])
        logger.info(f"QiskitVQC Predict Test Passed. Prediction: {prediction}")

    def test_convert_to_pennylane(self):
        logger.info("Testing QiskitVQC Convert to PennyLane")
        pennylane_model = self.qiskit_vqc.convert_to_pennylane()
        self.assertIsInstance(pennylane_model, VariationalQuantumClassifier)
        self.assertEqual(pennylane_model.n_qubits, self.qiskit_vqc.n_qubits)
        self.assertEqual(pennylane_model.n_classes, self.qiskit_vqc.n_classes)
        self.assertEqual(pennylane_model.shots, self.qiskit_vqc.shots)
        logger.info("QiskitVQC Convert to PennyLane Test Passed.")


class TestHybridQuantumModel(BaseTestCase):
    """Test Hybrid Quantum-Classical Model implementation."""

    def setUp(self):
        """Create a hybrid model for testing."""
        super().setUp()
        self.input_dim = 5
        self.hidden_dim = 8
        self.output_dim = 2
        self.n_qubits = 2
        # *** CORRECTED CALL: Use corrected arg names ***
        self.hybrid_model = HybridQuantumModel(
            n_qubits=self.n_qubits,
            classical_input_dim=self.input_dim,    # Corrected arg name
            classical_hidden_dim=self.hidden_dim,  # Corrected arg name
            classical_output_dim=self.output_dim, # Corrected arg name
            q_layers=1,
            q_circuit_type='basic',
            q_device_name='default.qubit',
            q_shots=None
        )
        self.hybrid_model.double()


    def test_initialization(self):
        """Test model initialization."""
        logger.info("Testing Hybrid Model Initialization")
        self.assertEqual(self.hybrid_model.n_qubits, self.n_qubits)
        self.assertEqual(self.hybrid_model.classical_input_dim, self.input_dim)
        self.assertEqual(self.hybrid_model.classical_hidden_dim, self.hidden_dim)
        self.assertEqual(self.hybrid_model.classical_output_dim, self.output_dim)
        self.assertIsInstance(self.hybrid_model.classical_pre, torch.nn.Sequential)
        self.assertIsInstance(self.hybrid_model.quantum_layer, QuantumNeuralNetwork)
        self.assertIsInstance(self.hybrid_model.classical_post, torch.nn.Sequential)
        logger.info("Hybrid Model Initialization Test Passed")

    def test_forward(self):
        """Test forward pass."""
        logger.info("Testing Hybrid Model Forward Pass")
        batch_size = 3
        x = torch.rand(batch_size, self.input_dim, dtype=torch.float64)
        output = self.hybrid_model(x)
        self.assertEqual(output.shape, (batch_size, self.output_dim))
        logger.info(f"Hybrid Model Forward Pass Test Passed. Output shape: {output.shape}")

    def test_pytorch_training_step(self):
        """Test a standard PyTorch training step."""
        logger.info("Testing Hybrid Model PyTorch Training Step")
        batch_size = 4
        x = torch.rand(batch_size, self.input_dim, dtype=torch.float64)
        y_target = torch.randint(0, self.output_dim, (batch_size,))
        loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.hybrid_model.parameters(), lr=0.01)
        initial_param_norm = torch.linalg.norm(next(iter(self.hybrid_model.parameters()))).item()

        optimizer.zero_grad()
        output = self.hybrid_model(x)
        loss = loss_fn(output, y_target)
        loss.backward()
        optimizer.step()

        final_param_norm = torch.linalg.norm(next(iter(self.hybrid_model.parameters()))).item()

        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertNotAlmostEqual(initial_param_norm, final_param_norm, places=6,
                                   msg="Parameters did not seem to update after optimizer step.")
        logger.info(f"Hybrid Model PyTorch Training Step Test Passed. Loss: {loss.item():.6f}")


if __name__ == "__main__":
    unittest.main()