"""
Integration tests for quantum federated learning client and server.
"""

import os
import sys
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import logging

# Add project root to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adjust if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)

from src.federated.aggregation import FedAvg
from src.quantum.models import VariationalQuantumClassifier
from src.quantum.models import QuantumNeuralNetwork
from src.core.quantum_client import QuantumFederatedClient
from src.core.quantum_manager import QuantumAggregationStrategy, QuantumFederatedServer
from src.quantum.utils import set_random_seed

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class TestQuantumFederatedIntegration(unittest.TestCase):
    """Test integration of quantum client and server."""

    def setUp(self):
        """Set up test environment."""
        logger.info(f"\n--- Setting up test: {self.id()} ---")
        set_random_seed(42)

        # Create synthetic dataset
        n_samples = 60 # Increased slightly
        n_features = 4
        X = np.random.uniform(-1, 1, size=(n_samples, n_features))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        X_tensor = torch.tensor(X, dtype=torch.float64) # Use float64
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        # Model configuration (VQC for this test)
        self.n_qubits = n_features
        self.n_layers = 1
        self.model_kwargs = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_classes": 2,
            "circuit_type": 'basic',
            # "encoding_type" no longer needed here, handled by circuit
            "shots": None # Use analytic for faster tests
        }
        self.client_model_kwargs = self.model_kwargs.copy()

        # Initialize QuantumFederatedServer
        global_model_instance = VariationalQuantumClassifier(**self.model_kwargs)
        self.server = QuantumFederatedServer(
            model=global_model_instance,
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()), # Wrap FedAvg
            evaluation_dataset=self.test_dataset,
            device=torch.device('cpu') # Force CPU for simplicity in basic tests
        )

        # Create client datasets
        self.n_clients = 2
        client_indices = np.array_split(np.arange(train_size), self.n_clients)
        self.client_datasets = [Subset(self.train_dataset, indices) for indices in client_indices]

    def tearDown(self):
        logger.info(f"--- Tearing down test: {self.id()} ---")

    def test_quantum_client_creation_and_params(self):
        """Test creating quantum client and parameter handling."""
        logger.info("Testing Quantum Client Creation & Params")
        client_model_instance = VariationalQuantumClassifier(**self.client_model_kwargs)
        client = QuantumFederatedClient(
            client_id="q_client_1",
            model=client_model_instance,
            dataset=self.client_datasets[0],
            batch_size=10,
            learning_rate=0.1,
            device=self.server.device # Match server device
        )

        self.assertEqual(client.client_id, "q_client_1")
        self.assertEqual(client.model_type, "vqc")
        self.assertIsNone(client.optimizer) # VQC has no external optimizer

        # Test parameter access
        params = client.get_parameters()
        self.assertIn("quantum_params", params)
        self.assertEqual(params["quantum_params"].dtype, torch.float64) # Check dtype
        self.assertEqual(params["quantum_params"].device.type, 'cpu') # Should be on CPU
        # Calculate expected params based on circuit
        expected_n_params = client.model.n_params # Get from VQC instance
        self.assertEqual(params["quantum_params"].shape[0], expected_n_params)

        # Test setting parameters
        new_param_val = torch.randn(expected_n_params, dtype=torch.float64)
        client.set_parameters({"quantum_params": new_param_val})
        np.testing.assert_array_almost_equal(client.model.params, new_param_val.numpy())
        logger.info("Quantum Client Creation & Params Test Passed.")


    def test_quantum_client_training_vqc(self):
        """Test training a VQC quantum client."""
        logger.info("Testing Quantum Client Training (VQC)")
        client_model_instance = VariationalQuantumClassifier(**self.client_model_kwargs)
        client = QuantumFederatedClient(
            client_id="q_client_2",
            model=client_model_instance,
            dataset=self.client_datasets[0],
            batch_size=10,
            learning_rate=0.1,
            device=self.server.device
        )

        initial_params_dict = client.get_parameters()
        initial_params_tensor = initial_params_dict["quantum_params"].clone()

        # Train for one epoch
        updated_params_dict, num_samples = client.train(epochs=1)
        updated_params_tensor = updated_params_dict["quantum_params"]

        self.assertFalse(torch.allclose(initial_params_tensor, updated_params_tensor))
        self.assertEqual(num_samples, len(self.client_datasets[0]))
        logger.info("Quantum Client Training (VQC) Test Passed.")

    def test_federated_quantum_round_vqc(self):
        """Test a complete federated round with VQC."""
        logger.info("Testing Federated Quantum Round (VQC)")
        # Register clients
        client_list = []
        for i in range(self.n_clients):
            client_model = VariationalQuantumClassifier(**self.client_model_kwargs)
            client = QuantumFederatedClient(
                client_id=f"q_client_{i}",
                model=client_model,
                dataset=self.client_datasets[i],
                batch_size=10,
                learning_rate=0.1,
                device=self.server.device
            )
            self.server.register_client(client)
            client_list.append(client)

        initial_params_dict = self.server.get_parameters()

        # Run a training round
        round_info = self.server.train_round(local_epochs=1)

        self.assertEqual(round_info.get("status"), "success")
        self.assertIn('participating_clients', round_info)
        self.assertEqual(len(round_info['participating_clients']), self.n_clients)
        self.assertIn('evaluation_metrics', round_info)
        self.assertIn('accuracy', round_info['evaluation_metrics']) # Check evaluation ran

        updated_params_dict = self.server.get_parameters()
        self.assertFalse(torch.allclose(
            initial_params_dict["quantum_params"],
            updated_params_dict["quantum_params"]
        ))
        logger.info("Federated Quantum Round (VQC) Test Passed.")


if __name__ == "__main__":
    unittest.main()