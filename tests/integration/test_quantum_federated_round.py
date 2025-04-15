"""
Integration tests for complete quantum federated learning rounds
with different models and strategies.
"""
import unittest
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
import logging
import os
import sys

# Add project root to path for imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Adjust if needed
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)


from src.federated.aggregation import FedAvg, FedProx
from src.federated.utils import set_seed
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.core.quantum_client import QuantumFederatedClient
from src.core.quantum_manager import QuantumFederatedServer, QuantumAggregationStrategy, FederatedQuantumManager

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


class TestQuantumFederatedRoundIntegration(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        logger.info(f"\n--- Setting up test: {self.id()} ---")
        set_seed(42)

        # Create synthetic dataset
        n_samples = 100; n_features = 4
        X = np.random.uniform(-1, 1, size=(n_samples, n_features))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int) # Binary classification
        # Use float64 for better compatibility with Pennylane default backends
        X_tensor = torch.tensor(X, dtype=torch.float64)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)

        train_size = int(0.8 * n_samples); test_size = n_samples - train_size
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        # Common parameters
        self.n_qubits = n_features # Match features for simple encoding
        self.n_layers = 1 # Use 1 layer for faster tests
        self.num_clients = 3 # Use fewer clients for faster QNN/Hybrid tests
        self.device = torch.device('cpu') # Force CPU for consistency

        # Client datasets (use subset for potentially slower models)
        client_indices = np.array_split(np.arange(train_size), self.num_clients)
        self.client_datasets = [Subset(self.train_dataset, indices) for indices in client_indices]

        # --- VQC Setup ---
        self.vqc_model_kwargs = {
            "n_qubits": self.n_qubits, "n_layers": self.n_layers, "n_classes": 2,
            "circuit_type": 'basic', "shots": None # Analytic
        }
        self.vqc_client_kwargs = {"batch_size": 8, "learning_rate": 0.1, "device": self.device}

        # --- QNN Setup ---
        # Output dim is 2 for binary classification logits/scores
        self.qnn_model_kwargs = {
            "n_qubits": self.n_qubits, "n_layers": self.n_layers, "output_dim": 2,
            "circuit_type": 'basic', "shots": None, "device_name": "default.qubit"
        }
        # QNN/Hybrid clients need optimizer and loss
        self.torch_client_kwargs = {
             "batch_size": 8, "learning_rate": 0.01,
             "optimizer_class": torch.optim.Adam, "loss_fn": nn.CrossEntropyLoss(),
             "device": self.device
        }

        # --- Hybrid Setup ---
        self.hybrid_model_kwargs = {
             "n_qubits": self.n_qubits,
             "classical_input_dim": n_features, # Input to classical_pre matches original features
             "classical_hidden_dim": 8,
             "classical_output_dim": 2, # Final output (e.g., logits for 2 classes)
             "q_layers": self.n_layers,
             "q_circuit_type": 'basic',
             "q_shots": None,
             "q_device_name": "default.qubit"
        }

    def tearDown(self):
        logger.info(f"--- Tearing down test: {self.id()} ---")

    # --- Helper to setup server and clients ---
    def _setup_scenario(self, model_class, model_kwargs, client_kwargs, aggregation_strategy, num_clients_to_use=None):
         """Initializes server and clients for a specific scenario."""
         logger.debug(f"Setting up scenario for {model_class.__name__}")
         server = FederatedQuantumManager.initialize_quantum_server(
             model_class=model_class, model_kwargs=model_kwargs,
             aggregation_strategy=aggregation_strategy,
             evaluation_dataset=self.test_dataset,
             server_kwargs={"device": self.device}
         )
         num_clients = num_clients_to_use if num_clients_to_use is not None else self.num_clients
         client_ids = [f"{model_class.__name__.lower()}_client_{i}" for i in range(num_clients)]

         # Ensure model_kwargs for client are consistent (e.g., if server model modified kwargs)
         client_model_kwargs = model_kwargs.copy()

         clients = FederatedQuantumManager.create_quantum_clients(
             client_ids=client_ids, model_class=model_class, model_kwargs=client_model_kwargs,
             datasets=self.client_datasets[:num_clients], client_kwargs=client_kwargs
         )
         for client in clients: server.register_client(client)
         # Ensure model and clients are using the correct dtype (float64 for pennylane default)
         if isinstance(server.model, nn.Module):
             server.model.double()
             for client in clients:
                 if isinstance(client.model, nn.Module):
                     client.model.double()
         logger.debug(f"Scenario setup complete for {model_class.__name__}")
         return server, clients

    # --- VQC Tests (Keep as they were) ---
    def test_vqc_fedavg_round(self):
        """Test VQC with FedAvg."""
        logger.info("Testing VQC + FedAvg Round")
        server, _ = self._setup_scenario(
            VariationalQuantumClassifier, self.vqc_model_kwargs, self.vqc_client_kwargs,
            QuantumAggregationStrategy(FedAvg()), # Use Quantum strategy for VQC
            num_clients_to_use=self.num_clients # Use all defined clients
        )
        round_info = server.train_round(local_epochs=1)
        self.assertEqual(round_info.get("status"), "success")
        self.assertEqual(len(round_info['participating_clients']), self.num_clients)
        self.assertIn('accuracy', round_info['evaluation_metrics'])
        self.assertIsNone(round_info['evaluation_metrics'].get('loss'))

    def test_vqc_fedprox_round(self):
        """Test VQC with FedProx."""
        logger.info("Testing VQC + FedProx Round")
        server, _ = self._setup_scenario(
            VariationalQuantumClassifier, self.vqc_model_kwargs, self.vqc_client_kwargs,
            QuantumAggregationStrategy(FedAvg()),
            num_clients_to_use=self.num_clients
        )
        mu = 0.1
        round_info = server.train_round(local_epochs=1, proximal_term=mu)
        self.assertEqual(round_info.get("status"), "success")
        self.assertEqual(len(round_info['participating_clients']), self.num_clients)
        self.assertIn('accuracy', round_info['evaluation_metrics'])

    def test_vqc_multiple_rounds(self):
        """Test multiple VQC rounds."""
        logger.info("Testing VQC Multiple Rounds")
        server, _ = self._setup_scenario(
            VariationalQuantumClassifier, self.vqc_model_kwargs, self.vqc_client_kwargs,
            QuantumAggregationStrategy(FedAvg()),
            num_clients_to_use=self.num_clients
        )
        num_rounds = 3
        history = server.train(num_rounds=num_rounds, local_epochs=1)
        self.assertEqual(len(history), num_rounds)
        self.assertTrue(all(r.get("status") == "success" for r in history))
        acc = [r['evaluation_metrics'].get('accuracy', -1) for r in history] # Use -1 default if missing
        self.assertTrue(all(a >= 0.0 for a in acc)) # Check accuracy is valid
        logger.info(f"VQC Accuracies over {num_rounds} rounds: {[f'{a:.4f}' for a in acc]}")

    def test_vqc_client_fraction(self):
        """Test VQC with client fraction."""
        logger.info("Testing VQC Client Fraction")
        server, _ = self._setup_scenario(
            VariationalQuantumClassifier, self.vqc_model_kwargs, self.vqc_client_kwargs,
            QuantumAggregationStrategy(FedAvg()),
            num_clients_to_use=self.num_clients
        )
        fraction = 0.6; min_clients = 1
        round_info = server.train_round(local_epochs=1, client_fraction=fraction, min_clients=min_clients)
        self.assertEqual(round_info.get("status"), "success")
        expected_clients = max(min_clients, int(fraction * self.num_clients))
        self.assertEqual(len(round_info['participating_clients']), expected_clients)

    # --- QNN Tests (Enabled) ---
    # @unittest.skip("QNN tests can be slow, skipping by default")
    def test_qnn_fedavg_round(self):
        """Test QNN with FedAvg."""
        logger.info("Testing QNN + FedAvg Round")
        server, _ = self._setup_scenario(
            QuantumNeuralNetwork, self.qnn_model_kwargs, self.torch_client_kwargs,
            FedAvg(), # Use standard FedAvg for PyTorch models
            num_clients_to_use=self.num_clients
        )
        round_info = server.train_round(local_epochs=1)
        self.assertEqual(round_info.get("status"), "success")
        self.assertEqual(len(round_info['participating_clients']), self.num_clients)
        self.assertIn('accuracy', round_info['evaluation_metrics'])
        self.assertIsNotNone(round_info['evaluation_metrics'].get('loss')) # Should have loss
        self.assertGreaterEqual(round_info['evaluation_metrics']['accuracy'], 0.0)
        self.assertLessEqual(round_info['evaluation_metrics']['accuracy'], 1.0)
        logger.info(f"QNN FedAvg Round Test Passed. Metrics: {round_info['evaluation_metrics']}")


    # --- Hybrid Tests (Enabled) ---
    # @unittest.skip("Hybrid tests can be slow, skipping by default")
    def test_hybrid_fedavg_round(self):
        """Test Hybrid model with FedAvg."""
        logger.info("Testing Hybrid + FedAvg Round")
        server, _ = self._setup_scenario(
            HybridQuantumModel, self.hybrid_model_kwargs, self.torch_client_kwargs,
            FedAvg(), # Use standard FedAvg
            num_clients_to_use=self.num_clients
        )
        round_info = server.train_round(local_epochs=1)
        self.assertEqual(round_info.get("status"), "success")
        self.assertEqual(len(round_info['participating_clients']), self.num_clients)
        self.assertIn('accuracy', round_info['evaluation_metrics'])
        self.assertIsNotNone(round_info['evaluation_metrics'].get('loss'))
        self.assertGreaterEqual(round_info['evaluation_metrics']['accuracy'], 0.0)
        self.assertLessEqual(round_info['evaluation_metrics']['accuracy'], 1.0)
        logger.info(f"Hybrid FedAvg Round Test Passed. Metrics: {round_info['evaluation_metrics']}")


    # --- Manager Test (Keep as it was) ---
    def test_quantum_manager_initialization(self):
        """Test the FederatedQuantumManager setup utilities."""
        logger.info("Testing FederatedQuantumManager Initialization")
        num_manager_clients = 3
        client_ids = [f"managed_client_{i}" for i in range(num_manager_clients)]

        server = FederatedQuantumManager.initialize_quantum_server(
            model_class=VariationalQuantumClassifier,
            model_kwargs=self.vqc_model_kwargs,
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=self.test_dataset,
            server_kwargs={"device": self.device}
        )
        self.assertIsInstance(server, QuantumFederatedServer)

        clients = FederatedQuantumManager.create_quantum_clients(
            client_ids=client_ids,
            model_class=VariationalQuantumClassifier,
            model_kwargs=self.vqc_model_kwargs,
            datasets=self.client_datasets[:num_manager_clients],
            client_kwargs=self.vqc_client_kwargs
        )
        self.assertEqual(len(clients), num_manager_clients)
        self.assertTrue(all(isinstance(c, QuantumFederatedClient) for c in clients))

        for client in clients: server.register_client(client)
        round_info = server.train_round(local_epochs=1)
        self.assertEqual(round_info.get("status"), "success")
        self.assertEqual(len(round_info['participating_clients']), num_manager_clients)
        self.assertIn('accuracy', round_info['evaluation_metrics'])
        logger.info("FederatedQuantumManager Initialization Test Passed.")


if __name__ == '__main__':
    unittest.main()