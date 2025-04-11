"""
Integration tests for complete quantum federated learning rounds.
"""

import unittest
import torch
import numpy as np
from torch.utils.data import TensorDataset

from src.federated.aggregation import FedAvg, FedProx
from src.federated.utils import set_seed
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.core.quantum_client import QuantumFederatedClient
from src.core.quantum_manager import QuantumFederatedServer, QuantumAggregationStrategy, FederatedQuantumManager


class TestQuantumFederatedRound(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment for quantum federated learning rounds."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create synthetic dataset
        n_samples = 100
        n_features = 4  # Small number of features for faster testing
        
        # Generate random data with a simple pattern
        X = np.random.uniform(-1, 1, size=(n_samples, n_features))
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)  # Simple binary pattern
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create dataset
        self.dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split for train/test
        train_size = int(0.8 * n_samples)
        test_size = n_samples - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        # Initialize model configurations
        self.n_qubits = n_features
        self.n_layers = 2
        self.vqc_config = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_classes": 2,
            "circuit_type": "basic",
            "encoding_type": "angle",
            "shots": 1000  # Use higher shots for more stable results
        }
        
        # Initialize server with VQC model
        self.vqc_model = VariationalQuantumClassifier(**self.vqc_config)
        self.vqc_server = QuantumFederatedServer(
            model=self.vqc_model,
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=self.test_dataset
        )
        
        # Prepare client datasets
        num_clients = 5
        client_indices = np.array_split(np.arange(train_size), num_clients)
        self.client_datasets = [
            torch.utils.data.Subset(self.train_dataset, indices) 
            for indices in client_indices
        ]
        
        # Create clients for VQC
        for i in range(num_clients):
            client_model = VariationalQuantumClassifier(**self.vqc_config)
            client = QuantumFederatedClient(
                client_id=f"vqc_client_{i}",
                model=client_model,
                dataset=self.client_datasets[i],
                batch_size=4,
                learning_rate=0.1
            )
            self.vqc_server.register_client(client)
    
    def test_vqc_fedavg_round(self):
        """Test a complete round with VQC models and FedAvg aggregation."""
        # Run a round
        round_info = self.vqc_server.train_round(local_epochs=1)
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('accuracy', round_info)
        
        # All clients should participate
        self.assertEqual(len(round_info['participating_clients']), 5)
        
        # Check if accuracy is reasonable
        self.assertGreaterEqual(round_info['accuracy'], 0.0)
        self.assertLessEqual(round_info['accuracy'], 1.0)
    
    def test_vqc_fedprox_round(self):
        """Test a complete round with VQC models and FedProx aggregation."""
        # Change to FedProx
        self.vqc_server.aggregation_strategy = QuantumAggregationStrategy(FedProx(mu=0.1))
        
        # Run a round with proximal term
        round_info = self.vqc_server.train_round(
            local_epochs=1,
            proximal_term=0.1
        )
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('accuracy', round_info)
        
        # All clients should participate
        self.assertEqual(len(round_info['participating_clients']), 5)
        
        # Check if accuracy is reasonable
        self.assertGreaterEqual(round_info['accuracy'], 0.0)
        self.assertLessEqual(round_info['accuracy'], 1.0)
    
    def test_vqc_multiple_rounds(self):
        """Test multiple training rounds for improving accuracy with VQC models."""
        # Run 3 rounds (fewer than classical due to computational constraints)
        history = self.vqc_server.train(num_rounds=3, local_epochs=1)
        
        # Check history length
        self.assertEqual(len(history), 3)
        
        # Check if accuracy is tracked
        for round_data in history:
            self.assertIn('accuracy', round_data)
        
        # Print for debugging purposes
        print(f"First round accuracy: {history[0].get('accuracy', 0):.4f}")
        print(f"Last round accuracy: {history[-1].get('accuracy', 0):.4f}")
    
    def test_vqc_client_fraction(self):
        """Test training with a fraction of clients using VQC models."""
        # Run with only 60% of clients
        round_info = self.vqc_server.train_round(
            local_epochs=1,
            client_fraction=0.6
        )
        
        # Should have fewer participating clients
        num_participating = len(round_info['participating_clients'])
        self.assertLess(num_participating, 5)
        self.assertGreaterEqual(num_participating, 3)  # 60% of 5 = 3
    
    def test_qnn_federated_round(self):
        """Test a complete round with QNN models."""
        # Create a new server with QNN model
        n_qubits = self.n_qubits
        n_features = n_qubits
        
        # QNN configuration
        qnn_model = QuantumNeuralNetwork(
            n_qubits=n_qubits,
            n_layers=1,
            input_size=n_features,
            output_size=2,
            encoding_type="angle"
        )
        
        qnn_server = QuantumFederatedServer(
            model=qnn_model,
            aggregation_strategy=FedAvg(),
            evaluation_dataset=self.test_dataset
        )
        
        # Create and register QNN clients
        num_clients = 3  # Fewer clients for QNN due to computational constraints
        for i in range(num_clients):
            client_model = QuantumNeuralNetwork(
                n_qubits=n_qubits,
                n_layers=1,
                input_size=n_features,
                output_size=2,
                encoding_type="angle"
            )
            client = QuantumFederatedClient(
                client_id=f"qnn_client_{i}",
                model=client_model,
                dataset=self.client_datasets[i],
                batch_size=4,
                learning_rate=0.05,
                optimizer_class=torch.optim.Adam
            )
            qnn_server.register_client(client)
        
        # Run a round
        round_info = qnn_server.train_round(local_epochs=1)
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('accuracy', round_info)
        
        # All clients should participate
        self.assertEqual(len(round_info['participating_clients']), num_clients)
        
        # Check if accuracy is reasonable
        self.assertGreaterEqual(round_info['accuracy'], 0.0)
        self.assertLessEqual(round_info['accuracy'], 1.0)
    
    def test_hybrid_model_quantum_learning(self):
        """Test federated learning with hybrid quantum-classical models."""
        # Initialize a hybrid model
        hybrid_model = HybridQuantumModel(
            n_qubits=self.n_qubits,
            n_layers=1,
            classical_sizes=[8, 4],
            output_size=2,
            encoding_type="angle"
        )
        
        # Create server with hybrid model
        hybrid_server = QuantumFederatedServer(
            model=hybrid_model,
            aggregation_strategy=FedAvg(),
            evaluation_dataset=self.test_dataset
        )
        
        # Create and register hybrid model clients
        num_clients = 3
        for i in range(num_clients):
            client_model = HybridQuantumModel(
                n_qubits=self.n_qubits,
                n_layers=1,
                classical_sizes=[8, 4],
                output_size=2,
                encoding_type="angle"
            )
            client = QuantumFederatedClient(
                client_id=f"hybrid_client_{i}",
                model=client_model,
                dataset=self.client_datasets[i],
                batch_size=4,
                learning_rate=0.01,
                optimizer_class=torch.optim.Adam
            )
            hybrid_server.register_client(client)
        
        # Run a round
        round_info = hybrid_server.train_round(local_epochs=1)
        
        # Check results
        self.assertEqual(len(round_info['participating_clients']), num_clients)
        self.assertIn('accuracy', round_info)
    
    def test_quantum_manager_initialization(self):
        """Test the FederatedQuantumManager initialization utilities."""
        # Create a fresh set of datasets for this test
        num_clients = 3
        client_ids = [f"managed_client_{i}" for i in range(num_clients)]
        
        # Initialize using the manager
        server = FederatedQuantumManager.initialize_quantum_server(
            model_class=VariationalQuantumClassifier,
            model_kwargs=self.vqc_config,
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=self.test_dataset
        )
        
        # Create clients using the manager
        clients = FederatedQuantumManager.create_quantum_clients(
            client_ids=client_ids,
            model_class=VariationalQuantumClassifier,
            model_kwargs=self.vqc_config,
            datasets=self.client_datasets[:num_clients],
            client_kwargs={"batch_size": 4, "learning_rate": 0.1}
        )
        
        # Register clients
        for client in clients:
            server.register_client(client)
        
        # Test a round
        round_info = server.train_round(local_epochs=1)
        
        # Verify the round completed successfully
        self.assertEqual(len(round_info['participating_clients']), num_clients)
        self.assertIn('accuracy', round_info)


if __name__ == '__main__':
    unittest.main()