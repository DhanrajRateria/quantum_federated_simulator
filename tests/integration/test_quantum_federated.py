"""
Integration tests for quantum federated learning.
"""

import os
import sys
import unittest
import numpy as np
import torch
from torch.utils.data import TensorDataset

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.federated.server import FederatedServer
from src.federated.aggregation import FedAvg
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork
from src.main.quantum_client import QuantumFederatedClient
from src.main.quantum_manager import QuantumAggregationStrategy
from src.quantum.utils import set_random_seed


class TestQuantumFederated(unittest.TestCase):
    """Test quantum federated learning integration."""
    
    def setUp(self):
        """Set up test environment for quantum federated learning."""
        # Set random seed for reproducibility
        set_random_seed(42)
        
        # Create synthetic dataset for binary classification
        n_samples = 50
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
        
        # Model configuration
        self.n_qubits = n_features
        self.n_layers = 1  # Use minimal layers for faster testing
        
        # Create global model
        self.global_model = VariationalQuantumClassifier(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_classes=2,
            circuit_type='basic',
            encoding_type='angle',
            shots=100  # Use small number of shots for faster testing
        )
        
        # Initialize server with quantum aggregation
        self.server = FederatedServer(
            model=self.global_model,
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=self.test_dataset
        )
        
        # Create client datasets
        n_clients = 2
        client_indices = np.array_split(np.arange(train_size), n_clients)
        self.client_datasets = [
            torch.utils.data.Subset(self.train_dataset, indices) 
            for indices in client_indices
        ]
        
    def test_quantum_client_creation(self):
        """Test creating quantum federated clients."""
        # Create a client
        client = QuantumFederatedClient(
            client_id="quantum_client_1",
            model=VariationalQuantumClassifier(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=2,
                circuit_type='basic',
                encoding_type='angle'
            ),
            dataset=self.client_datasets[0],
            batch_size=10,
            learning_rate=0.1
        )
        
        # Verify client attributes
        self.assertEqual(client.client_id, "quantum_client_1")
        self.assertEqual(client.model_type, "vqc")
        
        # Test parameter access
        params = client.get_parameters()
        self.assertIn("quantum_params", params)
        self.assertEqual(params["quantum_params"].shape[0], 
                        self.n_qubits * 3 * self.n_layers)  # Should match expected param count
    
    def test_quantum_client_training(self):
        """Test training a quantum client."""
        # Create a client
        client = QuantumFederatedClient(
            client_id="quantum_client_2",
            model=VariationalQuantumClassifier(
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_classes=2,
                circuit_type='basic',
                encoding_type='angle'
            ),
            dataset=self.client_datasets[0],
            batch_size=10,
            learning_rate=0.1
        )
        
        # Get initial parameters
        initial_params = client.get_parameters()["quantum_params"].clone()
        
        # Train for one epoch
        updated_params, num_samples = client.train(epochs=1)
        
        # Verify parameters changed
        self.assertFalse(torch.allclose(
            initial_params, 
            updated_params["quantum_params"]
        ))
        
        # Verify number of samples
        self.assertEqual(num_samples, len(self.client_datasets[0]))
    
    def test_federated_quantum_round(self):
        """Test a complete federated quantum round."""
        # Create and register clients
        for i in range(2):
            client = QuantumFederatedClient(
                client_id=f"quantum_client_{i}",
                model=VariationalQuantumClassifier(
                    n_qubits=self.n_qubits,
                    n_layers=self.n_layers,
                    n_classes=2,
                    circuit_type='basic',
                    encoding_type='angle'
                ),
                dataset=self.client_datasets[i],
                batch_size=10,
                learning_rate=0.1
            )
            self.server.register_client(client)
        
        # Initial global parameters
        initial_params = self.server.get_parameters()
        
        # Run a training round
        round_info = self.server.train_round(local_epochs=1)
        
        # Verify round completed
        self.assertIn('participating_clients', round_info)
        self.assertEqual(len(round_info['participating_clients']), 2)
        
        # Verify parameters changed
        updated_params = self.server.get_parameters()
        self.assertFalse(torch.allclose(
            initial_params["quantum_params"], 
            updated_params["quantum_params"]
        ))


if __name__ == "__main__":
    unittest.main()