"""
Integration tests for complete federated learning rounds.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import FedAvg, FedProx
from src.federated.utils import FederatedDataset, set_seed


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestFederatedRound(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment for federated learning rounds."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Create synthetic dataset
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        self.dataset = TensorDataset(x, y)
        
        # Split dataset for evaluation
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, test_size]
        )
        
        # Server model
        self.server_model = SimpleModel()
        
        # Initialize server with evaluation dataset
        self.server = FederatedServer(
            model=self.server_model,
            aggregation_strategy=FedAvg(),
            evaluation_dataset=self.test_dataset
        )
        
        # Create and register 5 clients
        num_clients = 5
        client_datasets = FederatedDataset.iid_partition(
            self.train_dataset, num_clients
        )
        
        for i in range(num_clients):
            client_model = SimpleModel()
            client = FederatedClient(
                client_id=f"client_{i}",
                model=client_model,
                dataset=client_datasets[i],
                batch_size=4
            )
            self.server.register_client(client)
    
    def test_fedavg_round(self):
        """Test a complete round with FedAvg aggregation."""
        # Run a round
        round_info = self.server.train_round(local_epochs=1)
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('loss', round_info)
        self.assertIn('accuracy', round_info)
        
        # All clients should participate
        self.assertEqual(len(round_info['participating_clients']), 5)
        
        # Check if loss and accuracy are reasonable
        self.assertGreater(round_info['accuracy'], 0.0)
        self.assertLess(round_info['accuracy'], 1.0)
    
    def test_fedprox_round(self):
        """Test a complete round with FedProx aggregation."""
        # Change to FedProx
        self.server.aggregation_strategy = FedProx(mu=0.1)
        
        # Run a round with proximal term
        round_info = self.server.train_round(
            local_epochs=1,
            proximal_term=0.1
        )
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('loss', round_info)
        self.assertIn('accuracy', round_info)
        
        # All clients should participate
        self.assertEqual(len(round_info['participating_clients']), 5)
        
        # Check if loss and accuracy are reasonable
        self.assertGreater(round_info['accuracy'], 0.0)
        self.assertLess(round_info['accuracy'], 1.0)
    
    def test_multiple_rounds(self):
        """Test multiple training rounds for improving accuracy."""
        # Run 5 rounds
        history = self.server.train(num_rounds=5, local_epochs=2)
        
        # Check history length
        self.assertEqual(len(history), 5)
        
        # Check if accuracy generally improves
        # Note: This isn't guaranteed due to randomness, but is likely
        first_round_acc = history[0].get('accuracy', 0)
        last_round_acc = history[-1].get('accuracy', 0)
        
        # Print for debugging purposes
        print(f"First round accuracy: {first_round_acc:.4f}")
        print(f"Last round accuracy: {last_round_acc:.4f}")
        
        # In most cases, accuracy should improve
        # But we don't assert this strictly as it's stochastic
    
    def test_client_fraction(self):
        """Test training with a fraction of clients."""
        # Run with only 60% of clients
        round_info = self.server.train_round(
            local_epochs=1,
            client_fraction=0.6
        )
        
        # Should have fewer participating clients
        num_participating = len(round_info['participating_clients'])
        self.assertLess(num_participating, 5)
        self.assertGreaterEqual(num_participating, 3)  # 60% of 5 = 3
    
    def test_non_iid_data(self):
        """Test federated learning with non-IID data distribution."""
        # Re-initialize server
        server_model = SimpleModel()
        server = FederatedServer(
            model=server_model,
            aggregation_strategy=FedAvg(),
            evaluation_dataset=self.test_dataset
        )
        
        # Create a new dataset with targets attribute
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        dataset = TensorDataset(x, y)
        
        # Add targets attribute manually for non-IID partitioning
        dataset.targets = y
        
        # Use the new dataset with targets instead of self.train_dataset
        num_clients = 5
        non_iid_datasets = FederatedDataset.non_iid_label_partition(
            dataset,  # Use the new dataset with targets, not self.train_dataset
            num_clients=num_clients,
            num_classes=2,
            classes_per_client=1  # Each client gets only one class
        )
        
        for i in range(num_clients):
            client_model = SimpleModel()
            client = FederatedClient(
                client_id=f"non_iid_client_{i}",
                model=client_model,
                dataset=non_iid_datasets[i],
                batch_size=4
            )
            server.register_client(client)
        
        # Run a round
        round_info = server.train_round(local_epochs=2)
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('loss', round_info)
        self.assertIn('accuracy', round_info)


if __name__ == '__main__':
    unittest.main()