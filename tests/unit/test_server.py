"""
Unit tests for the FederatedServer class.
"""

import unittest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.federated.aggregation import FedAvg, FedProx
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)


class TestFederatedServer(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a model for the server
        self.model = SimpleModel()
        
        # Initialize server
        self.server = FederatedServer(
            model=self.model,
            aggregation_strategy=FedAvg()
        )
        
        # Create clients
        self.clients = []
        for i in range(3):
            # Create synthetic datasets for clients
            x = torch.randn(20, 10)
            y = torch.randint(0, 2, (20,))
            dataset = TensorDataset(x, y)
            
            # Create client model
            client_model = SimpleModel()
            
            # Create client
            client = FederatedClient(
                client_id=f"client_{i}",
                model=client_model,
                dataset=dataset,
                batch_size=4
            )
            
            self.clients.append(client)
            self.server.register_client(client)
    
    def test_initialization(self):
        """Test server initialization."""
        self.assertIsInstance(self.server.model, nn.Module)
        self.assertIsInstance(self.server.aggregation_strategy, FedAvg)
        self.assertEqual(len(self.server.clients), 3)
        self.assertEqual(len(self.server.round_history), 0)
    
    def test_get_parameters(self):
        """Test getting model parameters."""
        params = self.server.get_parameters()
        
        # Check if all parameters are returned
        for name, _ in self.model.named_parameters():
            self.assertIn(name, params)
        
        # Check if parameters are tensors
        for param in params.values():
            self.assertIsInstance(param, torch.Tensor)
    
    def test_set_parameters(self):
        """Test setting model parameters."""
        # Get original parameters
        original_params = self.server.get_parameters()
        
        # Create modified parameters
        modified_params = {}
        for name, param in original_params.items():
            modified_params[name] = param + 0.1
        
        # Set modified parameters
        self.server.set_parameters(modified_params)
        
        # Check if parameters were updated
        updated_params = self.server.get_parameters()
        for name, param in updated_params.items():
            self.assertTrue(torch.allclose(param, modified_params[name]))
    
    def test_select_clients(self):
        """Test client selection."""
        # Select all clients
        selected = self.server.select_clients(fraction=1.0)
        self.assertEqual(len(selected), 3)
        
        # Select subset of clients
        selected = self.server.select_clients(fraction=0.5, min_clients=1)
        self.assertGreaterEqual(len(selected), 1)
        self.assertLessEqual(len(selected), 3)
        
        # Check if selected clients exist
        for client_id in selected:
            self.assertIn(client_id, self.server.clients)
    
    def test_train_round(self):
        """Test running a federated training round."""
        # Run a training round
        round_info = self.server.train_round(local_epochs=1)
        
        # Check round results
        self.assertIn('participating_clients', round_info)
        self.assertIn('client_samples', round_info)
        self.assertIn('duration_seconds', round_info)
        
        # Should have 3 participating clients
        self.assertEqual(len(round_info['participating_clients']), 3)
        
        # Check if server's round history was updated
        self.assertEqual(len(self.server.round_history), 1)
    
    def test_train(self):
        """Test running multiple training rounds."""
        # Run training for 2 rounds
        history = self.server.train(num_rounds=2, local_epochs=1)
        
        # Check history
        self.assertEqual(len(history), 2)
        
        # Check if round history was updated
        self.assertEqual(len(self.server.round_history), 2)
    
    def test_change_aggregation_strategy(self):
        """Test changing the aggregation strategy."""
        # Create a new aggregation strategy
        new_strategy = FedProx(mu=0.1)
        
        # Replace server's strategy
        self.server.aggregation_strategy = new_strategy
        
        # Verify the change
        self.assertIsInstance(self.server.aggregation_strategy, FedProx)
        
        # Run a round to ensure it still works
        round_info = self.server.train_round(local_epochs=1)
        self.assertIn('participating_clients', round_info)


if __name__ == '__main__':
    unittest.main()