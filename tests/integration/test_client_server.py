"""
Integration tests for client-server communication and interaction.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.federated.aggregation import FedAvg


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)


class TestClientServerIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up the test environment with server and clients."""
        # Server model
        self.server_model = SimpleModel()
        
        # Server
        self.server = FederatedServer(
            model=self.server_model,
            aggregation_strategy=FedAvg()
        )
        
        # Create 3 clients with different data
        for i in range(3):
            # Create synthetic dataset
            x = torch.randn(20, 10)  # 20 samples, 10 features
            y = torch.randint(0, 2, (20,))  # Binary classification
            dataset = TensorDataset(x, y)
            
            # Create client model - same architecture but different initialization
            client_model = SimpleModel()
            
            # Create client
            client = FederatedClient(
                client_id=f"client_{i}",
                model=client_model,
                dataset=dataset
            )
            
            # Register client with server
            self.server.register_client(client)
    
    def test_end_to_end_communication(self):
        """Test the complete client-server communication cycle."""
        # Get initial server parameters
        initial_params = self.server.get_parameters()
        
        # Select clients for training
        client_ids = self.server.select_clients(fraction=1.0)
        
        # Distribute parameters to clients
        for client_id in client_ids:
            client = self.server.clients[client_id]
            client.set_parameters(initial_params)
            
            # Check if parameters were correctly set
            client_params = client.get_parameters()
            for name, param in initial_params.items():
                self.assertTrue(torch.allclose(param, client_params[name]))
        
        # Clients train locally
        client_updates = []
        for client_id in client_ids:
            client = self.server.clients[client_id]
            updated_params, samples = client.train(epochs=1)
            client_updates.append((updated_params, samples))
            
            # Check if training changed parameters
            for name, param in updated_params.items():
                self.assertFalse(torch.allclose(param, initial_params[name]))
        
        # Aggregate updates
        aggregated_params = self.server.aggregation_strategy.aggregate(client_updates)
        
        # Update server model
        self.server.set_parameters(aggregated_params)
        
        # Check if server parameters changed
        updated_server_params = self.server.get_parameters()
        for name, param in initial_params.items():
            self.assertFalse(torch.allclose(param, updated_server_params[name]))
    
    def test_complete_round_execution(self):
        """Test the execution of a complete federated round through the server."""
        # Get initial server parameters
        initial_params = self.server.get_parameters()
        
        # Execute a training round
        round_info = self.server.train_round(local_epochs=1)
        
        # Check if the round completed successfully
        self.assertIn('participating_clients', round_info)
        self.assertGreater(len(round_info['participating_clients']), 0)
        
        # Check if server parameters changed
        updated_server_params = self.server.get_parameters()
        for name, param in initial_params.items():
            self.assertFalse(torch.allclose(param, updated_server_params[name]))
        
        # Check if round history was updated
        self.assertEqual(len(self.server.round_history), 1)
        
    def test_client_server_synchronization(self):
        """Test that clients stay synchronized with the server after multiple rounds."""
        # Run 3 rounds of training
        self.server.train(num_rounds=3, local_epochs=1)
        
        # Get final server parameters
        server_params = self.server.get_parameters()
        
        # Check that all clients can be synchronized with the server
        for client_id, client in self.server.clients.items():
            # Set client parameters to server parameters
            client.set_parameters(server_params)
            
            # Verify parameters are identical
            client_params = client.get_parameters()
            for name, param in server_params.items():
                self.assertTrue(torch.allclose(param, client_params[name]))
                
        # Round history should have 3 entries
        self.assertEqual(len(self.server.round_history), 3)


if __name__ == '__main__':
    unittest.main()