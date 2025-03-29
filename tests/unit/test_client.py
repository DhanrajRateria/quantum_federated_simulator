"""
Unit tests for the FederatedClient class.
"""

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

from src.federated.client import FederatedClient


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.fc(x)


class TestFederatedClient(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a small synthetic dataset
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(x, y)
        
        # Initialize model
        self.model = SimpleModel()
        
        # Initialize client
        self.client = FederatedClient(
            client_id="test_client",
            model=self.model,
            dataset=self.dataset,
            batch_size=4,
            learning_rate=0.01
        )
    
    def test_initialization(self):
        """Test client initialization."""
        self.assertEqual(self.client.client_id, "test_client")
        self.assertEqual(self.client.batch_size, 4)
        self.assertEqual(self.client.learning_rate, 0.01)
        self.assertIsInstance(self.client.optimizer, torch.optim.Optimizer)
        self.assertIsInstance(self.client.loss_fn, nn.Module)
    
    def test_get_parameters(self):
        """Test getting model parameters."""
        params = self.client.get_parameters()
        
        # Check if all parameters are returned
        for name, _ in self.model.named_parameters():
            self.assertIn(name, params)
        
        # Check if parameters are tensors
        for param in params.values():
            self.assertIsInstance(param, torch.Tensor)
    
    def test_set_parameters(self):
        """Test setting model parameters."""
        # Get original parameters
        original_params = self.client.get_parameters()
        
        # Create modified parameters
        modified_params = {}
        for name, param in original_params.items():
            modified_params[name] = param + 0.1
        
        # Set modified parameters
        self.client.set_parameters(modified_params)
        
        # Check if parameters were updated
        updated_params = self.client.get_parameters()
        for name, param in updated_params.items():
            self.assertTrue(torch.allclose(param, modified_params[name]))
    
    def test_train(self):
        """Test client training."""
        # Get parameters before training
        params_before = self.client.get_parameters()
        
        # Train the model
        updated_params, samples_used = self.client.train(epochs=2)
        
        # Check if training returned proper values
        self.assertEqual(samples_used, len(self.dataset) * 2)  # 2 epochs
        self.assertIsInstance(updated_params, dict)
        
        # Check if parameters changed during training
        for name, param in params_before.items():
            # Parameters should have changed after training
            self.assertFalse(torch.allclose(param, updated_params[name]))
    
    def test_evaluate(self):
        """Test client evaluation."""
        metrics = self.client.evaluate()
        
        # Check basic metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['accuracy'], float)
    
    def test_fedprox_training(self):
        """Test training with FedProx proximal term."""
        # Create a global model
        global_model = SimpleModel()
        
        # Make it different from client model
        for param in global_model.parameters():
            param.data = param.data + 0.5
            
        # Train with proximal term
        updated_params, _ = self.client.train(
            epochs=1, 
            proximal_term=0.1, 
            global_model=global_model
        )
        
        # We can't easily test the exact effect, but we can check it runs
        self.assertIsInstance(updated_params, dict)


if __name__ == '__main__':
    unittest.main()