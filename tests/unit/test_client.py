# --- tests/unit/test_client.py ---

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
import logging # Import logging

# Configure logging for tests (optional, can be configured globally)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


# Assume SimpleModel is defined here or imported correctly
class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

# Assume FederatedClient is imported correctly
from src.federated.client import FederatedClient

class TestFederatedClient(unittest.TestCase):

    def setUp(self):
        """Set up test environment before each test."""
        logger.info(f"\n--- Setting up test: {self.id()} ---")
        # Create a small synthetic dataset
        x = torch.randn(20, 10)
        y = torch.randint(0, 2, (20,))
        self.dataset = TensorDataset(x, y)

        # Initialize model
        self.model = SimpleModel()

        # Initialize client (without device specified, will auto-detect)
        self.client = FederatedClient(
            client_id="test_client",
            model=self.model,
            dataset=self.dataset,
            batch_size=4,
            learning_rate=0.01
            # Using default optimizer (SGD) and loss (CrossEntropyLoss)
        )

    def tearDown(self):
        logger.info(f"--- Tearing down test: {self.id()} ---")


    def test_initialization(self):
        """Test client initialization."""
        logger.info("Testing client initialization.")
        self.assertEqual(self.client.client_id, "test_client")
        self.assertEqual(self.client.batch_size, 4)
        self.assertEqual(self.client.learning_rate, 0.01)
        self.assertIsInstance(self.client.optimizer, torch.optim.Optimizer)
        self.assertIsInstance(self.client.loss_fn, nn.Module)
        self.assertIsNotNone(self.client.device) # Check device was set

    def test_get_parameters(self):
        """Test getting model parameters."""
        logger.info("Testing get_parameters.")
        params = self.client.get_parameters()
        self.assertIsInstance(params, dict)
        # Check if all parameters are returned (using state_dict keys)
        model_state_dict_keys = self.model.state_dict().keys()
        self.assertSetEqual(set(params.keys()), set(model_state_dict_keys))
        # Check if parameters are tensors on CPU
        for param in params.values():
            self.assertIsInstance(param, torch.Tensor)
            self.assertEqual(param.device.type, 'cpu')

    def test_set_parameters(self):
        """Test setting model parameters."""
        logger.info("Testing set_parameters.")
        # Get original parameters state_dict
        original_params_dict = self.client.get_parameters() # Already on CPU

        # Create modified parameters (state_dict)
        modified_params = {}
        for name, param in original_params_dict.items():
            modified_params[name] = param + 0.1 # Modify on CPU

        # Set modified parameters
        self.client.set_parameters(modified_params)

        # Check if parameters were updated in the model
        updated_params_dict = self.client.get_parameters() # Get updated params on CPU
        for name, updated_param in updated_params_dict.items():
            self.assertTrue(torch.allclose(updated_param, modified_params[name]), f"Parameter {name} mismatch.")
            # Also check they are different from original
            self.assertFalse(torch.allclose(updated_param, original_params_dict[name]), f"Parameter {name} did not change.")

    def test_train(self):
        """Test client training without FedProx."""
        logger.info("Testing train method (no FedProx).")
        # Get parameters before training
        params_before = self.client.get_parameters()

        # Train the model
        updated_params, samples_processed = self.client.train(epochs=2)

        # Check return types and values
        self.assertIsInstance(updated_params, dict)
        # Samples processed = epochs * dataset_size
        self.assertEqual(samples_processed, len(self.dataset) * 2)

        # Check if parameters changed during training
        self.assertNotEqual(params_before.keys(), set()) # Ensure params_before is not empty
        for name, param_before in params_before.items():
            self.assertIn(name, updated_params, f"Parameter {name} missing after training.")
            # Parameters should have changed after training
            self.assertFalse(torch.allclose(param_before, updated_params[name]), f"Parameter {name} did not change.")

    def test_evaluate(self):
        """Test client evaluation."""
        logger.info("Testing evaluate method.")
        # Ensure model is on correct device before evaluation if needed
        # self.client.model.to(self.client.device) # Should already be handled by init

        metrics = self.client.evaluate()

        # Check basic metrics
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIsInstance(metrics['loss'], float)
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)

    def test_fedprox_training(self):
        """Test training with FedProx proximal term."""
        logger.info("Testing train method (with FedProx).")
        # Create a separate global model state_dict
        global_model_state = SimpleModel().state_dict()

        # Make global params different from client's initial params
        client_initial_params = self.client.get_parameters()
        global_params_dict = {}
        for name, param in global_model_state.items():
             # Ensure they are different for the test
             global_params_dict[name] = client_initial_params[name] + 0.5

        # Train with proximal term, passing the state dictionary
        # *** CORRECTED CALL: Use global_params keyword ***
        updated_params, samples_processed = self.client.train(
            epochs=1,
            proximal_term=0.1,
            global_params=global_params_dict # Pass the state dict
        )

        # Basic checks: ensure it ran and returned correctly
        self.assertIsInstance(updated_params, dict)
        self.assertEqual(samples_processed, len(self.dataset) * 1)
        # Check parameters changed from initial client state
        self.assertNotEqual(client_initial_params.keys(), set())
        for name, param_initial in client_initial_params.items():
             self.assertIn(name, updated_params)
             self.assertFalse(torch.allclose(param_initial, updated_params[name]),
                              f"FedProx: Parameter {name} did not change.")


if __name__ == '__main__':
    unittest.main()