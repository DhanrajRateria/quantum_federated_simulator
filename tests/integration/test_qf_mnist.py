"""
Integration tests for quantum federated learning with MNIST dataset.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import numpy as np

from src.federated.aggregation import FedAvg
from src.federated.utils import FederatedDataset, set_seed
from src.quantum.models import HybridQuantumModel, VariationalQuantumClassifier
from src.core.quantum_client import QuantumFederatedClient
from src.core.quantum_manager import (
    QuantumFederatedServer,
    QuantumAggregationStrategy,
    FederatedQuantumManager
)


class FeatureExtractor(nn.Module):
    """CNN feature extractor to reduce MNIST dimensionality for quantum processing."""
    def __init__(self, output_dim=8):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, output_dim)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        return x


class HybridMNISTModel(nn.Module):
    """Hybrid classical-quantum model for MNIST classification."""
    def __init__(self, n_qubits, n_classes=10):
        super().__init__()
        # Classical feature extraction
        self.feature_extractor = FeatureExtractor(output_dim=n_qubits)
        
        # Quantum part (initialized in the client because it needs special handling)
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        
        # Final classification layer
        self.classifier = nn.Linear(n_qubits, n_classes)
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        # Classification
        output = self.classifier(features)
        return output


class TestQuantumFederatedMNIST(unittest.TestCase):
    """Test quantum federated learning with MNIST dataset."""
    
    def setUp(self):
        """Set up test environment for quantum federated learning with MNIST."""
        # Set random seed for reproducibility
        set_seed(42)
        
        # Load a subset of MNIST for faster testing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Download MNIST dataset
        self.full_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        
        # Use a small subset for testing (500 samples)
        subset_indices = torch.randperm(len(self.full_dataset))[:500]
        self.dataset = torch.utils.data.Subset(self.full_dataset, subset_indices)
        
        # Split for train/test
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        self.train_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, test_size]
        )
        
        # Number of qubits to use (reduced dimension for quantum processing)
        self.n_qubits = 8
        self.n_classes = 10  # MNIST has 10 classes (digits 0-9)
        
        # Initialize server with quantum aggregation
        self.server = FederatedQuantumManager.initialize_quantum_server(
            model_class=HybridQuantumModel,
            model_kwargs={
                "n_qubits": self.n_qubits,
                "n_layers": 2,
                "n_classes": self.n_classes,
                "circuit_type": "basic",
                "feature_extractor": FeatureExtractor(output_dim=self.n_qubits)
            },
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=self.test_dataset
        )
        
        # Create client datasets (IID partition for simplicity)
        n_clients = 3  # Use 3 clients for faster testing
        self.client_datasets = FederatedDataset.iid_partition(
            self.train_dataset, n_clients
        )
    
    def test_mnist_quantum_client_creation(self):
        """Test creating quantum federated clients for MNIST."""
        # Create a client
        client = QuantumFederatedClient(
            client_id="mnist_quantum_client_1",
            model=HybridQuantumModel(
                n_qubits=self.n_qubits,
                n_layers=2,
                n_classes=self.n_classes,
                circuit_type="basic",
                feature_extractor=FeatureExtractor(output_dim=self.n_qubits)
            ),
            dataset=self.client_datasets[0],
            batch_size=16,
            learning_rate=0.01
        )
        
        # Verify client attributes
        self.assertEqual(client.client_id, "mnist_quantum_client_1")
        self.assertEqual(client.model_type, "hybrid")
        
        # Test parameter access
        params = client.get_parameters()
        self.assertIsInstance(params, dict)
        # Check that we have parameters for feature extractor and classifier
        self.assertTrue(any("feature_extractor" in name for name in params.keys()))
        self.assertTrue(any("classifier" in name for name in params.keys()))
    
    def test_mnist_quantum_client_training(self):
        """Test training a quantum client on MNIST data."""
        # Create a client
        client = QuantumFederatedClient(
            client_id="mnist_quantum_client_2",
            model=HybridQuantumModel(
                n_qubits=self.n_qubits,
                n_layers=2,
                n_classes=self.n_classes,
                circuit_type="basic",
                feature_extractor=FeatureExtractor(output_dim=self.n_qubits)
            ),
            dataset=self.client_datasets[0],
            batch_size=16,
            learning_rate=0.01
        )
        
        # Get initial parameters
        initial_params = {name: param.clone() for name, param in client.get_parameters().items()}
        
        # Train for one epoch
        updated_params, num_samples = client.train(epochs=1)
        
        # Verify parameters changed
        self.assertFalse(all(
            torch.allclose(initial_params[name], updated_params[name])
            for name in initial_params if "weight" in name  # Check weight parameters
        ))
        
        # Verify number of samples
        self.assertEqual(num_samples, len(self.client_datasets[0]))
    
    def test_federated_mnist_quantum_round(self):
        """Test a complete federated quantum round on MNIST data."""
        # Create and register clients using the FederatedQuantumManager
        clients = FederatedQuantumManager.create_quantum_clients(
            client_ids=["mnist_quantum_client_1", "mnist_quantum_client_2", "mnist_quantum_client_3"],
            model_class=HybridQuantumModel,
            model_kwargs={
                "n_qubits": self.n_qubits,
                "n_layers": 2,
                "n_classes": self.n_classes,
                "circuit_type": "basic",
                "feature_extractor": FeatureExtractor(output_dim=self.n_qubits)
            },
            datasets=self.client_datasets,
            client_kwargs={
                "batch_size": 16,
                "learning_rate": 0.01
            }
        )
        
        # Register clients with the server
        for client in clients:
            self.server.register_client(client)
        
        # Initial global parameters
        initial_params = {name: param.clone() for name, param in self.server.get_parameters().items()}
        
        # Run a training round
        round_info = self.server.train_round(local_epochs=1)
        
        # Verify round completed
        self.assertIn('participating_clients', round_info)
        self.assertEqual(len(round_info['participating_clients']), 3)
        
        # Verify parameters changed
        updated_params = self.server.get_parameters()
        self.assertFalse(all(
            torch.allclose(initial_params[name], updated_params[name])
            for name in initial_params if "weight" in name  # Check weight parameters
        ))
    
    def test_full_mnist_quantum_training(self):
        """Test multiple rounds of federated learning with quantum models on MNIST."""
        # Create and register clients
        clients = FederatedQuantumManager.create_quantum_clients(
            client_ids=["mnist_quantum_client_1", "mnist_quantum_client_2", "mnist_quantum_client_3"],
            model_class=HybridQuantumModel,
            model_kwargs={
                "n_qubits": self.n_qubits,
                "n_layers": 2,
                "n_classes": self.n_classes,
                "circuit_type": "basic",
                "feature_extractor": FeatureExtractor(output_dim=self.n_qubits)
            },
            datasets=self.client_datasets,
            client_kwargs={
                "batch_size": 16,
                "learning_rate": 0.01
            }
        )
        
        # Register clients with the server
        for client in clients:
            self.server.register_client(client)
        
        # Run multiple training rounds
        num_rounds = 3  # Use a small number for testing
        results = self.server.train(
            num_rounds=num_rounds,
            local_epochs=1,
            client_fraction=1.0  # Use all clients
        )
        
        # Verify we have results for each round
        self.assertEqual(len(results), num_rounds)
        
        # Check that accuracy is reported
        for round_result in results:
            self.assertIn('accuracy', round_result)
        
        # Final evaluation
        final_metrics = self.server.evaluate()
        self.assertIn('accuracy', final_metrics)
        
        # Print metrics for inspection
        print(f"Final accuracy: {final_metrics['accuracy']:.4f}")
    
    def test_vqc_mnist_dimensionality_reduction(self):
        """Test VQC with preprocessing for dimensionality reduction on MNIST."""
        # For VQC models, we need to preprocess the data to reduce dimensionality
        
        # Define a function to extract features using our CNN
        feature_extractor = FeatureExtractor(output_dim=self.n_qubits)
        
        # Create preprocessed dataset
        def preprocess_dataset(dataset):
            # Extract features and convert to flat dataset
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            features_list = []
            labels_list = []
            
            with torch.no_grad():
                for data, target in dataloader:
                    # Extract features
                    features = feature_extractor(data)
                    # Normalize
                    features = F.normalize(features, p=2, dim=1)
                    
                    features_list.append(features)
                    labels_list.append(target)
            
            # Concatenate all batches
            features_tensor = torch.cat(features_list, dim=0)
            labels_tensor = torch.cat(labels_list, dim=0)
            
            return TensorDataset(features_tensor, labels_tensor)
        
        # Preprocess train and test datasets
        preprocessed_train = preprocess_dataset(self.train_dataset)
        preprocessed_test = preprocess_dataset(self.test_dataset)
        
        # Partition the preprocessed training data
        n_clients = 3
        preprocessed_client_datasets = FederatedDataset.iid_partition(
            preprocessed_train, n_clients
        )
        
        # Initialize a server with VQC model
        vqc_server = FederatedQuantumManager.initialize_quantum_server(
            model_class=VariationalQuantumClassifier,
            model_kwargs={
                "n_qubits": self.n_qubits,
                "n_layers": 2,
                "n_classes": self.n_classes,
                "circuit_type": "basic",
                "encoding_type": "angle"
            },
            aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
            evaluation_dataset=preprocessed_test
        )
        
        # Create and register VQC clients
        vqc_clients = FederatedQuantumManager.create_quantum_clients(
            client_ids=["vqc_mnist_client_1", "vqc_mnist_client_2", "vqc_mnist_client_3"],
            model_class=VariationalQuantumClassifier,
            model_kwargs={
                "n_qubits": self.n_qubits,
                "n_layers": 2,
                "n_classes": self.n_classes,
                "circuit_type": "basic", 
                "encoding_type": "angle"
            },
            datasets=preprocessed_client_datasets,
            client_kwargs={
                "batch_size": 16,
                "learning_rate": 0.05
            }
        )
        
        # Register clients
        for client in vqc_clients:
            vqc_server.register_client(client)
        
        # Run a single training round
        round_info = vqc_server.train_round(local_epochs=1)
        
        # Verify round completed
        self.assertIn('participating_clients', round_info)
        self.assertEqual(len(round_info['participating_clients']), 3)
        
        # Check that metrics are reported
        self.assertIn('accuracy', round_info)


if __name__ == "__main__":
    unittest.main()