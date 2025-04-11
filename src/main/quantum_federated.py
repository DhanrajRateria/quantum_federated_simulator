"""
Demonstration of federated learning with quantum models.
"""

import os
import logging
import numpy as np
import torch
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

from federated.aggregation import FedAvg
from federated.utils import setup_logger, set_seed, FederatedDataset
from quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from quantum_client import QuantumFederatedClient
from quantum_manager import FederatedQuantumManager, QuantumAggregationStrategy


# Setup logging
logger = setup_logger(name="quantum_federated", level=logging.INFO)


def prepare_dataset():
    """Prepare a simple dataset for quantum ML."""
    # Load a small dataset (Iris or Breast Cancer)
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    return train_dataset, test_dataset, X.shape[1]


def run_federated_quantum():
    """Run federated learning with quantum models."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Prepare datasets
    train_dataset, test_dataset, n_features = prepare_dataset()
    
    # Create client datasets (IID partition for simplicity)
    num_clients = 3
    client_datasets = FederatedDataset.iid_partition(train_dataset, num_clients)
    
    # Model configuration
    n_qubits = min(10, n_features)  # Use at most 10 qubits
    model_kwargs = {
        "n_qubits": n_qubits,
        "n_layers": 2,
        "n_classes": 2,  # Binary classification
        "circuit_type": "basic",
        "encoding_type": "angle"
    }
    
    # Create server with global quantum model
    quantum_server = FederatedQuantumManager.initialize_quantum_server(
        model_class=VariationalQuantumClassifier,
        model_kwargs=model_kwargs,
        aggregation_strategy=QuantumAggregationStrategy(FedAvg()),
        evaluation_dataset=test_dataset
    )
    
    # Create and register quantum clients
    client_kwargs = {
        "batch_size": 16,
        "learning_rate": 0.05
    }
    
    client_ids = [f"quantum_client_{i}" for i in range(num_clients)]
    quantum_clients = FederatedQuantumManager.create_quantum_clients(
        client_ids=client_ids,
        model_class=VariationalQuantumClassifier,
        model_kwargs=model_kwargs,
        datasets=client_datasets,
        client_kwargs=client_kwargs
    )
    
    # Register clients with the server
    for client in quantum_clients:
        quantum_server.register_client(client)
    
    # Run federated training
    num_rounds = 5
    logger.info(f"Starting federated quantum training for {num_rounds} rounds with {num_clients} clients")
    
    results = quantum_server.train(
        num_rounds=num_rounds,
        local_epochs=1,
        client_fraction=1.0  # Use all clients
    )
    
    # Print results
    for round_idx, round_metrics in enumerate(results):
        if 'accuracy' in round_metrics:
            logger.info(f"Round {round_idx+1}: Accuracy = {round_metrics['accuracy']:.4f}")
    
    # Final evaluation
    final_metrics = quantum_server.evaluate()
    logger.info(f"Final evaluation metrics: {final_metrics}")


if __name__ == "__main__":
    run_federated_quantum()