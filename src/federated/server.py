"""
Federated learning server implementation.

This module provides the server-side functionality for federated learning,
coordinating client updates and maintaining the global model.
"""

import logging
import copy
import time
from typing import Dict, List, Optional, Tuple, Callable, Any, Union

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from federated.aggregation import AggregationStrategy, FedAvg
from federated.client import FederatedClient

logger = logging.getLogger(__name__)


class FederatedServer:
    """
    Server in a federated learning system.
    
    Responsible for:
    - Coordinating the federated learning process
    - Aggregating model updates from clients
    - Maintaining the global model
    - Evaluating the global model
    """
    
    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        evaluation_dataset: Optional[Dataset] = None,
        evaluation_metric: Optional[Callable] = None,
    ):
        """
        Initialize a federated learning server.
        
        Args:
            model: PyTorch model to be trained (global model)
            aggregation_strategy: Strategy for aggregating client updates
            evaluation_dataset: Optional dataset for server-side evaluation
            evaluation_metric: Optional metric function for evaluation
        """
        self.model = model
        self.aggregation_strategy = aggregation_strategy or FedAvg()
        self.clients: Dict[str, FederatedClient] = {}
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_metric = evaluation_metric
        self.round_history: List[Dict[str, Any]] = []
        
        logger.info("Federated server initialized")
    
    def register_client(self, client: FederatedClient) -> None:
        """
        Register a client with the server.
        
        Args:
            client: Federated client to register
        """
        if client.client_id in self.clients:
            logger.warning(f"Client {client.client_id} already registered, overwriting")
        self.clients[client.client_id] = client
        logger.debug(f"Registered client {client.client_id}, total clients: {len(self.clients)}")
    
    def select_clients(self, fraction: float = 1.0, min_clients: int = 1) -> List[str]:
        """
        Select a subset of clients to participate in the current round.
        
        Args:
            fraction: Fraction of clients to select
            min_clients: Minimum number of clients to select
            
        Returns:
            List of selected client IDs
        """
        import random
        
        client_ids = list(self.clients.keys())
        num_clients = max(min_clients, int(fraction * len(client_ids)))
        num_clients = min(num_clients, len(client_ids))
        
        selected_clients = random.sample(client_ids, num_clients)
        logger.debug(f"Selected {len(selected_clients)}/{len(client_ids)} clients for training")
        return selected_clients
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the current global model parameters.
        
        Returns:
            Dictionary of parameter name -> tensor
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Update global model with provided parameters.
        
        Args:
            parameters: Dictionary of parameter name -> tensor
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
    
    def train_round(
        self,
        client_ids: Optional[List[str]] = None,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        proximal_term: float = 0.0
    ) -> Dict[str, Any]:
        """
        Execute a single round of federated training.
        
        Args:
            client_ids: List of client IDs to include (if None, use selection strategy)
            local_epochs: Number of local epochs for client training
            client_fraction: Fraction of clients to select if client_ids is None
            proximal_term: Coefficient for proximal term (for FedProx)
            
        Returns:
            Dictionary with round metrics
        """
        round_start_time = time.time()
        
        # Select clients for this round
        if client_ids is None:
            client_ids = self.select_clients(fraction=client_fraction)
            
        if not client_ids:
            logger.warning("No clients selected for training")
            return {"error": "No clients selected"}
            
        # Get global model parameters
        global_params = self.get_parameters()
        
        # Train on each selected client
        client_updates = []
        participating_clients = []
        
        for client_id in client_ids:
            if client_id not in self.clients:
                logger.warning(f"Client {client_id} not found, skipping")
                continue
                
            client = self.clients[client_id]
            
            # Update client with global model
            client.set_parameters(global_params)
            
            # Train client model
            updated_params, num_samples = client.train(
                epochs=local_epochs,
                proximal_term=proximal_term,
                global_model=self.model if proximal_term > 0 else None
            )
            
            # Store update
            client_updates.append((updated_params, num_samples))
            participating_clients.append(client_id)
            
            logger.debug(f"Client {client_id} completed training with {num_samples} samples")
        
        if not client_updates:
            logger.warning("No client updates received")
            return {"error": "No client updates received"}
        
        # Aggregate updates
        logger.info(f"Aggregating updates from {len(client_updates)} clients")
        aggregated_params = self.aggregation_strategy.aggregate(client_updates)
        
        # Update global model
        self.set_parameters(aggregated_params)
        
        # Evaluate if dataset available
        metrics = {}
        if self.evaluation_dataset is not None:
            metrics = self.evaluate()
            
        # Record round information
        round_info = {
            "participating_clients": participating_clients,
            "client_samples": [samples for _, samples in client_updates],
            "duration_seconds": time.time() - round_start_time,
            **metrics
        }
        
        self.round_history.append(round_info)
        logger.info(f"Round completed in {round_info['duration_seconds']:.2f}s, metrics: {metrics}")
        
        return round_info
    
    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the global model on a dataset.
        
        Args:
            dataset: Dataset to use for evaluation (if None, uses server's evaluation dataset)
            
        Returns:
            Dictionary with metrics (loss, accuracy)
        """
        if dataset is None:
            dataset = self.evaluation_dataset
            
        if dataset is None:
            logger.warning("No evaluation dataset available")
            return {}
            
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                total_loss += loss_fn(output, target).item() * data.shape[0]
                
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        metrics = {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
        
        # Add custom metrics if available
        if self.evaluation_metric is not None:
            self.model.eval()
            custom_metric = self.evaluation_metric(self.model, dataset)
            if isinstance(custom_metric, dict):
                metrics.update(custom_metric)
            else:
                metrics['custom_metric'] = custom_metric
        
        logger.info(f"Global model evaluation: {metrics}")
        return metrics
    
    def train(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        client_fraction: float = 1.0,
        proximal_term: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple rounds of federated training.
        
        Args:
            num_rounds: Number of federated training rounds
            local_epochs: Number of local epochs for client training
            client_fraction: Fraction of clients to select in each round
            proximal_term: Coefficient for proximal term (for FedProx)
            
        Returns:
            List of round metrics
        """
        logger.info(f"Starting federated training for {num_rounds} rounds")
        
        for round_idx in range(num_rounds):
            logger.info(f"Starting round {round_idx + 1}/{num_rounds}")
            
            round_metrics = self.train_round(
                local_epochs=local_epochs,
                client_fraction=client_fraction,
                proximal_term=proximal_term
            )
            
            if "error" in round_metrics:
                logger.error(f"Round {round_idx + 1} failed: {round_metrics['error']}")
                
        return self.round_history
        
    def save_model(self, path: str) -> None:
        """
        Save the global model to a file.
        
        Args:
            path: Path to save the model
        """
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved global model to {path}")
        
    def load_model(self, path: str) -> None:
        """
        Load the global model from a file.
        
        Args:
            path: Path to load the model from
        """
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Loaded global model from {path}")