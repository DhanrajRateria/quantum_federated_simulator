"""
Server implementation for quantum federated learning.
This module defines the server-side logic for coordinating the federated
learning process across quantum clients.
"""

import numpy as np
import json
import logging
import os
import time
from typing import List, Dict, Optional, Tuple, Union, Callable, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

class QuantumFederatedServer:
    """Server for quantum federated learning."""
    
    def __init__(self, config: Dict):
        """
        Initialize a federated learning server.
        
        Args:
            config: Dictionary containing server configuration
        """
        self.config = config
        
        # Federated learning configuration
        self.rounds = config.get('rounds', 10)
        self.min_clients = config.get('min_clients', 2)
        self.client_fraction = config.get('client_fraction', 1.0)
        
        # Aggregation settings
        self.aggregation_method = config.get('aggregation_method', 'fedavg')
        
        # Model configuration
        self.model_config = config.get('model', {})
        self.global_parameters = None
        self.parameter_shape = None
        
        # Tracking and metrics
        self.metrics = {
            'global_rounds': [],
            'participating_clients': [],
            'aggregation_time': [],
            'global_loss': [],
            'global_accuracy': []
        }
        
        # Client state tracking
        self.clients = {}
        
        logger.info(f"Initialized server with {self.aggregation_method} aggregation")
    
    def register_client(self, client_id: str, client_info: Dict) -> None:
        """
        Register a new client with the server.
        
        Args:
            client_id: Unique identifier for the client
            client_info: Information about the client
        """
        self.clients[client_id] = {
            'info': client_info,
            'samples': client_info.get('n_samples', 0),
            'last_update': None,
            'metrics': defaultdict(list)
        }
        logger.info(f"Registered client {client_id} with {client_info.get('n_samples', 0)} samples")
    
    def select_clients(self, round_num: int) -> List[str]:
        """
        Select clients to participate in the current round.
        
        Args:
            round_num: Current round number
            
        Returns:
            List of selected client IDs
        """
        available_clients = list(self.clients.keys())
        n_clients = max(self.min_clients, int(len(available_clients) * self.client_fraction))
        n_clients = min(n_clients, len(available_clients))
        
        # Deterministic selection based on round number for reproducibility
        np.random.seed(round_num)
        selected_clients = np.random.choice(available_clients, n_clients, replace=False).tolist()
        
        logger.info(f"Selected {len(selected_clients)} clients for round {round_num}")
        return selected_clients
    
    def initialize_global_parameters(self, parameters: np.ndarray) -> None:
        """
        Initialize global model parameters.
        
        Args:
            parameters: Initial model parameters
        """
        self.global_parameters = parameters
        self.parameter_shape = parameters.shape
        logger.info(f"Initialized global parameters with shape {self.parameter_shape}")
    
    def aggregate_parameters(self, client_parameters: Dict[str, np.ndarray], client_metrics: Dict[str, Dict]) -> np.ndarray:
        """
        Aggregate parameters from multiple clients.
        
        Args:
            client_parameters: Dictionary mapping client IDs to their parameters
            client_metrics: Dictionary mapping client IDs to their training metrics
            
        Returns:
            Aggregated global parameters
        """
        start_time = time.time()
        
        if self.aggregation_method == 'fedavg':
            # FedAvg: Weighted average based on number of samples
            total_samples = sum(self.clients[client_id]['samples'] for client_id in client_parameters)
            
            # Initialize with zeros in the correct shape
            if self.parameter_shape is None and len(client_parameters) > 0:
                # Get shape from first client
                first_client_id = list(client_parameters.keys())[0]
                self.parameter_shape = client_parameters[first_client_id].shape
            
            aggregated_params = np.zeros(self.parameter_shape)
            
            # Weighted aggregation
            for client_id, params in client_parameters.items():
                weight = self.clients[client_id]['samples'] / total_samples
                aggregated_params += weight * params
                
                # Record client metrics
                for metric_name, metric_value in client_metrics.get(client_id, {}).items():
                    self.clients[client_id]['metrics'][metric_name].append(metric_value)
            
        elif self.aggregation_method == 'fedprox':
            # FedProx: Similar to FedAvg but with proximal term regularization
            # In actual aggregation, this is the same as FedAvg, but clients use different
            # optimization objective during training
            total_samples = sum(self.clients[client_id]['samples'] for client_id in client_parameters)
            
            aggregated_params = np.zeros(self.parameter_shape)
            
            for client_id, params in client_parameters.items():
                weight = self.clients[client_id]['samples'] / total_samples
                aggregated_params += weight * params
                
                # Record client metrics
                for metric_name, metric_value in client_metrics.get(client_id, {}).items():
                    self.clients[client_id]['metrics'][metric_name].append(metric_value)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        aggregation_time = time.time() - start_time
        self.metrics['aggregation_time'].append(aggregation_time)
        
        logger.info(f"Aggregated parameters from {len(client_parameters)} clients in {aggregation_time:.2f} seconds")
        return aggregated_params
    
    def update_global_model(self, client_parameters: Dict[str, np.ndarray], client_metrics: Dict[str, Dict]) -> None:
        """
        Update the global model with aggregated client parameters.
        
        Args:
            client_parameters: Dictionary mapping client IDs to their parameters
            client_metrics: Dictionary mapping client IDs to their training metrics
        """
        self.global_parameters = self.aggregate_parameters(client_parameters, client_metrics)
    
    def get_global_parameters(self) -> np.ndarray:
        """
        Get current global model parameters.
        
        Returns:
            Global model parameters
        """
        return self.global_parameters
    
    def run_federated_learning(self, evaluate_fn: Optional[Callable] = None) -> Dict:
        """
        Execute the complete federated learning process.
        
        Args:
            evaluate_fn: Optional function to evaluate global model
            
        Returns:
            Training metrics
        """
        for round_num in range(1, self.rounds + 1):
            logger.info(f"Starting federated round {round_num}/{self.rounds}")
            
            # Select clients
            selected_clients = self.select_clients(round_num)
            self.metrics['participating_clients'].append(len(selected_clients))
            
            # Distribute global model
            # This would be handled by communication module in a real system
            
            # Collect client updates (placeholder for actual client training)
            client_parameters = {}
            client_metrics = {}
            
            # In a real system, clients would train in parallel
            # Here we represent this sequentially
            
            # Update global model
            self.update_global_model(client_parameters, client_metrics)
            self.metrics['global_rounds'].append(round_num)
            
            # Evaluate global model if evaluation function provided
            if evaluate_fn is not None:
                eval_results = evaluate_fn(self.global_parameters)
                self.metrics['global_loss'].append(eval_results.get('loss'))
                self.metrics['global_accuracy'].append(eval_results.get('accuracy'))
                
                logger.info(f"Round {round_num} evaluation - Loss: {eval_results.get('loss'):.4f}, "
                           f"Accuracy: {eval_results.get('accuracy'):.4f}")
        
        return self.metrics
    
    def save_metrics(self, output_dir: str) -> None:
        """
        Save server metrics to file.
        
        Args:
            output_dir: Directory to save metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save global metrics
        global_metrics_file = os.path.join(output_dir, "server_global_metrics.json")
        with open(global_metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save client metrics
        client_metrics_file = os.path.join(output_dir, "server_client_metrics.json")
        client_metrics = {client_id: data['metrics'] for client_id, data in self.clients.items()}
        with open(client_metrics_file, 'w') as f:
            json.dump(client_metrics, f, indent=2)
        
        logger.info(f"Saved server metrics to {output_dir}")
    
    def serialize_parameters(self) -> Dict:
        """
        Serialize global parameters for transmission.
        
        Returns:
            Dictionary with serialized parameters
        """
        params = self.get_global_parameters()
        
        # Convert to serializable format
        if isinstance(params, np.ndarray):
            return {
                'format': 'numpy',
                'shape': params.shape,
                'dtype': str(params.dtype),
                'data': params.tolist()
            }
        else:
            # Handle other parameter types
            return {
                'format': 'list',
                'data': params
            }
    
    def deserialize_parameters(self, serialized_params: Dict) -> np.ndarray:
        """
        Deserialize parameters received from clients.
        
        Args:
            serialized_params: Dictionary with serialized parameters
            
        Returns:
            Deserialized parameters
        """
        if serialized_params['format'] == 'numpy':
            # Reconstruct numpy array
            shape = tuple(serialized_params['shape'])
            dtype = np.dtype(serialized_params['dtype'])
            return np.array(serialized_params['data'], dtype=dtype).reshape(shape)
        else:
            # Handle other parameter types
            return serialized_params['data']