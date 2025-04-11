"""
Federated Quantum Learning Manager.

This module provides utilities for managing quantum models in federated learning,
including initialization, aggregation, and adaptation strategies.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import types

from src.federated.server import FederatedServer
from src.federated.aggregation import AggregationStrategy
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.core.quantum_client import QuantumFederatedClient

logger = logging.getLogger(__name__)


class FederatedQuantumManager:
    """
    Manager for federated learning with quantum models.
    
    This class provides utilities for initializing and managing quantum models
    in a federated learning setting.
    """
    
    @staticmethod
    def create_quantum_clients(
        client_ids: List[str],
        model_class: Any,
        model_kwargs: Dict[str, Any],
        datasets: List[Any],
        client_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[QuantumFederatedClient]:
        """
        Create a list of quantum federated clients with identical model architectures.
        
        Args:
            client_ids: List of client IDs
            model_class: Quantum model class (VariationalQuantumClassifier, QuantumNeuralNetwork, etc.)
            model_kwargs: Keyword arguments for model initialization
            datasets: List of datasets, one per client
            client_kwargs: Additional keyword arguments for client initialization
            
        Returns:
            List of initialized quantum federated clients
        """
        if len(client_ids) != len(datasets):
            raise ValueError("Number of client IDs must match number of datasets")
        
        clients = []
        client_kwargs = client_kwargs or {}
        
        for i, (client_id, dataset) in enumerate(zip(client_ids, datasets)):
            # Create a fresh model instance for each client
            model = model_class(**model_kwargs)
            
            # Initialize client
            client = QuantumFederatedClient(
                client_id=client_id,
                model=model,
                dataset=dataset,
                **client_kwargs
            )
            
            clients.append(client)
            logger.debug(f"Created quantum client {client_id} with {len(dataset)} samples")
        
        return clients
    
    @staticmethod
    def initialize_quantum_server(
        model_class: Any,
        model_kwargs: Dict[str, Any],
        aggregation_strategy: AggregationStrategy,
        evaluation_dataset: Optional[Any] = None,
        server_kwargs: Optional[Dict[str, Any]] = None
    ) -> FederatedServer:
        """
        Initialize a federated server with a quantum model.
        
        Args:
            model_class: Quantum model class
            model_kwargs: Keyword arguments for model initialization
            aggregation_strategy: Strategy for aggregating client updates
            evaluation_dataset: Optional dataset for server-side evaluation
            server_kwargs: Additional keyword arguments for server initialization
            
        Returns:
            Initialized federated server
        """
        # Initialize the global model
        global_model = model_class(**model_kwargs)
        
        # Initialize the server
        server_kwargs = server_kwargs or {}
        server = FederatedServer(
            model=global_model,
            aggregation_strategy=aggregation_strategy,
            evaluation_dataset=evaluation_dataset,
            **server_kwargs
        )
        
        logger.debug(f"Initialized quantum federated server with {model_class.__name__} model")
        return server
    
    @staticmethod
    def adapt_evaluation_for_quantum_models(
        server: FederatedServer,
        evaluation_fn: Callable
    ) -> None:
        """
        Adapt the server's evaluation method for quantum models.
        
        Args:
            server: Federated server instance
            evaluation_fn: Custom evaluation function for quantum models
            
        This function replaces the server's evaluate method with a custom one.
        """
        server.evaluate = types.MethodType(evaluation_fn, server)
        logger.debug("Adapted server evaluation method for quantum models")


class QuantumAggregationStrategy(AggregationStrategy):
    """
    Aggregation strategy specialized for quantum models.
    
    This strategy handles the nuances of aggregating quantum model parameters,
    which may require special handling due to their unitary nature.
    """
    
    def __init__(self, base_strategy: AggregationStrategy = None):
        """
        Initialize the quantum aggregation strategy.
        
        Args:
            base_strategy: Underlying aggregation strategy to adapt for quantum models
        """
        self.base_strategy = base_strategy or FedAvg()
        
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate quantum model updates.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated quantum model parameters
        """
        # Check if this is a VQC model by looking for "quantum_params" in parameters
        is_vqc_model = all("quantum_params" in params for params, _ in client_updates)
        
        if is_vqc_model:
            # Special handling for VQC model parameters
            return self._aggregate_vqc_params(client_updates)
        else:
            # Use base strategy for other model types
            return self.base_strategy.aggregate(client_updates)
    
    def _aggregate_vqc_params(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate VQC model parameters.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated VQC model parameters
        """
        if not client_updates:
            logger.warning("No client updates provided for aggregation")
            return {}
            
        # Extract quantum parameters and weights
        param_tensors = [params["quantum_params"] for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates])
        
        # Normalize weights
        total_weight = weights.sum()
        if total_weight == 0:
            logger.warning("Total weight is zero, using simple averaging")
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total_weight
            
        # Initialize result tensor
        result_tensor = torch.zeros_like(param_tensors[0])
        
        # Weighted averaging
        for i, param_tensor in enumerate(param_tensors):
            result_tensor += weights[i] * param_tensor
        
        return {"quantum_params": result_tensor}