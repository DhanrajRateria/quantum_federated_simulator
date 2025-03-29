"""
Aggregation strategies for federated learning.

This module provides various aggregation methods for combining model updates
from multiple clients in a federated learning setting.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AggregationStrategy(ABC):
    """Base abstract class for aggregation strategies."""

    @abstractmethod
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client model updates into a single global update.

        Args:
            client_updates: List of tuples, where each tuple contains:
                - A dictionary of model parameters (name -> tensor)
                - The weight to assign to this client (e.g., sample count)

        Returns:
            Dictionary of aggregated model parameters
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AggregationStrategy':
        """
        Create an instance of the aggregation strategy from configuration.
        
        Args:
            config: Dictionary containing configuration parameters
            
        Returns:
            Instance of the aggregation strategy
        """
        strategy_type = config.get('type', 'fedavg').lower()
        
        if strategy_type == 'fedavg':
            return FedAvg()
        elif strategy_type == 'fedprox':
            mu = config.get('mu', 0.01)
            return FedProx(mu=mu)
        elif strategy_type == 'median':
            return Median()
        elif strategy_type == 'trimmed_mean':
            trim_ratio = config.get('trim_ratio', 0.1)
            return TrimmedMean(trim_ratio=trim_ratio)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")


class FedAvg(AggregationStrategy):
    """
    Federated Averaging (FedAvg) aggregation strategy.
    
    As described in "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    by McMahan et al.
    """
    
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using weighted averaging.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            logger.warning("No client updates provided for aggregation")
            return {}
            
        # Extract parameters and weights
        parameters_list = [params for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates])
        
        # Normalize weights
        total_weight = weights.sum()
        if total_weight == 0:
            logger.warning("Total weight is zero, using simple averaging")
            weights = np.ones_like(weights) / len(weights)
        else:
            weights = weights / total_weight
            
        # Initialize result with zeros like the first client's parameters
        result = {}
        for name, tensor in parameters_list[0].items():
            result[name] = torch.zeros_like(tensor)
        
        # Perform weighted averaging
        for client_idx, parameters in enumerate(parameters_list):
            client_weight = weights[client_idx]
            for name, tensor in parameters.items():
                if name in result:
                    result[name] += client_weight * tensor
        
        logger.debug(f"Aggregated updates from {len(client_updates)} clients using FedAvg")
        return result


class FedProx(FedAvg):
    """
    FedProx aggregation strategy.
    
    As described in "Federated Optimization in Heterogeneous Networks"
    by Li et al. This implementation only handles the aggregation part.
    The proximal term needs to be applied during client training.
    """
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregator.
        
        Args:
            mu: Proximal term parameter that controls how much to penalize
                deviation from the global model
        """
        super().__init__()
        self.mu = mu
        
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using FedProx.
        
        Note: The actual proximal term is applied during client training.
        The aggregation is the same as FedAvg.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated model parameters
        """
        logger.debug(f"Aggregating with FedProx (mu={self.mu})")
        return super().aggregate(client_updates)


class Median(AggregationStrategy):
    """
    Coordinate-wise median aggregation strategy.
    
    More robust to outliers and poisoning attacks than simple averaging.
    """
    
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using coordinate-wise median.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            logger.warning("No client updates provided for aggregation")
            return {}
            
        # Extract parameters (weights are ignored for median)
        parameters_list = [params for params, _ in client_updates]
        
        # Initialize result
        result = {}
        
        # For each parameter tensor
        for name, tensor in parameters_list[0].items():
            # Stack corresponding tensors from all clients
            stacked_tensors = torch.stack([params[name] for params in parameters_list if name in params])
            # Compute median along the first dimension (client dimension)
            result[name] = torch.median(stacked_tensors, dim=0).values
        
        logger.debug(f"Aggregated updates from {len(client_updates)} clients using Median")
        return result


class TrimmedMean(AggregationStrategy):
    """
    Trimmed mean aggregation strategy.
    
    Removes a percentage of the smallest and largest values before averaging.
    More robust to outliers and poisoning attacks than simple averaging.
    """
    
    def __init__(self, trim_ratio: float = 0.1):
        """
        Initialize Trimmed Mean aggregator.
        
        Args:
            trim_ratio: Percentage of values to trim from each end
        """
        super().__init__()
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be between 0 and 0.5")
        self.trim_ratio = trim_ratio
        
    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates using trimmed mean.
        
        Args:
            client_updates: List of (model_parameters, weight) tuples
            
        Returns:
            Aggregated model parameters
        """
        if not client_updates:
            logger.warning("No client updates provided for aggregation")
            return {}
            
        # Extract parameters (weights are ignored for trimmed mean)
        parameters_list = [params for params, _ in client_updates]
        num_clients = len(parameters_list)
        
        # Calculate how many clients to trim from each end
        k = int(self.trim_ratio * num_clients)
        
        # Initialize result
        result = {}
        
        # For each parameter tensor
        for name, tensor in parameters_list[0].items():
            # Stack corresponding tensors from all clients
            stacked_tensors = torch.stack([params[name] for params in parameters_list if name in params])
            
            # Sort along the first dimension (client dimension)
            sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)
            
            # Remove k smallest and k largest values
            if 2*k < num_clients:
                trimmed_tensors = sorted_tensors[k:num_clients-k]
                # Average the remaining tensors
                result[name] = torch.mean(trimmed_tensors, dim=0)
            else:
                # If we're trying to trim too much, fall back to median
                result[name] = torch.median(stacked_tensors, dim=0).values
        
        logger.debug(f"Aggregated updates from {len(client_updates)} clients using TrimmedMean (ratio={self.trim_ratio})")
        return result