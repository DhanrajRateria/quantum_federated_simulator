"""
Aggregation strategies for quantum federated learning.
This module defines various methods to aggregate quantum model parameters
from multiple clients.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

def federated_average(parameters: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """
    Implement the Federated Averaging (FedAvg) algorithm.
    
    Args:
        parameters: Dict mapping client IDs to their model parameters
        weights: Dict mapping client IDs to their aggregation weights
        
    Returns:
        Weighted average of client parameters
    """
    if not parameters:
        raise ValueError("No parameters to aggregate")
    
    # Verify all clients have weights
    for client_id in parameters:
        if client_id not in weights:
            raise ValueError(f"Client {client_id} has no assigned weight")
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        raise ValueError("Sum of weights is zero")
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Get the shape from the first client to initialize the result
    first_client_id = list(parameters.keys())[0]
    aggregated = np.zeros_like(parameters[first_client_id])
    
    # Compute weighted average
    for client_id, params in parameters.items():
        if params.shape != aggregated.shape:
            raise ValueError(f"Parameter shape mismatch for client {client_id}")
        
        weight = normalized_weights[client_id]
        aggregated += weight * params
    
    logger.debug(f"Aggregated parameters from {len(parameters)} clients using FedAvg")
    return aggregated

def federated_median(parameters: Dict[str, np.ndarray], weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Implement a coordinate-wise median aggregation, which is more robust to outliers.
    
    Args:
        parameters: Dict mapping client IDs to their model parameters
        weights: Optional dict mapping client IDs to weights (not used for median)
        
    Returns:
        Element-wise median of client parameters
    """
    if not parameters:
        raise ValueError("No parameters to aggregate")
    
    # Get the shape from the first client to validate consistency
    first_client_id = list(parameters.keys())[0]
    first_params = parameters[first_client_id]
    param_shape = first_params.shape
    
    # Stack parameters from all clients
    param_stack = []
    for client_id, params in parameters.items():
        if params.shape != param_shape:
            raise ValueError(f"Parameter shape mismatch for client {client_id}")
        param_stack.append(params)
    
    # Convert to numpy array
    stacked_params = np.stack(param_stack, axis=0)
    
    # Compute median along client dimension
    aggregated = np.median(stacked_params, axis=0)
    
    logger.debug(f"Aggregated parameters from {len(parameters)} clients using coordinate-wise median")
    return aggregated

def trimmed_mean(parameters: Dict[str, np.ndarray], beta: float = 0.1) -> np.ndarray:
    """
    Implement trimmed mean aggregation, which removes the highest and lowest beta fraction
    of values before averaging.
    
    Args:
        parameters: Dict mapping client IDs to their model parameters
        beta: Fraction of highest and lowest values to exclude
        
    Returns:
        Trimmed mean of client parameters
    """
    if not parameters:
        raise ValueError("No parameters to aggregate")
    
    if beta < 0 or beta >= 0.5:
        raise ValueError("Beta must be in [0, 0.5)")
    
    # Get the shape from the first client
    first_client_id = list(parameters.keys())[0]
    first_params = parameters[first_client_id]
    param_shape = first_params.shape
    
    # Stack parameters from all clients
    param_stack = []
    for client_id, params in parameters.items():
        if params.shape != param_shape:
            raise ValueError(f"Parameter shape mismatch for client {client_id}")
        param_stack.append(params)
    
    # Convert to numpy array
    stacked_params = np.stack(param_stack, axis=0)
    
    # Sort along client dimension
    n_clients = len(parameters)
    n_remove = int(beta * n_clients)
    
    # Compute trimmed mean
    sorted_params = np.sort(stacked_params, axis=0)
    trimmed_params = sorted_params[n_remove:n_clients - n_remove]
    aggregated = np.mean(trimmed_params, axis=0)
    
    logger.debug(f"Aggregated parameters from {len(parameters)} clients using trimmed mean (beta={beta})")
    return aggregated

def krum(parameters: Dict[str, np.ndarray], f: int = 1) -> np.ndarray:
    """
    Implement Krum aggregation, which is robust to Byzantine attacks.
    Selects the parameter vector with minimal sum of squared distances to its closest
    n-f-2 neighbors.
    
    Args:
        parameters: Dict mapping client IDs to their model parameters
        f: Upper bound on number of Byzantine clients
        
    Returns:
        Selected client's parameters
    """
    if not parameters:
        raise ValueError("No parameters to aggregate")
    
    n_clients = len(parameters)
    if f >= n_clients:
        raise ValueError(f"f must be less than the number of clients ({n_clients})")
    
    # Flatten parameters to compute distances
    flattened_params = {}
    for client_id, params in parameters.items():
        flattened_params[client_id] = params.flatten()
    
    # Compute pairwise squared distances
    distances = {}
    for client_id_1 in flattened_params:
        distances[client_id_1] = {}
        for client_id_2 in flattened_params:
            if client_id_1 != client_id_2:
                dist = np.sum((flattened_params[client_id_1] - flattened_params[client_id_2]) ** 2)
                distances[client_id_1][client_id_2] = dist
    
    # For each client, compute sum of squared distances to closest n-f-2 neighbors
    scores = {}
    for client_id in distances:
        sorted_dists = sorted(distances[client_id].values())
        # Sum of distances to n-f-2 closest vectors
        scores[client_id] = sum(sorted_dists[:n_clients - f - 2])
    
    # Select client with minimal score
    selected_client = min(scores, key=scores.get)
    
    logger.debug(f"Selected parameters from client {selected_client} using Krum (f={f})")
    return parameters[selected_client]

def coordinate_wise_op(parameters: Dict[str, np.ndarray], op_func) -> np.ndarray:
    """
    Apply a coordinate-wise operation to aggregate parameters.
    
    Args:
        parameters: Dict mapping client IDs to their model parameters
        op_func: Element-wise operation function
        
    Returns:
        Aggregated parameters
    """
    if not parameters:
        raise ValueError("No parameters to aggregate")
    
    # Get the shape from the first client
    first_client_id = list(parameters.keys())[0]
    first_params = parameters[first_client_id]
    param_shape = first_params.shape
    
    # Stack parameters from all clients
    param_stack = []
    for client_id, params in parameters.items():
        if params.shape != param_shape:
            raise ValueError(f"Parameter shape mismatch for client {client_id}")
        param_stack.append(params)
    
    # Convert to numpy array and apply operation
    stacked_params = np.stack(param_stack, axis=0)
    aggregated = op_func(stacked_params, axis=0)
    
    return aggregated