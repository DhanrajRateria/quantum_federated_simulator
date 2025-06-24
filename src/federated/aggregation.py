"""
Aggregation strategies for federated learning.
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
        """Aggregate client model updates."""
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AggregationStrategy':
        """Create strategy instance from configuration."""
        strategy_type = config.get('type', 'fedavg').lower()
        logger.info(f"Creating aggregation strategy of type: {strategy_type}")

        if strategy_type == 'fedavg':
            return FedAvg()
        # FedProx config is handled during client training, server aggregation is FedAvg
        # We might still want to pass mu for logging/config consistency, but it's not used here.
        elif strategy_type == 'fedprox':
            # mu = config.get('mu', 0.0) # Mu not used in server aggregation
            logger.info("Using FedAvg aggregation for FedProx strategy on server-side.")
            return FedAvg() # Aggregation is FedAvg
        elif strategy_type == 'median':
            return Median()
        elif strategy_type == 'trimmed_mean':
            trim_ratio = config.get('trim_ratio', 0.1)
            logger.info(f"TrimmedMean trim_ratio: {trim_ratio}")
            return TrimmedMean(trim_ratio=trim_ratio)
        elif strategy_type == 'feddive':
            momentum = config.get('momentum', 0.9)
            epsilon = config.get('epsilon', 1e-8)
            temperature = config.get('temperature', 1.0)
            normalize_distances = config.get('normalize_distances', True)
            logger.info(f"FedDive parameters: momentum={momentum}, epsilon={epsilon}, "
                        f"temperature={temperature}, normalize_distances={normalize_distances}")
            return FedDive(momentum=momentum, epsilon=epsilon, temperature=temperature,
                           normalize_distances=normalize_distances)
        elif strategy_type == 'feddiver':
            momentum = config.get('momentum', 0.9)
            epsilon = config.get('epsilon', 1e-8)
            temperature = config.get('temperature', 1.0)
            normalize_distances = config.get('normalize_distances', True)
            logger.info(f"FedDive-R parameters: momentum={momentum}, epsilon={epsilon}, "
                        f"temperature={temperature}, normalize_distances={normalize_distances}")
            return FedDiveR(momentum=momentum, epsilon=epsilon, temperature=temperature,
                            normalize_distances=normalize_distances)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy_type}")


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy."""

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using weighted averaging."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("FedAvg: No client updates provided for aggregation.")
            return {}

        logger.info(f"FedAvg: Aggregating updates from {num_clients} clients.")

        # Extract parameters (state dicts) and weights
        # Assume parameters are already on CPU from client.get_parameters()
        parameters_list = [params for params, _ in client_updates]
        weights = np.array([weight for _, weight in client_updates], dtype=np.float32) # Use float32 for weights

        logger.debug(f"FedAvg: Client weights (sample counts): {weights}")

        # Check for invalid weights
        if np.any(weights < 0):
            logger.error("FedAvg: Received negative weights, aborting aggregation.")
            raise ValueError("Client weights cannot be negative.")

        # Normalize weights
        total_weight = weights.sum()
        if total_weight <= 0:
            logger.warning(f"FedAvg: Total weight is {total_weight}. Using equal weights for {num_clients} clients.")
            normalized_weights = np.ones(num_clients, dtype=np.float32) / num_clients
        else:
            normalized_weights = weights / total_weight
            logger.debug(f"FedAvg: Normalized weights: {normalized_weights}")


        # Initialize aggregated result (use structure of the first client's update)
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("FedAvg: First client update dictionary is empty.")
             return {}

        for name, tensor in first_client_params.items():
            aggregated_params[name] = torch.zeros_like(tensor, dtype=torch.float32) # Ensure float32 for accumulation

        # Perform weighted averaging
        for client_idx, client_params in enumerate(parameters_list):
            client_weight = normalized_weights[client_idx]
            if not client_params:
                 logger.warning(f"FedAvg: Client update at index {client_idx} is empty, skipping.")
                 continue

            for name, param_tensor in client_params.items():
                if name in aggregated_params:
                    # Ensure tensors are float for accumulation if they aren't already
                    aggregated_params[name] += client_weight * param_tensor.float()
                else:
                     logger.warning(f"FedAvg: Parameter '{name}' from client {client_idx} "
                                    f"not found in the first client's structure. Skipping.")

        # Optional: Convert back to original dtype if necessary, though float32 is common
        # for name in aggregated_params:
        #     aggregated_params[name] = aggregated_params[name].to(first_client_params[name].dtype)

        logger.info(f"FedAvg: Aggregation complete.")
        return aggregated_params


# FedProx on server just uses FedAvg aggregation
class FedProx(FedAvg):
     """Server-side aggregation for FedProx (identical to FedAvg)."""
     def __init__(self, mu: float = 0.0): # Mu is unused server-side but kept for consistency
        super().__init__()
        self.mu = mu # Store mu maybe for logging or config, but not used in aggregate()
        logger.debug(f"FedProx aggregation strategy initialized server-side (uses FedAvg). Mu={mu} (applied client-side).")

     # aggregate method is inherited from FedAvg


class Median(AggregationStrategy):
    """Coordinate-wise median aggregation strategy."""

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using coordinate-wise median."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("Median: No client updates provided.")
            return {}

        logger.info(f"Median: Aggregating updates from {num_clients} clients.")
        parameters_list = [params for params, _ in client_updates]

        # Initialize result
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("Median: First client update dictionary is empty.")
             return {}

        # Iterate through parameter names from the first client
        for name in first_client_params.keys():
            try:
                # Stack corresponding tensors from all clients (ensure float for median)
                # Handle cases where a client might not have the param (shouldn't happen with state_dict)
                tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
                if len(tensors_to_stack) != num_clients:
                     logger.warning(f"Median: Parameter '{name}' missing from some clients. "
                                    f"Aggregating over {len(tensors_to_stack)} values.")
                if not tensors_to_stack:
                     logger.warning(f"Median: No tensors found for parameter '{name}'. Skipping.")
                     continue

                stacked_tensors = torch.stack(tensors_to_stack, dim=0)
                # Compute median along the client dimension (dim=0)
                median_values = torch.median(stacked_tensors, dim=0).values
                # Convert back to original dtype if needed
                aggregated_params[name] = median_values.to(first_client_params[name].dtype)
                logger.debug(f"Median: Aggregated parameter '{name}' shape: {aggregated_params[name].shape}")
            except KeyError:
                 logger.warning(f"Median: Parameter '{name}' from first client not found in subsequent client, skipping.")
            except Exception as e:
                 logger.error(f"Median: Error aggregating parameter '{name}': {e}", exc_info=True)
                 # Decide whether to skip this param or raise error

        logger.info("Median: Aggregation complete.")
        return aggregated_params


class TrimmedMean(AggregationStrategy):
    """Trimmed mean aggregation strategy."""

    def __init__(self, trim_ratio: float = 0.1):
        super().__init__()
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("trim_ratio must be between 0 (inclusive) and 0.5 (exclusive)")
        self.trim_ratio = trim_ratio
        logger.debug(f"TrimmedMean strategy initialized with trim_ratio={trim_ratio}")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate using trimmed mean."""
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("TrimmedMean: No client updates provided.")
            return {}

        logger.info(f"TrimmedMean: Aggregating updates from {num_clients} clients (trim_ratio={self.trim_ratio}).")
        parameters_list = [params for params, _ in client_updates]

        # Calculate how many clients to trim from each end
        k = int(np.floor(self.trim_ratio * num_clients)) # Use floor to be safe
        logger.debug(f"TrimmedMean: Trimming k={k} clients from each end.")

        if 2 * k >= num_clients:
             logger.warning(f"TrimmedMean: Trim ratio ({self.trim_ratio}) too high for {num_clients} clients (k={k}). "
                            "Falling back to Median aggregation.")
             # Fallback to Median
             median_agg = Median()
             return median_agg.aggregate(client_updates)

        # Initialize result
        aggregated_params = {}
        first_client_params = parameters_list[0]
        if not first_client_params:
             logger.warning("TrimmedMean: First client update dictionary is empty.")
             return {}

        # For each parameter tensor name
        for name in first_client_params.keys():
            try:
                 # Stack corresponding tensors (ensure float for sorting/mean)
                 tensors_to_stack = [params[name].float() for params in parameters_list if name in params]
                 if len(tensors_to_stack) != num_clients:
                     logger.warning(f"TrimmedMean: Parameter '{name}' missing from some clients. "
                                    f"Aggregating over {len(tensors_to_stack)} values.")
                     # TODO: Decide how to handle missing params - skip? error? adjust k?
                     # For now, we proceed but the trim might be less effective
                 if not tensors_to_stack:
                     logger.warning(f"TrimmedMean: No tensors found for parameter '{name}'. Skipping.")
                     continue

                 stacked_tensors = torch.stack(tensors_to_stack, dim=0)

                 # Sort along the client dimension (dim=0)
                 sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)

                 # Select the tensors after trimming k from both ends
                 trimmed_tensors = sorted_tensors[k : num_clients - k]

                 # Average the remaining tensors
                 mean_values = torch.mean(trimmed_tensors, dim=0)
                 aggregated_params[name] = mean_values.to(first_client_params[name].dtype) # Convert back
                 logger.debug(f"TrimmedMean: Aggregated parameter '{name}' shape: {aggregated_params[name].shape} "
                              f"(trimmed {2*k}/{num_clients} values)")
            except KeyError:
                 logger.warning(f"TrimmedMean: Parameter '{name}' from first client not found in subsequent client, skipping.")
            except Exception as e:
                 logger.error(f"TrimmedMean: Error aggregating parameter '{name}': {e}", exc_info=True)

        logger.info("TrimmedMean: Aggregation complete.")
        return aggregated_params
    
class FedDive(AggregationStrategy):
    """
    Federated Diversity Averaging (FedDive) aggregation strategy.

    Combines momentum with a weighting scheme that prioritizes client updates
    that diverge most from the momentum-based average velocity.
    This implementation works with model parameters and maintains proper momentum.
    """
    def __init__(self, 
                 momentum: float = 0.9, 
                 epsilon: float = 1e-8, 
                 temperature: float = 1.0,
                 normalize_distances: bool = True):
        """
        Initialize FedDive aggregator.

        Args:
            momentum: Momentum coefficient (typically between 0 and 1).
            epsilon: Small value for numerical stability in distance calculation and softmax.
            temperature: Temperature parameter for softmax to control weighting sensitivity.
            normalize_distances: Whether to normalize distances before softmax.
        """
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Momentum must be between 0 and 1 (exclusive).")
        self.momentum = momentum
        self.velocity: Dict[str, torch.Tensor] = {}  # Store velocity per parameter name
        self.epsilon = epsilon  # For numerical stability
        self.temperature = temperature  # Control softmax sensitivity
        self.normalize_distances = normalize_distances  # Whether to normalize distances
        self.previous_global_params: Dict[str, torch.Tensor] = {}  # Store previous global parameters
        logger.debug(f"FedDive strategy initialized with momentum={momentum}, "
                    f"epsilon={epsilon}, temperature={temperature}, "
                    f"normalize_distances={normalize_distances}")

    def _calculate_update_distance(self, params: Dict[str, torch.Tensor], velocity: Dict[str, torch.Tensor]) -> float:
        """Calculates the L2 norm of the difference between parameter dicts."""
        total_diff_sq = 0.0
        with torch.no_grad():
            for name in params.keys():
                if name in velocity:
                    diff = params[name] - velocity[name]
                    total_diff_sq += torch.sum(diff * diff).item()
                else:
                    # Handle case where velocity hasn't been initialized for this param yet
                    total_diff_sq += torch.sum(params[name] * params[name]).item()
        return np.sqrt(total_diff_sq)

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client parameters using FedDive.

        Args:
            client_updates: List of (model_parameters, sample_count) tuples.
                           Sample counts are currently IGNORED by FedDive's core logic.

        Returns:
            Aggregated model parameters.
        """
        num_clients = len(client_updates)
        if num_clients == 0:
            logger.warning("FedDive: No client updates provided for aggregation.")
            return {}

        logger.info(f"FedDive: Aggregating updates from {num_clients} clients (momentum={self.momentum}).")
        parameters_list = [params for params, _ in client_updates]

        first_client_params = parameters_list[0]
        if not first_client_params:
            logger.warning("FedDive: First client update is empty.")
            return {}

        # --- Step 1: Calculate Average Parameters ---
        avg_params = {}
        for name, tensor in first_client_params.items():
            avg_params[name] = torch.zeros_like(tensor)
        
        for client_params in parameters_list:
            for name, param_tensor in client_params.items():
                if name in avg_params:
                    avg_params[name] += param_tensor  # Accumulate
        
        for name in avg_params:
            avg_params[name] /= num_clients  # Divide by count

        # --- Step 2: Update Velocity (Momentum) based on DELTAS ---
        # Initialize velocity and previous_global_params if needed
        if not self.velocity or not self.previous_global_params:
            logger.info("FedDive: Initializing velocity and previous parameters.")
            self.velocity = {name: torch.zeros_like(tensor) for name, tensor in first_client_params.items()}
            self.previous_global_params = {name: tensor.clone() for name, tensor in avg_params.items()}
        elif any(name not in self.velocity for name in first_client_params):
            logger.info("FedDive: Updating velocity and previous parameters structure.")
            for name, tensor in first_client_params.items():
                if name not in self.velocity:
                    self.velocity[name] = torch.zeros_like(tensor)
                if name not in self.previous_global_params:
                    self.previous_global_params[name] = tensor.clone()

        # Calculate deltas from previous global parameters to current average
        deltas = {}
        for name, avg_param in avg_params.items():
            if name in self.previous_global_params:
                deltas[name] = avg_param - self.previous_global_params[name]
            else:
                deltas[name] = avg_param  # If no previous, use current as delta

        # Update velocity using proper momentum formula: v = momentum * v + (1 - momentum) * delta
        with torch.no_grad():
            for name in self.velocity:
                if name in deltas:
                    self.velocity[name] = self.momentum * self.velocity[name] + \
                                         (1.0 - self.momentum) * deltas[name]

        logger.debug("FedDive: Velocity updated based on parameter deltas.")

        # --- Step 3: Calculate Diversity Weights ---
        distances = np.array([
            self._calculate_update_distance(params, self.velocity) for params in parameters_list
        ])
        
        # Normalize distances if enabled (subtract mean and optionally scale by std)
        if self.normalize_distances and len(distances) > 1:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances) if np.std(distances) > self.epsilon else 1.0
            distances = (distances - mean_dist) / std_dist
            logger.debug(f"FedDive: Normalized distances: {distances}")
        
        # Apply temperature scaling to control sensitivity
        scaled_distances = distances / self.temperature
        
        # Use softmax with temperature scaling
        exp_distances = np.exp(scaled_distances + self.epsilon)
        diversity_weights = exp_distances / np.sum(exp_distances)
        
        logger.debug(f"FedDive: Diversity weights: {diversity_weights}")

        # --- Step 4: Weighted Aggregation using Diversity Weights ---
        aggregated_params = {}
        for name, tensor in first_client_params.items():
            aggregated_params[name] = torch.zeros_like(tensor)

        for client_idx, client_params in enumerate(parameters_list):
            client_weight = diversity_weights[client_idx]
            for name, param_tensor in client_params.items():
                if name in aggregated_params:
                    aggregated_params[name] += client_weight * param_tensor
                else:
                    logger.warning(f"FedDive: Param '{name}' missing.")

        # Store current aggregated parameters as the previous for next round
        self.previous_global_params = {name: tensor.clone() for name, tensor in aggregated_params.items()}

        logger.info("FedDive: Aggregation complete.")
        return aggregated_params
    
class FedDiveR(FedDive):
    """
    Robust Federated Diversity Averaging (FedDive-R) aggregation strategy.

    This version enhances FedDive's robustness to outliers by using a
    robust aggregator (Median) to calculate the parameter delta for updating
    the velocity vector. This prevents a single noisy client from poisoning
    the momentum-based consensus direction.
    """
    def __init__(self,
                 momentum: float = 0.9,
                 epsilon: float = 1e-8,
                 temperature: float = 1.0,
                 normalize_distances: bool = True):
        """Initialize FedDive-R aggregator."""
        super().__init__(momentum, epsilon, temperature, normalize_distances)
        self.robust_aggregator = Median()
        logger.info("FedDive-R (Robust) strategy initialized.")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client parameters using FedDive-R.
        """
        num_clients = len(client_updates)
        if num_clients == 0:
            return {}

        logger.info(f"FedDive-R: Aggregating updates from {num_clients} clients.")
        parameters_list = [params for params, _ in client_updates]
        first_client_params = parameters_list[0]

        # --- Step 1: Calculate Robust Average for Velocity Update ---
        # This is the key change: use a robust aggregator to find the "true" center
        # to prevent outliers from poisoning the velocity calculation.
        logger.info("FedDive-R: Using Median to calculate robust center for velocity update.")
        robust_avg_params = self.robust_aggregator.aggregate(client_updates)

        # --- Step 2: Update Velocity (Momentum) based on ROBUST DELTAS ---
        if not self.velocity or not self.previous_global_params:
            # Initialize velocity and previous params on the first run
            logger.info("FedDive-R: Initializing velocity and previous parameters.")
            self.velocity = {name: torch.zeros_like(tensor) for name, tensor in first_client_params.items()}
            self.previous_global_params = {name: tensor.clone() for name, tensor in robust_avg_params.items()}

        # Calculate deltas from previous global parameters to the current ROBUST average
        deltas = {name: robust_avg_params[name] - self.previous_global_params[name] for name in robust_avg_params}
        
        # Update velocity using the standard momentum formula on the robust delta
        with torch.no_grad():
            for name in self.velocity:
                if name in deltas:
                    self.velocity[name] = self.momentum * self.velocity[name] + \
                                         (1.0 - self.momentum) * deltas[name]

        logger.debug("FedDive-R: Robust velocity updated.")

        # --- Step 3: Calculate Diversity Weights (using original params and robust velocity) ---
        # The rest of the algorithm proceeds as in standard FedDive. We measure the
        # distance of each original update from the now-robust velocity.
        distances = np.array([
            self._calculate_update_distance(params, self.velocity) for params in parameters_list
        ])
        
        if self.normalize_distances and len(distances) > 1:
            mean_dist = np.mean(distances)
            std_dist = np.std(distances) if np.std(distances) > self.epsilon else 1.0
            distances = (distances - mean_dist) / std_dist
        
        scaled_distances = distances / self.temperature
        exp_distances = np.exp(scaled_distances + self.epsilon)
        diversity_weights = exp_distances / np.sum(exp_distances)
        
        logger.debug(f"FedDive-R: Diversity weights: {diversity_weights}")

        # --- Step 4: Weighted Aggregation using Diversity Weights on original params ---
        aggregated_params = {name: torch.zeros_like(tensor) for name, tensor in first_client_params.items()}

        for client_idx, client_params in enumerate(parameters_list):
            client_weight = diversity_weights[client_idx]
            for name, param_tensor in client_params.items():
                if name in aggregated_params:
                    aggregated_params[name] += client_weight * param_tensor

        # Store current aggregated parameters as the previous for the next round
        self.previous_global_params = {name: tensor.clone() for name, tensor in aggregated_params.items()}

        logger.info("FedDive-R: Aggregation complete.")
        return aggregated_params