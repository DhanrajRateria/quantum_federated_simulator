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