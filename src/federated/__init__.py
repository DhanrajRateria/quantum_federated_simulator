"""
Federated learning package for the Quantum Federated Learning Simulator.

This package provides components for federated learning, including client and server
implementations and aggregation strategies. The implementation is designed to be
extended for quantum computing in the future.
"""

from src.federated.aggregation import (
    AggregationStrategy,
    FedAvg,
    FedProx,
    Median,
    TrimmedMean
)

from src.federated.client import FederatedClient
from src.federated.server import FederatedServer

__all__ = [
    # Aggregation strategies
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "Median",
    "TrimmedMean",
    
    # Core components
    "FederatedClient",
    "FederatedServer",
]

# Package metadata
__version__ = "0.1.0"