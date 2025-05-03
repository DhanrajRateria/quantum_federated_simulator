"""
Distributed computing module for Federated Learning Simulator.

This package provides components for managing distributed nodes,
communication between nodes, and integration with Apache Spark
for large-scale simulations.
"""

from .spark_manager import SparkManager
from .spark_executor_task import run_client_training_partition

__all__ = [
    'SparkManager',
    'run_client_training_partition'
]