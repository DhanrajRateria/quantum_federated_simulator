"""
Distributed module for the Quantum Federated Learning Simulator.

This module provides functionality for managing distributed computation
across multiple nodes, handling communication, and integrating with
Apache Spark for large-scale simulations.
"""

from src.distributed.communication import Communication, CommunicationProtocol
from src.distributed.node_manager import NodeManager, NodeStatus, Node
from src.distributed.spark_manager import SparkManager, SparkClusterMode

__all__ = [
    'Communication', 
    'CommunicationProtocol',
    'NodeManager', 
    'NodeStatus', 
    'Node',
    'SparkManager',
    'SparkClusterMode'
]