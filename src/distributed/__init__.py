"""
Distributed Package for Quantum Federated Learning Simulator

This package contains modules for distributed computing functionality,
enabling the simulation of quantum federated learning across multiple nodes.
"""

from .spark_manager import SparkManager
from .node_manager import NodeManager
from .communication import (
    CommunicationManager, 
    MessageType, 
    CompressionType
)

__all__ = [
    'SparkManager',
    'NodeManager',
    'CommunicationManager',
    'MessageType',
    'CompressionType'
]