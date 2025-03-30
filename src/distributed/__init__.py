"""
Distributed computing module for Federated Learning Simulator.

This package provides components for managing distributed nodes,
communication between nodes, and integration with Apache Spark
for large-scale simulations.
"""

from .node_manager import NodeManager, Node, NodeStatus
from .communication import MessageType, CommunicationProtocol

__all__ = [
    'NodeManager',
    'Node',
    'NodeStatus',
    'MessageType',
    'CommunicationProtocol',
]