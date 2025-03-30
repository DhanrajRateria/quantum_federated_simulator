"""
Node Manager for Quantum Federated Learning Simulator.

This module provides functionality for managing distributed nodes,
including registration, status monitoring, selection for training,
and lifecycle management.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Set, Callable, Any
import uuid
from enum import Enum

from .communication import MessageType, CommunicationProtocol

# Configure logging
logger = logging.getLogger(__name__)

class NodeStatus(Enum):
    """Possible states of a federated learning node."""
    OFFLINE = "offline"
    ONLINE = "online"
    TRAINING = "training"
    ERROR = "error"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"


@dataclass
class Node:
    """Representation of a node in the federated learning network."""
    node_id: str
    ip_address: str
    port: int
    status: NodeStatus = NodeStatus.OFFLINE
    compute_power: float = 1.0  # Relative computational capability
    memory: float = 1.0  # Available memory in GB
    last_heartbeat: float = 0.0  # Timestamp of last heartbeat
    capabilities: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "ip_address": self.ip_address,
            "port": self.port,
            "status": self.status.value,
            "compute_power": self.compute_power,
            "memory": self.memory,
            "last_heartbeat": self.last_heartbeat,
            "capabilities": self.capabilities
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create a Node instance from a dictionary."""
        # Convert string status back to enum
        data["status"] = NodeStatus(data["status"])
        return cls(**data)
    
    def serialize(self) -> str:
        """Serialize node to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def deserialize(cls, serialized: str) -> 'Node':
        """Deserialize JSON string to Node instance."""
        return cls.from_dict(json.loads(serialized))


class NodeManager:
    """Manager for federated learning nodes."""
    
    def __init__(
        self,
        heartbeat_interval: int = 30,
        heartbeat_timeout: int = 60,
        start_heartbeat_monitor: bool = True
    ):
        """
        Initialize the NodeManager.
        
        Args:
            heartbeat_interval: Seconds between heartbeat checks
            heartbeat_timeout: Seconds after which a node is considered offline
            start_heartbeat_monitor: Whether to start monitoring thread automatically
        """
        self.nodes: Dict[str, Node] = {}
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self._stop_monitoring = threading.Event()
        self._monitor_thread = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        
        if start_heartbeat_monitor:
            self.start_heartbeat_monitor()
    
    def register_node(self, node: Node) -> bool:
        """
        Register a new node or update an existing one.
        
        Args:
            node: The node to register
            
        Returns:
            bool: True if registration successful, False otherwise
        """
        with self._lock:
            node.last_heartbeat = time.time()
            self.nodes[node.node_id] = node
            logger.info(f"Node {node.node_id} registered successfully")
            return True
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the system.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            bool: True if successfully unregistered, False if not found
        """
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Node {node_id} unregistered successfully")
                return True
            logger.warning(f"Attempted to unregister non-existent node {node_id}")
            return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node if found, None otherwise
        """
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> List[Node]:
        """
        Get all registered nodes.
        
        Returns:
            List of all nodes
        """
        with self._lock:
            return list(self.nodes.values())
    
    def get_active_nodes(self) -> List[Node]:
        """
        Get all active (online) nodes.
        
        Returns:
            List of nodes with active status
        """
        with self._lock:
            return [
                node for node in self.nodes.values()
                if node.status == NodeStatus.ONLINE
            ]
    
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        Update the status of a node.
        
        Args:
            node_id: ID of the node to update
            status: New status to set
            
        Returns:
            bool: True if update successful, False if node not found
        """
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                logger.info(f"Node {node_id} status updated to {status.value}")
                return True
            logger.warning(f"Attempted to update status of non-existent node {node_id}")
            return False
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """
        Update the last heartbeat time for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            bool: True if update successful, False if node not found
        """
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = time.time()
                return True
            return False
    
    def start_heartbeat_monitor(self) -> None:
        """Start the heartbeat monitoring thread."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            logger.warning("Heartbeat monitor is already running")
            return
            
        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._heartbeat_monitor_loop,
            daemon=True,
            name="HeartbeatMonitor"
        )
        self._monitor_thread.start()
        logger.info("Heartbeat monitor started")
    
    def stop_heartbeat_monitor(self) -> None:
        """Stop the heartbeat monitoring thread."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            logger.warning("Heartbeat monitor is not running")
            return
            
        self._stop_monitoring.set()
        self._monitor_thread.join(timeout=2.0)
        if self._monitor_thread.is_alive():
            logger.warning("Heartbeat monitor did not stop gracefully")
        else:
            logger.info("Heartbeat monitor stopped successfully")
        self._monitor_thread = None
    
    def _heartbeat_monitor_loop(self) -> None:
        """Monitor node heartbeats and update status accordingly."""
        logger.info("Heartbeat monitor loop started")
        
        while not self._stop_monitoring.is_set():
            current_time = time.time()
            offline_nodes = []
            
            with self._lock:
                for node_id, node in self.nodes.items():
                    if (node.status != NodeStatus.OFFLINE and 
                        current_time - node.last_heartbeat > self.heartbeat_timeout):
                        node.status = NodeStatus.OFFLINE
                        offline_nodes.append(node_id)
            
            if offline_nodes:
                logger.warning(f"Nodes marked as offline due to timeout: {offline_nodes}")
                
            # Sleep for the monitoring interval
            self._stop_monitoring.wait(self.heartbeat_interval)
    
    def select_nodes_for_training(self, 
                                 count: Optional[int] = None, 
                                 min_compute_power: Optional[float] = None,
                                 max_compute_power: Optional[float] = None,
                                 required_status: NodeStatus = NodeStatus.ONLINE) -> List[Node]:
        """
        Select nodes for training based on criteria.
        
        Args:
            count: Maximum number of nodes to select (optional)
            min_compute_power: Minimum compute power required (optional)
            max_compute_power: Maximum compute power allowed (optional)
            required_status: Required node status (default: ONLINE)
            
        Returns:
            List of selected nodes meeting the criteria
        """
        with self._lock:
            # Start with nodes that have the required status
            candidates = [
                node for node in self.nodes.values()
                if node.status == required_status
            ]
            
            # Apply compute power filters if specified
            if min_compute_power is not None:
                candidates = [
                    node for node in candidates
                    if node.compute_power >= min_compute_power
                ]
            
            if max_compute_power is not None:
                candidates = [
                    node for node in candidates
                    if node.compute_power <= max_compute_power
                ]
            
            # Sort by compute power in descending order
            candidates.sort(key=lambda node: node.compute_power, reverse=True)
            
            # Limit to specified count if needed
            if count is not None:
                candidates = candidates[:count]
            
            return candidates
    
    def shutdown(self) -> None:
        """Shutdown the node manager and release resources."""
        self.stop_heartbeat_monitor()
        with self._lock:
            self.nodes.clear()
        logger.info("NodeManager shut down successfully")