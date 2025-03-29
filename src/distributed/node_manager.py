"""
Node management for distributed federated learning.

This module provides functionality for managing compute nodes
in a distributed federated learning environment. It handles node
registration, status tracking, and coordination.
"""

import logging
import uuid
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import Enum

from src.distributed.communication import Communication, Message, MessageType

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Status of a federated learning node."""
    IDLE = "idle"
    TRAINING = "training"
    AGGREGATING = "aggregating"
    EVALUATING = "evaluating"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class NodeResources:
    """Resource information for a federated learning node."""
    cpu_cores: int
    memory_mb: int
    gpu_count: Optional[int] = None
    quantum_processor: Optional[str] = None  # For future quantum node integration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert resources to dictionary."""
        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "gpu_count": self.gpu_count,
            "quantum_processor": self.quantum_processor
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeResources':
        """Create resources from dictionary."""
        return cls(
            cpu_cores=data.get("cpu_cores", 1),
            memory_mb=data.get("memory_mb", 1024),
            gpu_count=data.get("gpu_count"),
            quantum_processor=data.get("quantum_processor")
        )


class Node:
    """
    Represents a compute node in the federated learning framework.
    
    This class encapsulates all information and functionality related to 
    a single node in the distributed environment.
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        name: Optional[str] = None,
        address: Optional[str] = None,
        resources: Optional[NodeResources] = None,
        capabilities: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a node.
        
        Args:
            node_id: Unique identifier for the node
            name: Human-readable name for the node
            address: Network address for communication
            resources: Computational resources available on the node
            capabilities: Additional node capabilities
        """
        self.node_id = node_id if node_id else str(uuid.uuid4())
        self.name = name if name else f"node-{self.node_id[:8]}"
        self.address = address
        self.resources = resources or NodeResources(cpu_cores=1, memory_mb=1024)
        self.capabilities = capabilities or {}
        self.status = NodeStatus.IDLE
        self.last_heartbeat = time.time()
        self._metrics = {
            "training_rounds": 0,
            "total_training_time": 0.0,
            "avg_training_time": 0.0,
            "total_data_processed": 0,
            "communication_overhead": 0.0
        }
        
    @property
    def is_active(self) -> bool:
        """Check if node is active based on recent heartbeat."""
        return (time.time() - self.last_heartbeat) < 60.0  # 60 seconds timeout
        
    def update_status(self, new_status: NodeStatus) -> None:
        """Update the node's status."""
        if self.status != new_status:
            logger.info(f"Node {self.name} status changed: {self.status.value} -> {new_status.value}")
            self.status = new_status
            
    def heartbeat(self) -> None:
        """Record a heartbeat from this node."""
        self.last_heartbeat = time.time()
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update node performance metrics."""
        self._metrics.update(metrics)
        
        # Recalculate average training time
        if self._metrics["training_rounds"] > 0:
            self._metrics["avg_training_time"] = (
                self._metrics["total_training_time"] / self._metrics["training_rounds"]
            )
        
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get node metrics."""
        return self._metrics
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "node_id": self.node_id,
            "name": self.name,
            "address": self.address,
            "resources": self.resources.to_dict(),
            "capabilities": self.capabilities,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat,
            "metrics": self._metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create a node from dictionary representation."""
        node = cls(
            node_id=data.get("node_id"),
            name=data.get("name"),
            address=data.get("address"),
            resources=NodeResources.from_dict(data.get("resources", {})),
            capabilities=data.get("capabilities", {})
        )
        
        # Set status
        status_value = data.get("status")
        if status_value:
            try:
                node.status = NodeStatus(status_value)
            except ValueError:
                node.status = NodeStatus.IDLE
                
        # Set other attributes
        node.last_heartbeat = data.get("last_heartbeat", time.time())
        node._metrics = data.get("metrics", node._metrics)
        
        return node


class NodeManager:
    """
    Manages federated learning nodes in a distributed environment.
    
    Provides functionality for node registration, status tracking,
    and coordination for federated learning tasks.
    """
    
    def __init__(
        self, 
        communication: Optional[Communication] = None,
        heartbeat_interval: int = 30
    ):
        """
        Initialize the node manager.
        
        Args:
            communication: Communication instance for node messaging
            heartbeat_interval: Interval in seconds for node heartbeat checks
        """
        self._nodes: Dict[str, Node] = {}
        self._node_lock = threading.RLock()
        self.communication = communication
        self._heartbeat_interval = heartbeat_interval
        self._status_callbacks: Dict[str, List[Callable[[str, NodeStatus], None]]] = {}
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Set up event handlers if communication is provided
        if self.communication:
            self._setup_communication_handlers()
    
    def _setup_communication_handlers(self) -> None:
        """Set up message handlers for node communication."""
        self.communication.register_handler(
            MessageType.NODE_REGISTER, 
            self._handle_node_register
        )
        self.communication.register_handler(
            MessageType.NODE_HEARTBEAT, 
            self._handle_node_heartbeat
        )
        self.communication.register_handler(
            MessageType.NODE_STATUS, 
            self._handle_node_status_update
        )
    
    def start(self) -> None:
        """Start the node manager's background processes."""
        if self._running:
            return
            
        self._running = True
        
        # Start heartbeat monitoring thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_monitor,
            daemon=True
        )
        self._heartbeat_thread.start()
        
        logger.info("NodeManager started")
    
    def stop(self) -> None:
        """Stop the node manager's background processes."""
        self._running = False
        
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
            self._heartbeat_thread = None
            
        logger.info("NodeManager stopped")
    
    def _heartbeat_monitor(self) -> None:
        """Monitor node heartbeats and update status for inactive nodes."""
        while self._running:
            try:
                self._check_node_heartbeats()
                time.sleep(self._heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
    
    def _check_node_heartbeats(self) -> None:
        """Check all nodes for recent heartbeats and update status for inactive nodes."""
        current_time = time.time()
        
        with self._node_lock:
            inactive_nodes = []
            
            for node_id, node in self._nodes.items():
                if (current_time - node.last_heartbeat) > self._heartbeat_interval * 2:
                    if node.status != NodeStatus.DISCONNECTED:
                        inactive_nodes.append(node_id)
            
            # Update status for inactive nodes
            for node_id in inactive_nodes:
                self._nodes[node_id].update_status(NodeStatus.DISCONNECTED)
                self._trigger_status_callbacks(node_id, NodeStatus.DISCONNECTED)
    
    def register_node(self, node: Union[Node, Dict[str, Any]]) -> str:
        """
        Register a node with the manager.
        
        Args:
            node: Node instance or dictionary with node data
            
        Returns:
            node_id: The ID of the registered node
        """
        if isinstance(node, dict):
            node = Node.from_dict(node)
        
        with self._node_lock:
            self._nodes[node.node_id] = node
            logger.info(f"Registered node {node.name} with ID {node.node_id}")
        
        # Notify about the new node
        self._trigger_status_callbacks(node.node_id, node.status)
        
        return node.node_id
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the manager.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            bool: True if the node was unregistered, False if not found
        """
        with self._node_lock:
            if node_id in self._nodes:
                del self._nodes[node_id]
                logger.info(f"Unregistered node with ID {node_id}")
                return True
        
        logger.warning(f"Attempted to unregister non-existent node: {node_id}")
        return False
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """
        Get a node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node: The node if found, None otherwise
        """
        with self._node_lock:
            return self._nodes.get(node_id)
    
    def list_nodes(self, status_filter: Optional[NodeStatus] = None) -> List[Node]:
        """
        List all registered nodes, optionally filtered by status.
        
        Args:
            status_filter: Optional filter for node status
            
        Returns:
            List[Node]: List of nodes matching the filter
        """
        with self._node_lock:
            if status_filter is None:
                return list(self._nodes.values())
            else:
                return [node for node in self._nodes.values() if node.status == status_filter]
    
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """
        Update a node's status.
        
        Args:
            node_id: ID of the node to update
            status: New status for the node
            
        Returns:
            bool: True if successful, False if node not found
        """
        with self._node_lock:
            if node_id in self._nodes:
                self._nodes[node_id].update_status(status)
                # Trigger status callbacks
                self._trigger_status_callbacks(node_id, status)
                return True
        
        logger.warning(f"Attempted to update status of non-existent node: {node_id}")
        return False
    
    def register_status_callback(self, node_id: str, callback: Callable[[str, NodeStatus], None]) -> None:
        """
        Register a callback for node status changes.
        
        Args:
            node_id: ID of the node to monitor
            callback: Function to call when status changes, receives (node_id, new_status)
        """
        if node_id not in self._status_callbacks:
            self._status_callbacks[node_id] = []
        
        self._status_callbacks[node_id].append(callback)
    
    def _trigger_status_callbacks(self, node_id: str, status: NodeStatus) -> None:
        """Trigger all callbacks registered for a node's status."""
        callbacks = self._status_callbacks.get(node_id, [])
        for callback in callbacks:
            try:
                callback(node_id, status)
            except Exception as e:
                logger.error(f"Error in status callback for node {node_id}: {e}")
    
    def _handle_node_register(self, message: Message) -> None:
        """Handle node registration messages."""
        if not message.data:
            logger.warning("Received empty node registration message")
            return
            
        try:
            node_data = message.data
            node_id = self.register_node(node_data)
            
            # Send acknowledgment if communication is available
            if self.communication and message.sender:
                self.communication.send_message(
                    MessageType.NODE_REGISTER_ACK,
                    {"node_id": node_id, "success": True},
                    message.sender
                )
        except Exception as e:
            logger.error(f"Error handling node registration: {e}")
            
            # Send error response
            if self.communication and message.sender:
                self.communication.send_message(
                    MessageType.NODE_REGISTER_ACK,
                    {"success": False, "error": str(e)},
                    message.sender
                )
    
    def _handle_node_heartbeat(self, message: Message) -> None:
        """Handle node heartbeat messages."""
        if not message.data or "node_id" not in message.data:
            logger.warning("Received invalid heartbeat message")
            return
            
        node_id = message.data["node_id"]
        
        with self._node_lock:
            if node_id in self._nodes:
                self._nodes[node_id].heartbeat()
                
                # Update metrics if provided
                if "metrics" in message.data:
                    self._nodes[node_id].update_metrics(message.data["metrics"])
            else:
                logger.warning(f"Received heartbeat from unknown node: {node_id}")
    
    def _handle_node_status_update(self, message: Message) -> None:
        """Handle node status update messages."""
        if not message.data or "node_id" not in message.data or "status" not in message.data:
            logger.warning("Received invalid status update message")
            return
            
        try:
            node_id = message.data["node_id"]
            status = NodeStatus(message.data["status"])
            
            self.update_node_status(node_id, status)
        except (KeyError, ValueError) as e:
            logger.error(f"Error handling status update: {e}")

    def get_active_nodes_count(self) -> int:
        """Get the count of currently active nodes."""
        with self._node_lock:
            return sum(1 for node in self._nodes.values() if node.is_active)