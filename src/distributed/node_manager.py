"""
Node Manager Module for Quantum Federated Learning Simulator

This module manages the lifecycle and coordination of quantum nodes in the
federated learning network, allowing for dynamic addition or removal of
nodes during training to test resilience.
"""

import logging
import uuid
import threading
import time
from typing import Dict, List, Any, Optional, Callable

class NodeManager:
    """
    Manages the lifecycle and coordination of quantum nodes in the federated network.
    Handles node registration, resource tracking, and dynamic scaling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Node Manager with configuration parameters.
        
        Args:
            config: Dictionary containing node management configuration
        """
        self.config = config
        self.nodes = {}  # Dict of all registered nodes
        self.active_nodes = {}  # Dict of currently active nodes
        self.node_lock = threading.Lock()
        self.max_nodes = config.get("max_nodes", 100)
        self.logger = logging.getLogger(__name__)
        self.node_health_check_interval = config.get("health_check_interval", 60)
        self.health_monitor_active = False
        self.health_monitor_thread = None
        
    def start(self) -> None:
        """
        Start the node manager and health monitoring.
        """
        self.health_monitor_active = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitoring_loop, 
            daemon=True
        )
        self.health_monitor_thread.start()
        self.logger.info("Node manager started")
        
    def stop(self) -> None:
        """
        Stop the node manager and health monitoring.
        """
        self.health_monitor_active = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5.0)
        self.logger.info("Node manager stopped")
        
    def register_node(self, node_info: Dict[str, Any]) -> str:
        """
        Register a new quantum node with the manager.
        
        Args:
            node_info: Dictionary containing node information
                Should include: 
                - node_type: Type of quantum simulator
                - resources: Available quantum resources
                - capabilities: Node capabilities (circuit depth, qubits, etc.)
                
        Returns:
            Node ID for the registered node
        """
        with self.node_lock:
            if len(self.nodes) >= self.max_nodes:
                raise ValueError(f"Maximum node limit reached: {self.max_nodes}")
            
            node_id = str(uuid.uuid4())
            timestamp = time.time()
            
            complete_node_info = {
                "node_id": node_id,
                "registration_time": timestamp,
                "last_seen": timestamp,
                "status": "registered",
                **node_info
            }
            
            self.nodes[node_id] = complete_node_info
            self.logger.info(f"Node registered: {node_id}")
            
            return node_id
    
    def activate_node(self, node_id: str) -> bool:
        """
        Activate a registered node for participation in federated learning.
        
        Args:
            node_id: ID of the node to activate
            
        Returns:
            True if successful, False otherwise
        """
        with self.node_lock:
            if node_id not in self.nodes:
                self.logger.warning(f"Cannot activate unknown node: {node_id}")
                return False
            
            self.nodes[node_id]["status"] = "active"
            self.nodes[node_id]["activation_time"] = time.time()
            self.active_nodes[node_id] = self.nodes[node_id]
            self.logger.info(f"Node activated: {node_id}")
            
            return True
    
    def deactivate_node(self, node_id: str) -> bool:
        """
        Deactivate an active node, removing it from active participation.
        
        Args:
            node_id: ID of the node to deactivate
            
        Returns:
            True if successful, False otherwise
        """
        with self.node_lock:
            if node_id not in self.nodes:
                self.logger.warning(f"Cannot deactivate unknown node: {node_id}")
                return False
            
            self.nodes[node_id]["status"] = "inactive"
            self.nodes[node_id]["deactivation_time"] = time.time()
            if node_id in self.active_nodes:
                del self.active_nodes[node_id]
                self.logger.info(f"Node deactivated: {node_id}")
                return True
            
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """
        Unregister a node from the system.
        
        Args:
            node_id: ID of the node to unregister
            
        Returns:
            True if successful, False otherwise
        """
        with self.node_lock:
            if node_id not in self.nodes:
                return False
            
            # Deactivate if active
            if node_id in self.active_nodes:
                del self.active_nodes[node_id]
            
            # Remove node
            del self.nodes[node_id]
            self.logger.info(f"Node unregistered: {node_id}")
            
            return True
    
    def heartbeat(self, node_id: str) -> bool:
        """
        Update the last seen timestamp for a node.
        
        Args:
            node_id: ID of the node sending heartbeat
            
        Returns:
            True if successful, False otherwise
        """
        with self.node_lock:
            if node_id not in self.nodes:
                return False
            
            self.nodes[node_id]["last_seen"] = time.time()
            return True
    
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Dictionary containing node information or None if not found
        """
        with self.node_lock:
            return self.nodes.get(node_id)
    
    def get_active_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all active nodes.
        
        Returns:
            Dictionary mapping node IDs to node information
        """
        with self.node_lock:
            return self.active_nodes.copy()
    
    def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all registered nodes.
        
        Returns:
            Dictionary mapping node IDs to node information
        """
        with self.node_lock:
            return self.nodes.copy()
    
    def select_nodes_for_round(self, count: int = None, 
                              selection_criteria: Optional[Callable] = None) -> List[str]:
        """
        Select nodes for a federated learning round based on criteria.
        
        Args:
            count: Number of nodes to select (default: all active nodes)
            selection_criteria: Function to filter nodes (default: all active nodes)
            
        Returns:
            List of selected node IDs
        """
        with self.node_lock:
            eligible_nodes = list(self.active_nodes.keys())
            
            # Apply custom selection criteria if provided
            if selection_criteria:
                eligible_nodes = [
                    node_id for node_id in eligible_nodes
                    if selection_criteria(self.active_nodes[node_id])
                ]
            
            # Limit to requested count if specified
            if count is not None and count < len(eligible_nodes):
                # Simple random selection for now
                import random
                selected = random.sample(eligible_nodes, count)
            else:
                selected = eligible_nodes
                
            self.logger.info(f"Selected {len(selected)} nodes for round")
            return selected
    
    def _health_monitoring_loop(self) -> None:
        """
        Background thread for monitoring node health and timeout detection.
        """
        timeout_threshold = self.config.get("node_timeout", 300)  # 5 minutes default
        
        while self.health_monitor_active:
            try:
                current_time = time.time()
                
                with self.node_lock:
                    # Check all active nodes for timeout
                    timed_out_nodes = []
                    
                    for node_id, node_info in list(self.active_nodes.items()):
                        last_seen = node_info.get("last_seen", 0)
                        if current_time - last_seen > timeout_threshold:
                            timed_out_nodes.append(node_id)
                    
                    # Deactivate timed out nodes
                    for node_id in timed_out_nodes:
                        self.nodes[node_id]["status"] = "timeout"
                        del self.active_nodes[node_id]
                        self.logger.warning(f"Node timed out: {node_id}")
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {str(e)}")
            
            # Sleep for the configured interval
            time.sleep(self.node_health_check_interval)