"""
Integration tests for the distributed module.

These tests verify that different components of the distributed system
work together correctly.
"""

import unittest
import time
import threading
import logging
from unittest.mock import MagicMock, patch

from src.distributed import NodeManager, Node, NodeStatus

class TestDistributedIntegration(unittest.TestCase):
    """Test integration between components of the distributed system."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a node manager without starting the heartbeat monitor
        self.node_manager = NodeManager(
            heartbeat_interval=1,
            heartbeat_timeout=2,
            start_heartbeat_monitor=False
        )
        
        # Create test nodes with different capabilities
        self.nodes = [
            Node(
                node_id=f"node-{i}",
                ip_address=f"192.168.1.{i}",
                port=8000 + i,
                status=NodeStatus.ONLINE,
                compute_power=i * 1.0,
                memory=2.0 + i * 0.5
            )
            for i in range(1, 6)  # Create 5 nodes
        ]

    def tearDown(self):
        """Clean up after each test."""
        self.node_manager.shutdown()
        
    def test_node_registration_and_retrieval(self):
        """Test that nodes can be registered and retrieved correctly."""
        # Register a node
        node = self.nodes[0]
        result = self.node_manager.register_node(node)
        self.assertTrue(result)
        
        # Retrieve the node
        retrieved = self.node_manager.get_node(node.node_id)
        self.assertEqual(retrieved.node_id, node.node_id)
        self.assertEqual(retrieved.ip_address, node.ip_address)
        self.assertEqual(retrieved.status, NodeStatus.ONLINE)
        
        # Get all nodes
        all_nodes = self.node_manager.get_all_nodes()
        self.assertEqual(len(all_nodes), 1)
        
    def test_node_status_updates(self):
        """Test updating node status."""
        # Register a node
        node = self.nodes[0]
        self.node_manager.register_node(node)
        
        # Update status
        result = self.node_manager.update_node_status(node.node_id, NodeStatus.TRAINING)
        self.assertTrue(result)
        
        # Verify status was updated
        updated_node = self.node_manager.get_node(node.node_id)
        self.assertEqual(updated_node.status, NodeStatus.TRAINING)
        
    def test_node_serialization(self):
        """Test node serialization and deserialization."""
        original_node = self.nodes[0]
        
        # Serialize
        serialized = original_node.serialize()
        self.assertIsInstance(serialized, str)
        
        # Deserialize
        deserialized = Node.deserialize(serialized)
        
        # Verify all fields match
        self.assertEqual(deserialized.node_id, original_node.node_id)
        self.assertEqual(deserialized.ip_address, original_node.ip_address)
        self.assertEqual(deserialized.port, original_node.port)
        self.assertEqual(deserialized.status, original_node.status)
        self.assertEqual(deserialized.compute_power, original_node.compute_power)
        
    def test_heartbeat_monitoring(self):
        """Test that the heartbeat monitor correctly identifies inactive nodes."""
        # Register nodes
        for node in self.nodes[:2]:
            self.node_manager.register_node(node)
            
        # Manually start the heartbeat monitor
        self.node_manager.start_heartbeat_monitor()
        
        try:
            # Wait for one monitoring interval
            time.sleep(1.5)
            
            # Update heartbeat for the first node only
            self.node_manager.update_node_heartbeat(self.nodes[0].node_id)
            
            # Wait for timeout to occur for the second node
            time.sleep(2.5)
            
            # Check node statuses
            node1 = self.node_manager.get_node(self.nodes[0].node_id)
            node2 = self.node_manager.get_node(self.nodes[1].node_id)
            
            self.assertEqual(node1.status, NodeStatus.ONLINE)
            self.assertEqual(node2.status, NodeStatus.OFFLINE)
        finally:
            # Stop the heartbeat monitor
            self.node_manager.stop_heartbeat_monitor()
            
    def test_node_selection_for_training(self):
        """Test selecting nodes for training based on criteria."""
        # Register nodes
        for node in self.nodes:
            self.node_manager.register_node(node)
            
        # Select top 2 nodes by compute power
        selected = self.node_manager.select_nodes_for_training(count=2)
        self.assertEqual(len(selected), 2)
        self.assertIn("node-5", [n.node_id for n in selected])
        self.assertIn("node-4", [n.node_id for n in selected])
        
        # Select nodes with minimum compute power
        selected = self.node_manager.select_nodes_for_training(min_compute_power=2.5)
        self.assertEqual(len(selected), 3)  # Should be nodes 3, 4, and 5
        
        # Test with both min_compute_power and count
        selected = self.node_manager.select_nodes_for_training(
            min_compute_power=2.5,
            count=2
        )
        self.assertEqual(len(selected), 2)
        self.assertIn("node-5", [n.node_id for n in selected])
        self.assertIn("node-4", [n.node_id for n in selected])


if __name__ == '__main__':
    unittest.main()