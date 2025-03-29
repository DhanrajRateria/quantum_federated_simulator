import os
import unittest
import tempfile
import subprocess
import time
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from src.distributed.communication import CommunicationProtocol, MessageType
from src.distributed.node_manager import NodeManager, NodeStatus, NodeType
from src.distributed.spark_manager import SparkManager

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDistributedIntegration(unittest.TestCase):
    """Integration tests for the distributed module components."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment for all tests."""
        # Create temporary directory for test artifacts
        cls.temp_dir = tempfile.TemporaryDirectory()
        
        # Load test configurations
        cls.network_config = {
            "protocol": "grpc",
            "server_address": "localhost:50051",
            "connection_timeout": 30,
            "retry_attempts": 3,
            "buffer_size": 4096,
            "secure": False
        }
        
        cls.spark_config = {
            "app_name": "QFL_Test",
            "master": "local[*]",
            "executor_memory": "2g",
            "driver_memory": "1g",
            "max_result_size": "1g",
            "enable_hive_support": False
        }

        # Test data for classical federated learning
        cls.test_model_params = np.random.rand(100).tolist()
        cls.test_data = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'target': np.random.randint(0, 2, 50)
        })
        
        # Path for saving/loading model parameters
        cls.params_path = os.path.join(cls.temp_dir.name, "model_params.npy")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests are done."""
        cls.temp_dir.cleanup()

    def setUp(self):
        """Set up for each individual test."""
        # Create instances of each component for testing
        self.comm_protocol = CommunicationProtocol(config=self.network_config)
        self.node_manager = NodeManager(config=self.network_config)
        
        # Only initialize SparkManager if Spark is available
        try:
            self.spark_manager = SparkManager(config=self.spark_config)
        except Exception as e:
            logger.warning(f"Spark initialization failed: {e}. Spark tests will be skipped.")
            self.spark_manager = None

    def tearDown(self):
        """Clean up after each individual test."""
        if hasattr(self, 'comm_protocol'):
            self.comm_protocol.close()
        
        if hasattr(self, 'node_manager') and self.node_manager:
            self.node_manager.shutdown()
            
        if hasattr(self, 'spark_manager') and self.spark_manager:
            self.spark_manager.stop_session()

    def test_communication_protocol_messaging(self):
        """Test communication protocol can send and receive messages."""
        # Mock the actual transport layer
        self.comm_protocol._send = MagicMock(return_value=True)
        self.comm_protocol._receive = MagicMock(return_value={"status": "success", "data": self.test_model_params})
        
        # Test sending model parameters
        success = self.comm_protocol.send_message(
            destination="node1",
            message_type=MessageType.MODEL_UPDATE,
            payload=self.test_model_params
        )
        self.assertTrue(success)
        self.comm_protocol._send.assert_called_once()
        
        # Test receiving model parameters
        response = self.comm_protocol.receive_message(source="server")
        self.assertEqual(response["status"], "success")
        self.assertEqual(len(response["data"]), len(self.test_model_params))

    def test_node_manager_lifecycle(self):
        """Test node manager can manage node lifecycle."""
        # Register test nodes
        node1_id = self.node_manager.register_node("node1", NodeType.CLIENT)
        node2_id = self.node_manager.register_node("node2", NodeType.CLIENT)
        server_id = self.node_manager.register_node("server", NodeType.SERVER)
        
        # Verify nodes are registered
        self.assertIn(node1_id, self.node_manager.get_all_nodes())
        self.assertIn(node2_id, self.node_manager.get_all_nodes())
        self.assertIn(server_id, self.node_manager.get_all_nodes())
        
        # Test node status changes
        self.node_manager.set_node_status(node1_id, NodeStatus.TRAINING)
        self.assertEqual(self.node_manager.get_node_status(node1_id), NodeStatus.TRAINING)
        
        self.node_manager.set_node_status(node1_id, NodeStatus.IDLE)
        self.assertEqual(self.node_manager.get_node_status(node1_id), NodeStatus.IDLE)
        
        # Test node deregistration
        self.node_manager.deregister_node(node2_id)
        self.assertNotIn(node2_id, self.node_manager.get_all_nodes())
        
        # Test getting nodes by type
        clients = self.node_manager.get_nodes_by_type(NodeType.CLIENT)
        self.assertIn(node1_id, clients)
        self.assertNotIn(node2_id, clients)
        
        servers = self.node_manager.get_nodes_by_type(NodeType.SERVER)
        self.assertIn(server_id, servers)

    @pytest.mark.skipif(not os.environ.get("SPARK_HOME"), reason="Spark not available")
    def test_spark_manager_initialization(self):
        """Test Spark manager can initialize a Spark session."""
        if not self.spark_manager:
            self.skipTest("Spark manager initialization failed in setup")
            
        # Test if Spark session is active
        self.assertTrue(self.spark_manager.is_active())
        
        # Get Spark session and test simple operation
        spark = self.spark_manager.get_session()
        df = spark.createDataFrame(self.test_data)
        self.assertEqual(df.count(), len(self.test_data))
        
        # Test stopping the session
        self.spark_manager.stop_session()
        self.assertFalse(self.spark_manager.is_active())

    def test_distributed_parameter_exchange(self):
        """Test parameter exchange in a simulated distributed environment."""
        # Mock communication protocol
        with patch('distributed.communication.CommunicationProtocol') as mock_comm:
            comm_instance = mock_comm.return_value
            comm_instance.send_message.return_value = True
            comm_instance.receive_message.return_value = {
                "status": "success",
                "data": self.test_model_params
            }
            
            # Simulate server distributing initial model parameters
            server_node_id = self.node_manager.register_node("server", NodeType.SERVER)
            client1_node_id = self.node_manager.register_node("client1", NodeType.CLIENT)
            client2_node_id = self.node_manager.register_node("client2", NodeType.CLIENT)
            
            # Server sends parameters to clients
            for client_id in [client1_node_id, client2_node_id]:
                success = comm_instance.send_message(
                    destination=client_id,
                    message_type=MessageType.MODEL_DISTRIBUTION,
                    payload=self.test_model_params
                )
                self.assertTrue(success)
            
            # Clients perform local training (simulated)
            for client_id in [client1_node_id, client2_node_id]:
                self.node_manager.set_node_status(client_id, NodeStatus.TRAINING)
                # In a real scenario, clients would train the model here
                time.sleep(0.1)  # Simulate training time
                self.node_manager.set_node_status(client_id, NodeStatus.READY)
            
            # Clients send updated parameters back to server
            for client_id in [client1_node_id, client2_node_id]:
                # Simulate slightly modified parameters after training
                updated_params = [p + np.random.normal(0, 0.01) for p in self.test_model_params]
                success = comm_instance.send_message(
                    destination=server_node_id,
                    message_type=MessageType.MODEL_UPDATE,
                    payload=updated_params
                )
                self.assertTrue(success)
            
            # Server aggregates parameters (not tested here, would be part of federated logic)
            self.node_manager.set_node_status(server_node_id, NodeStatus.AGGREGATING)
            time.sleep(0.1)  # Simulate aggregation time
            self.node_manager.set_node_status(server_node_id, NodeStatus.READY)
            
            # Verify node manager tracked status changes correctly
            self.assertEqual(self.node_manager.get_node_status(server_node_id), NodeStatus.READY)

    def test_fault_tolerance(self):
        """Test system resilience when nodes fail."""
        # Register nodes
        server_id = self.node_manager.register_node("server", NodeType.SERVER)
        node_ids = [
            self.node_manager.register_node(f"client{i}", NodeType.CLIENT)
            for i in range(5)
        ]
        
        # Simulate some nodes failing
        failed_nodes = node_ids[1:3]  # Clients 1 and 2 fail
        for node_id in failed_nodes:
            self.node_manager.set_node_status(node_id, NodeStatus.FAILED)
        
        # Get active clients
        active_clients = self.node_manager.get_active_nodes(NodeType.CLIENT)
        
        # Check that failed nodes are not in active list
        for node_id in failed_nodes:
            self.assertNotIn(node_id, active_clients)
        
        # Check that other nodes are still active
        self.assertEqual(len(active_clients), 3)  # 5 total - 2 failed = 3 active
        
        # Test recovery - bringing a node back online
        recovered_node = failed_nodes[0]
        self.node_manager.set_node_status(recovered_node, NodeStatus.IDLE)
        active_clients = self.node_manager.get_active_nodes(NodeType.CLIENT)
        self.assertIn(recovered_node, active_clients)
        self.assertEqual(len(active_clients), 4)  # 3 active + 1 recovered

    @pytest.mark.skipif(not os.environ.get("SPARK_HOME"), reason="Spark not available")
    def test_distributed_computation(self):
        """Test distributed computation using Spark."""
        if not self.spark_manager:
            self.skipTest("Spark manager initialization failed in setup")
            
        # Create a simple RDD and perform a distributed computation
        spark = self.spark_manager.get_session()
        
        # Create a distributed dataset
        rdd = spark.sparkContext.parallelize(range(100), 4)
        
        # Perform a distributed computation (sum of squares)
        result = rdd.map(lambda x: x * x).sum()
        
        # Verify the result
        expected = sum(i * i for i in range(100))
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()