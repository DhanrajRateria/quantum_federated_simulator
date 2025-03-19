"""
Spark Manager Module for Quantum Federated Learning Simulator

This module handles Apache Spark integration for large-scale simulation
of quantum federated learning, allowing the system to scale to hundreds
of virtual quantum nodes running in parallel on a cluster.
"""

import os
import logging
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from typing import Dict, List, Any, Optional, Callable

class SparkManager:
    """
    Manages Spark integration for distributed quantum federated learning simulations.
    Provides an interface for scaling simulations across multiple nodes using Spark.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Spark manager with configuration parameters.
        
        Args:
            config: Dictionary containing Spark configuration parameters
        """
        self.config = config
        self.app_name = config.get("app_name", "quantum_federated_learning")
        self.master = config.get("master", "local[*]")
        self.log_level = config.get("log_level", "WARN")
        self.spark_session = None
        self.spark_context = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """
        Initialize Spark session and context based on configuration.
        """
        try:
            conf = SparkConf().setAppName(self.app_name).setMaster(self.master)
            
            # Add any additional configuration from the config file
            for key, value in self.config.get("spark_properties", {}).items():
                conf = conf.set(key, value)
            
            # Create Spark session
            self.spark_session = SparkSession.builder.config(conf=conf).getOrCreate()
            self.spark_context = self.spark_session.sparkContext
            self.spark_context.setLogLevel(self.log_level)
            
            self.logger.info(f"Spark session initialized. Master: {self.master}, " 
                            f"App name: {self.app_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Spark: {str(e)}")
            raise
    
    def shutdown(self) -> None:
        """
        Shutdown Spark session and clean up resources.
        """
        if self.spark_session:
            self.spark_session.stop()
            self.logger.info("Spark session stopped")
    
    def distribute_training(self, node_configs: List[Dict[str, Any]], 
                           training_function: Callable) -> List[Any]:
        """
        Distribute quantum model training across Spark cluster.
        
        Args:
            node_configs: List of node configuration dictionaries
            training_function: Function to execute on each node
            
        Returns:
            List of results from all nodes
        """
        if not self.spark_context:
            self.initialize()
        
        # Convert node configs to RDD
        node_configs_rdd = self.spark_context.parallelize(node_configs)
        
        # Execute training function on each node
        results = node_configs_rdd.map(training_function).collect()
        self.logger.info(f"Distributed training completed on {len(results)} nodes")
        
        return results
    
    def distribute_evaluation(self, global_model_params: Dict[str, Any], 
                             node_configs: List[Dict[str, Any]],
                             evaluation_function: Callable) -> List[Any]:
        """
        Distribute quantum model evaluation across Spark cluster.
        
        Args:
            global_model_params: Parameters of the global model
            node_configs: List of node configuration dictionaries
            evaluation_function: Function to execute on each node
            
        Returns:
            List of evaluation results from all nodes
        """
        if not self.spark_context:
            self.initialize()
        
        # Broadcast global model parameters
        global_params_bc = self.spark_context.broadcast(global_model_params)
        
        # Convert node configs to RDD
        node_configs_rdd = self.spark_context.parallelize(node_configs)
        
        # Create evaluation tasks by mapping configurations with the global model parameters
        tasks = node_configs_rdd.map(
            lambda config: (config, global_params_bc.value)
        )
        
        # Execute evaluation function on each node
        results = tasks.map(lambda args: evaluation_function(*args)).collect()
        self.logger.info(f"Distributed evaluation completed on {len(results)} nodes")
        
        return results
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Spark cluster.
        
        Returns:
            Dictionary containing cluster statistics
        """
        if not self.spark_context:
            self.initialize()
        
        stats = {
            "number_of_nodes": len(self.spark_context.statusTracker().getExecutorInfos()),
            "active_jobs": len(self.spark_context.statusTracker().getActiveJobIds()),
            "app_id": self.spark_context.applicationId,
            "version": self.spark_context.version
        }
        
        return stats