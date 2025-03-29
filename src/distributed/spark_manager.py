"""
Spark Manager for Distributed Federated Learning

This module provides a comprehensive interface to Apache Spark for distributed
computation in federated learning scenarios. It handles Spark session management,
data distribution, parallel execution of training tasks, and result collection.
"""

import os
import logging
import yaml
from typing import Dict, Any, List, Optional, Union, Callable
import time
from pathlib import Path

try:
    import findspark
    findspark.init()
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession
    import pyspark.serializers as serializers
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

class SparkManager:
    """
    Manages Apache Spark resources for distributed federated learning.
    
    This class provides high-level functionality for distributing federated
    learning workloads across a Spark cluster, handling serialization/
    deserialization of models, coordinating training, and collecting results.
    
    Attributes:
        config (Dict[str, Any]): Spark configuration parameters
        session (Optional[SparkSession]): Active Spark session
        app_name (str): Name of the Spark application
        logger (logging.Logger): Logger instance for this class
    """
    
    def __init__(self, config_path: Union[str, Path] = None) -> None:
        """
        Initialize a SparkManager with the given configuration.
        
        Args:
            config_path: Path to a YAML configuration file for Spark settings.
                         If None, uses default configuration.
        
        Raises:
            ImportError: If PySpark is not available
            ValueError: If the configuration file cannot be loaded
        """
        self.logger = logging.getLogger(__name__)
        
        if not SPARK_AVAILABLE:
            self.logger.error("PySpark is not available. Install it with 'pip install pyspark'")
            raise ImportError("PySpark is required but not installed")
        
        self.config = self._load_config(config_path)
        self.app_name = self.config.get("app_name", "FederatedLearningSimulator")
        self.session = None
        self.sc = None
        self.logger.info(f"SparkManager initialized with app_name: {self.app_name}")
    
    def _load_config(self, config_path: Union[str, Path] = None) -> Dict[str, Any]:
        """
        Load Spark configuration from a YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration parameters
            
        Raises:
            ValueError: If the configuration file cannot be loaded
        """
        default_config = {
            "app_name": "FederatedLearningSimulator",
            "master": "local[*]",
            "executor_memory": "4g",
            "driver_memory": "4g",
            "serializer": "org.apache.spark.serializer.KryoSerializer",
            "kryo_registrator": "federated.spark.FederatedKryoRegistrator",
            "max_result_size": "2g",
            "executor_cores": 4,
            "network_timeout": 600,
            "ui_port": 4040
        }
        
        if not config_path:
            self.logger.info("Using default Spark configuration")
            return default_config
            
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                # Merge with defaults, with provided config taking precedence
                merged_config = {**default_config, **config}
                self.logger.info(f"Loaded Spark configuration from {config_path}")
                return merged_config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {str(e)}")
            raise ValueError(f"Failed to load Spark configuration: {str(e)}")
    
    def start_session(self) -> SparkSession:
        """
        Start a Spark session with the configured parameters.
        
        Returns:
            Active SparkSession
            
        Raises:
            RuntimeError: If unable to create a Spark session
        """
        if self.session:
            self.logger.info("Returning existing Spark session")
            return self.session
            
        try:
            # Configure Spark
            conf = SparkConf().setAppName(self.app_name)
            conf.setMaster(self.config.get("master", "local[*]"))
            conf.set("spark.executor.memory", self.config.get("executor_memory", "4g"))
            conf.set("spark.driver.memory", self.config.get("driver_memory", "4g"))
            conf.set("spark.serializer", self.config.get("serializer", "org.apache.spark.serializer.KryoSerializer"))
            conf.set("spark.executor.cores", str(self.config.get("executor_cores", 4)))
            conf.set("spark.network.timeout", str(self.config.get("network_timeout", 600)))
            conf.set("spark.ui.port", str(self.config.get("ui_port", 4040)))
            conf.set("spark.driver.maxResultSize", self.config.get("max_result_size", "2g"))
            
            # Add any custom configurations
            for key, value in self.config.items():
                if key.startswith("spark."):
                    conf.set(key, str(value))
            
            # Create and return the session
            self.logger.info(f"Creating new Spark session with app_name: {self.app_name}")
            self.session = SparkSession.builder.config(conf=conf).getOrCreate()
            self.sc = self.session.sparkContext
            
            # Set log level
            log_level = self.config.get("log_level", "ERROR")
            self.sc.setLogLevel(log_level)
            
            self.logger.info(f"Spark session started successfully. UI available at: http://localhost:{self.config.get('ui_port', 4040)}")
            return self.session
            
        except Exception as e:
            self.logger.error(f"Failed to start Spark session: {str(e)}")
            raise RuntimeError(f"Failed to start Spark session: {str(e)}")
    
    def distribute_training(self, node_configs: List[Dict[str, Any]], 
                           training_function: Callable, 
                           *args, **kwargs) -> List[Any]:
        """
        Distribute federated learning training across Spark cluster.
        
        Args:
            node_configs: List of configurations for each federated node
            training_function: Function to be executed on each node
            *args, **kwargs: Additional arguments to pass to the training function
            
        Returns:
            List of results from each node
            
        Raises:
            RuntimeError: If distribution fails
        """
        if not self.session:
            self.start_session()
            
        try:
            # Create RDD from node configs
            self.logger.info(f"Distributing training across {len(node_configs)} nodes")
            node_rdd = self.sc.parallelize(node_configs, len(node_configs))
            
            # Define function to execute on each node
            def execute_training(node_config):
                start_time = time.time()
                try:
                    # Execute the training function with the node config and additional args
                    result = training_function(node_config, *args, **kwargs)
                    execution_time = time.time() - start_time
                    return {
                        "node_id": node_config.get("node_id"),
                        "success": True,
                        "result": result,
                        "execution_time": execution_time
                    }
                except Exception as e:
                    execution_time = time.time() - start_time
                    return {
                        "node_id": node_config.get("node_id"),
                        "success": False,
                        "error": str(e),
                        "execution_time": execution_time
                    }
            
            # Execute training across nodes and collect results
            results = node_rdd.map(execute_training).collect()
            
            # Log summary
            successful = sum(1 for r in results if r["success"])
            self.logger.info(f"Training completed: {successful}/{len(results)} nodes successful")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to distribute training: {str(e)}")
            raise RuntimeError(f"Failed to distribute training: {str(e)}")
    
    def parallelize_data(self, data: List[Any], num_partitions: Optional[int] = None) -> Any:
        """
        Distribute data across the Spark cluster.
        
        Args:
            data: List of data items to distribute
            num_partitions: Number of partitions (defaults to number of executors)
            
        Returns:
            Spark RDD containing the distributed data
        """
        if not self.session:
            self.start_session()
            
        if num_partitions is None:
            # Get the number of executors if available, otherwise use a default value
            try:
                num_partitions = max(int(self.session.conf.get("spark.executor.instances", "2")), 2)
            except ValueError:
                num_partitions = 2
                
        self.logger.info(f"Parallelizing data into {num_partitions} partitions")
        return self.sc.parallelize(data, num_partitions)
    
    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate results from distributed training.
        
        This is a placeholder method that should be extended with specific
        aggregation logic for federated learning models.
        
        Args:
            results: List of results from each node
            
        Returns:
            Aggregated results dictionary
        """
        # Basic aggregation - can be overridden in subclasses
        successful_results = [r["result"] for r in results if r.get("success", False)]
        failed_count = len(results) - len(successful_results)
        
        self.logger.info(f"Aggregating results from {len(successful_results)} successful nodes ({failed_count} failed)")
        
        # This is a simplistic aggregation - real implementation would depend on model type
        aggregated = {
            "num_nodes_total": len(results),
            "num_nodes_successful": len(successful_results),
            "num_nodes_failed": failed_count,
            "execution_times": [r.get("execution_time", 0) for r in results if r.get("success", False)],
            "results": successful_results
        }
        
        return aggregated
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """
        Get status information about the Spark cluster.
        
        Returns:
            Dictionary containing cluster status information
        """
        if not self.session:
            return {"status": "not_started"}
            
        try:
            # Gather cluster information
            cluster_info = {
                "status": "running",
                "app_id": self.sc.applicationId,
                "app_name": self.app_name,
                "master": self.sc.master,
                "executors": len(self.sc._jsc.sc().getExecutorIds()),
                "default_parallelism": self.sc.defaultParallelism,
                "version": self.sc.version
            }
            return cluster_info
        except Exception as e:
            self.logger.error(f"Error getting cluster status: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def stop_session(self) -> None:
        """Stop the active Spark session."""
        if self.session:
            self.logger.info("Stopping Spark session")
            self.session.stop()
            self.session = None
            self.sc = None
        else:
            self.logger.info("No active Spark session to stop")
    
    def __enter__(self):
        """Context manager entry point."""
        self.start_session()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.stop_session()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    try:
        # Initialize SparkManager with default config
        spark_manager = SparkManager()
        
        # Start session and show cluster info
        spark_manager.start_session()
        status = spark_manager.get_cluster_status()
        print(f"Spark Cluster Status: {status}")
        
        # Example of distributing a simple task
        def dummy_training(node_config, factor=1.0):
            """Dummy training function for example purposes"""
            import time
            time.sleep(1)  # Simulate training time
            node_id = node_config.get("node_id", 0)
            return {"node_id": node_id, "result": node_id * factor}
        
        # Create dummy node configs
        node_configs = [{"node_id": i} for i in range(10)]
        
        # Distribute training
        results = spark_manager.distribute_training(node_configs, dummy_training, factor=2.0)
        print(f"Training results: {results}")
        
        # Aggregate results
        aggregated = spark_manager.aggregate_results(results)
        print(f"Aggregated results: {aggregated}")
        
    finally:
        # Ensure clean shutdown
        if 'spark_manager' in locals() and spark_manager.session:
            spark_manager.stop_session()