"""
Spark Manager for distributed computing in the Federated Learning Simulator.

This module provides integration with Apache Spark for large-scale distributed
computations across federated learning nodes. It handles Spark session management,
distributed data processing, and coordination of parallel tasks.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union
import yaml

try:
    import findspark
    findspark.init()
except ImportError:
    pass  # Skip if findspark is not available

try:
    from pyspark import SparkConf, SparkContext
    from pyspark.sql import SparkSession
    from pyspark.ml import Pipeline
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

logger = logging.getLogger(__name__)


class SparkManager:
    """
    Manages Spark sessions and distributed computing operations for the federated learning system.
    
    This class provides a wrapper around Spark functionality, making it easier to:
    - Initialize and configure Spark sessions
    - Distribute datasets and models across a cluster
    - Execute parallel training tasks
    - Aggregate results from distributed nodes
    
    It's designed to work with both classical and quantum models (future integration).
    """
    
    def __init__(self, config_path: Optional[str] = None, app_name: str = "FederatedLearningSimulator"):
        """
        Initialize the SparkManager with optional configuration.
        
        Args:
            config_path: Path to a YAML configuration file for Spark settings
            app_name: Name of the Spark application
        
        Raises:
            ImportError: If PySpark is not available
            RuntimeError: If Spark initialization fails
        """
        self.app_name = app_name
        self.config = self._load_config(config_path)
        self.spark_session = None
        self.spark_context = None
        
        if not SPARK_AVAILABLE:
            logger.warning("PySpark is not installed. SparkManager will operate in local-only mode.")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load Spark configuration from a YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        default_config = {
            "master": "local[*]",
            "executor_memory": "4g",
            "driver_memory": "4g",
            "max_result_size": "2g",
            "shuffle_partitions": 10,
            "serializer": "org.apache.spark.serializer.KryoSerializer",
            "local_dir": "/tmp",
            "log_level": "WARN"
        }
        
        if not config_path:
            logger.info("No Spark config provided, using default settings")
            return default_config
            
        try:
            with open(config_path, 'r') as file:
                user_config = yaml.safe_load(file)
                # Merge with defaults, with user config taking precedence
                return {**default_config, **user_config}
        except Exception as e:
            logger.warning(f"Failed to load Spark config from {config_path}: {e}")
            logger.info("Using default Spark configuration")
            return default_config
    
    def start(self) -> "SparkManager":
        """
        Start the Spark session with configured parameters.
        
        Returns:
            Self for method chaining
            
        Raises:
            RuntimeError: If Spark initialization fails
        """
        if not SPARK_AVAILABLE:
            logger.warning("PySpark not available, running in local-only mode")
            return self
            
        try:
            # Configure Spark
            conf = SparkConf().setAppName(self.app_name)
            conf.setMaster(self.config["master"])
            conf.set("spark.executor.memory", self.config["executor_memory"])
            conf.set("spark.driver.memory", self.config["driver_memory"])
            conf.set("spark.driver.maxResultSize", self.config["max_result_size"])
            conf.set("spark.sql.shuffle.partitions", str(self.config["shuffle_partitions"]))
            conf.set("spark.serializer", self.config["serializer"])
            conf.set("spark.local.dir", self.config["local_dir"])
            
            # Create or get existing session
            self.spark_session = (SparkSession.builder
                                 .config(conf=conf)
                                 .getOrCreate())
            
            self.spark_context = self.spark_session.sparkContext
            self.spark_context.setLogLevel(self.config["log_level"])
            
            logger.info(f"Spark session started with master: {self.config['master']}")
            logger.info(f"Spark UI available at: {self.spark_context.uiWebUrl}")
            
            return self
            
        except Exception as e:
            logger.error(f"Failed to initialize Spark: {e}")
            raise RuntimeError(f"Spark initialization failed: {e}")
    
    def stop(self) -> None:
        """Stop the Spark session and release resources."""
        if self.spark_session:
            logger.info("Stopping Spark session")
            self.spark_session.stop()
            self.spark_session = None
            self.spark_context = None
    
    def parallelize(self, data: List[Any], num_partitions: Optional[int] = None) -> Any:
        """
        Distribute data across the cluster.
        
        Args:
            data: List of data items to distribute
            num_partitions: Number of partitions (if None, Spark decides)
            
        Returns:
            Spark RDD containing the distributed data
            
        Raises:
            RuntimeError: If Spark is not available or initialized
        """
        self._check_spark_available()
        
        if num_partitions:
            return self.spark_context.parallelize(data, num_partitions)
        else:
            return self.spark_context.parallelize(data)
    
    def distribute_training(self, 
                           model_params: Dict[str, Any], 
                           node_data_mapping: Dict[str, Any],
                           train_fn: callable,
                           **kwargs) -> Dict[str, Any]:
        """
        Distribute a training task across multiple nodes.
        
        Args:
            model_params: Parameters of the model to train
            node_data_mapping: Mapping of node IDs to their data
            train_fn: Function that performs training on a single node
            **kwargs: Additional arguments for the training function
            
        Returns:
            Dictionary of node results with trained model parameters
            
        Raises:
            RuntimeError: If Spark is not available or initialized
        """
        self._check_spark_available()
        
        # Convert the data mapping to a list of (node_id, data) tuples
        tasks = list(node_data_mapping.items())
        
        # Distribute the tasks
        distributed_tasks = self.parallelize(tasks)
        
        # Execute training on each node and collect results
        results = (distributed_tasks
                  .map(lambda node_data: (
                      node_data[0],  # node_id
                      train_fn(
                          node_id=node_data[0],
                          data=node_data[1],
                          model_params=model_params,
                          **kwargs
                      )
                  ))
                  .collect())
        
        # Convert results list back to dictionary
        return dict(results)
    
    def aggregate_results(self, results: Dict[str, Any], aggregation_fn: callable) -> Any:
        """
        Aggregate results from multiple nodes.
        
        Args:
            results: Dictionary mapping node IDs to their results
            aggregation_fn: Function to aggregate results
            
        Returns:
            Aggregated result
        """
        return aggregation_fn(results)
    
    def _check_spark_available(self) -> None:
        """
        Check if Spark is available and initialized.
        
        Raises:
            RuntimeError: If Spark is not available or not initialized
        """
        if not SPARK_AVAILABLE:
            raise RuntimeError("PySpark is not installed")
        
        if not self.spark_context:
            raise RuntimeError("Spark session not initialized. Call start() first.")
    
    def get_spark_session(self) -> Optional[SparkSession]:
        """Get the current Spark session."""
        return self.spark_session
    
    def get_spark_context(self) -> Optional[SparkContext]:
        """Get the current Spark context."""
        return self.spark_context
    
    def __enter__(self) -> "SparkManager":
        """Context manager entry - starts Spark session."""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops Spark session."""
        self.stop()

    def configure_dynamic_allocation(self, initial_executors=2, min_executors=1, 
                               max_executors=10, executor_idle_timeout=60):
        """Configure dynamic allocation for the Spark session"""
        if not self.spark_session:
            raise RuntimeError("Spark session not initialized")
            
        # Enable dynamic allocation
        self.spark_session.conf.set("spark.dynamicAllocation.enabled", "true")
        self.spark_session.conf.set("spark.dynamicAllocation.initialExecutors", initial_executors)
        self.spark_session.conf.set("spark.dynamicAllocation.minExecutors", min_executors)
        self.spark_session.conf.set("spark.dynamicAllocation.maxExecutors", max_executors)
        self.spark_session.conf.set("spark.dynamicAllocation.executorIdleTimeout", executor_idle_timeout)
        
        # Enable shuffle service (required for dynamic allocation)
        self.spark_session.conf.set("spark.shuffle.service.enabled", "true")
        
        logger.info(f"Dynamic allocation configured: min={min_executors}, max={max_executors}")
        return self

    # Improved Spark tuning configuration
    def tune_for_ml_workload(self):
        """Configure Spark for machine learning workloads"""
        if not self.spark_session:
            raise RuntimeError("Spark session not initialized")
            
        # Memory management
        self.spark_session.conf.set("spark.memory.fraction", "0.8")  # % of heap for execution and storage
        self.spark_session.conf.set("spark.memory.storageFraction", "0.5")  # % of (spark.memory.fraction) for storage
        
        # Network timeouts for long-running tasks
        self.spark_session.conf.set("spark.network.timeout", "800s")
        self.spark_session.conf.set("spark.executor.heartbeatInterval", "60s")
        
        # Serialization for PyTorch/NumPy
        self.spark_session.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        self.spark_session.conf.set("spark.kryo.registrator", "org.apache.spark.serializer.KryoRegistrator")
        self.spark_session.conf.set("spark.kryoserializer.buffer.max", "512m")
        
        # Speculative execution (start backup tasks for stragglers)
        self.spark_session.conf.set("spark.speculation", "true")
        self.spark_session.conf.set("spark.speculation.interval", "5000ms")
        self.spark_session.conf.set("spark.speculation.multiplier", "2")
        
        logger.info("Spark tuned for ML workload")
        return self