"""
Metrics Collector Module for Quantum Federated Learning Simulator

This module collects performance metrics during training, gathering detailed
training statistics from each quantum node to analyze system performance.
"""

import logging
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd

class MetricsCollector:
    """
    Collects and stores performance metrics from quantum federated learning experiments.
    
    Responsible for collecting, storing, and preliminary processing of metrics from
    various aspects of the quantum federated learning system including:
    - Training metrics (accuracy, loss)
    - Quantum resource utilization
    - Communication statistics
    - Time performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the metrics collector with configuration parameters.
        
        Args:
            config: Dictionary containing metrics configuration
        """
        self.config = config
        self.output_dir = config.get("metrics_output_dir", "experiments/results")
        self.save_frequency = config.get("save_frequency", 1)  # Save every n rounds
        self.experiment_id = config.get("experiment_id", f"exp_{int(time.time())}")
        
        # Create metrics storage
        self.global_metrics = {
            "rounds": [],
            "global_accuracy": [],
            "global_loss": [],
            "convergence_rate": [],
            "training_time": [],
            "communication_overhead": [],
            "quantum_resource_usage": [],
            "participating_nodes": []
        }
        
        self.node_metrics = {}  # Dictionary to store metrics by node ID
        self.round_metrics = {}  # Dictionary to store metrics by round
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Metrics collector initialized for experiment {self.experiment_id}")

    def start_round(self, round_num: int, participating_nodes: List[str]) -> None:
        """
        Start collecting metrics for a new round.
        
        Args:
            round_num: Current federated learning round number
            participating_nodes: List of participating node IDs
        """
        round_start_time = time.time()
        self.round_metrics[round_num] = {
            "start_time": round_start_time,
            "participating_nodes": participating_nodes,
            "node_metrics": {},
            "communication_stats": {
                "bytes_sent": 0,
                "bytes_received": 0,
                "message_count": 0
            }
        }
        
        self.logger.debug(f"Started metrics collection for round {round_num} with {len(participating_nodes)} nodes")

    def end_round(self, round_num: int, global_metrics: Dict[str, Any]) -> None:
        """
        Finalize metrics collection for a completed round.
        
        Args:
            round_num: Current federated learning round number
            global_metrics: Global model performance metrics
        """
        if round_num not in self.round_metrics:
            self.logger.warning(f"No started metrics found for round {round_num}")
            return
            
        round_end_time = time.time()
        round_data = self.round_metrics[round_num]
        
        # Calculate round duration
        round_duration = round_end_time - round_data["start_time"]
        
        # Update global metrics storage
        self.global_metrics["rounds"].append(round_num)
        self.global_metrics["global_accuracy"].append(global_metrics.get("accuracy", 0.0))
        self.global_metrics["global_loss"].append(global_metrics.get("loss", 0.0))
        self.global_metrics["training_time"].append(round_duration)
        self.global_metrics["participating_nodes"].append(len(round_data["participating_nodes"]))
        
        # Calculate and store communication overhead
        comm_stats = round_data["communication_stats"]
        total_bytes = comm_stats["bytes_sent"] + comm_stats["bytes_received"]
        self.global_metrics["communication_overhead"].append(total_bytes)
        
        # Calculate quantum resource usage
        total_quantum_resources = 0
        for node_id, metrics in round_data["node_metrics"].items():
            total_quantum_resources += metrics.get("quantum_resources", {}).get("total_circuit_depth", 0)
        self.global_metrics["quantum_resource_usage"].append(total_quantum_resources)
        
        # Calculate convergence rate (change in accuracy from previous round)
        if len(self.global_metrics["global_accuracy"]) > 1:
            prev_accuracy = self.global_metrics["global_accuracy"][-2]
            curr_accuracy = self.global_metrics["global_accuracy"][-1]
            convergence_rate = curr_accuracy - prev_accuracy
        else:
            convergence_rate = 0.0
        self.global_metrics["convergence_rate"].append(convergence_rate)
        
        # Save metrics if it's time
        if round_num % self.save_frequency == 0:
            self.save_metrics()
            
        self.logger.info(f"Completed metrics collection for round {round_num}: "
                        f"accuracy={global_metrics.get('accuracy', 0.0):.4f}, "
                        f"loss={global_metrics.get('loss', 0.0):.4f}, "
                        f"duration={round_duration:.2f}s")

    def collect_node_metrics(self, round_num: int, node_id: str, metrics: Dict[str, Any]) -> None:
        """
        Collect metrics from an individual node.
        
        Args:
            round_num: Current federated learning round number
            node_id: ID of the reporting node
            metrics: Dictionary of metrics from the node
        """
        if round_num not in self.round_metrics:
            self.logger.warning(f"No started metrics found for round {round_num}")
            return
            
        # Store node metrics for this round
        self.round_metrics[round_num]["node_metrics"][node_id] = metrics
        
        # Update node-specific metrics history
        if node_id not in self.node_metrics:
            self.node_metrics[node_id] = {
                "rounds": [],
                "accuracy": [],
                "loss": [],
                "training_time": [],
                "quantum_resources": [],
                "parameter_updates": []
            }
            
        self.node_metrics[node_id]["rounds"].append(round_num)
        self.node_metrics[node_id]["accuracy"].append(metrics.get("accuracy", 0.0))
        self.node_metrics[node_id]["loss"].append(metrics.get("loss", 0.0))
        self.node_metrics[node_id]["training_time"].append(metrics.get("training_time", 0.0))
        
        # Extract quantum resource metrics
        quantum_resources = metrics.get("quantum_resources", {})
        self.node_metrics[node_id]["quantum_resources"].append(quantum_resources)
        
        # Extract parameter update statistics
        parameter_updates = metrics.get("parameter_updates", {})
        self.node_metrics[node_id]["parameter_updates"].append(parameter_updates)
        
        self.logger.debug(f"Collected metrics from node {node_id} for round {round_num}: "
                         f"accuracy={metrics.get('accuracy', 0.0):.4f}, "
                         f"loss={metrics.get('loss', 0.0):.4f}")

    def collect_communication_stats(self, round_num: int, stats: Dict[str, Any]) -> None:
        """
        Collect communication statistics.
        
        Args:
            round_num: Current federated learning round number
            stats: Dictionary of communication statistics
        """
        if round_num not in self.round_metrics:
            self.logger.warning(f"No started metrics found for round {round_num}")
            return
            
        # Update communication stats for this round
        comm_stats = self.round_metrics[round_num]["communication_stats"]
        comm_stats["bytes_sent"] += stats.get("bytes_sent", 0)
        comm_stats["bytes_received"] += stats.get("bytes_received", 0)
        comm_stats["message_count"] += stats.get("message_count", 0)
        
        self.logger.debug(f"Updated communication stats for round {round_num}: "
                         f"sent={comm_stats['bytes_sent']/1024:.2f}KB, "
                         f"received={comm_stats['bytes_received']/1024:.2f}KB")

    def get_global_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of global metrics.
        
        Returns:
            Dictionary containing summary metrics
        """
        if not self.global_metrics["rounds"]:
            return {"status": "No metrics collected yet"}
            
        summary = {
            "experiment_id": self.experiment_id,
            "total_rounds": len(self.global_metrics["rounds"]),
            "final_accuracy": self.global_metrics["global_accuracy"][-1],
            "best_accuracy": max(self.global_metrics["global_accuracy"]),
            "average_round_time": np.mean(self.global_metrics["training_time"]),
            "total_training_time": sum(self.global_metrics["training_time"]),
            "total_communication": sum(self.global_metrics["communication_overhead"]),
            "average_nodes_per_round": np.mean(self.global_metrics["participating_nodes"])
        }
        
        return summary

    def get_node_performance_comparison(self) -> Dict[str, Any]:
        """
        Get a comparison of performance across nodes.
        
        Returns:
            Dictionary containing node comparison metrics
        """
        if not self.node_metrics:
            return {"status": "No node metrics collected yet"}
            
        node_comparison = {
            "node_count": len(self.node_metrics),
            "nodes": {},
            "best_performing_node": "",
            "worst_performing_node": ""
        }
        
        # Calculate performance metrics for each node
        best_accuracy = -1.0
        worst_accuracy = float('inf')
        best_node = ""
        worst_node = ""
        
        for node_id, metrics in self.node_metrics.items():
            if not metrics["accuracy"]:
                continue
                
            avg_accuracy = np.mean(metrics["accuracy"])
            final_accuracy = metrics["accuracy"][-1]
            
            node_comparison["nodes"][node_id] = {
                "average_accuracy": avg_accuracy,
                "final_accuracy": final_accuracy,
                "average_training_time": np.mean(metrics["training_time"]),
                "rounds_participated": len(metrics["rounds"])
            }
            
            # Track best and worst performers
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_node = node_id
                
            if final_accuracy < worst_accuracy:
                worst_accuracy = final_accuracy
                worst_node = node_id
        
        node_comparison["best_performing_node"] = best_node
        node_comparison["worst_performing_node"] = worst_node
        
        return node_comparison

    def save_metrics(self) -> str:
        """
        Save all collected metrics to files.
        
        Returns:
            Path to the saved metrics directory
        """
        # Create experiment-specific directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.output_dir, f"{self.experiment_id}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save global metrics
        global_df = pd.DataFrame(self.global_metrics)
        global_path = os.path.join(save_dir, "global_metrics.csv")
        global_df.to_csv(global_path, index=False)
        
        # Save node metrics
        for node_id, metrics in self.node_metrics.items():
            node_df = pd.DataFrame(metrics)
            node_path = os.path.join(save_dir, f"node_{node_id}_metrics.csv")
            node_df.to_csv(node_path, index=False)
        
        # Save round metrics
        round_metrics_path = os.path.join(save_dir, "round_metrics.json")
        with open(round_metrics_path, 'w') as f:
            json.dump(self.round_metrics, f, indent=2)
            
        # Save summary metrics
        summary = self.get_global_metrics_summary()
        summary_path = os.path.join(save_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        self.logger.info(f"Saved metrics to {save_dir}")
        return save_dir

    def load_metrics(self, metrics_dir: str) -> bool:
        """
        Load metrics from saved files.
        
        Args:
            metrics_dir: Path to the directory containing saved metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load global metrics
            global_path = os.path.join(metrics_dir, "global_metrics.csv")
            if os.path.exists(global_path):
                global_df = pd.read_csv(global_path)
                self.global_metrics = global_df.to_dict(orient='list')
            
            # Load round metrics
            round_metrics_path = os.path.join(metrics_dir, "round_metrics.json")
            if os.path.exists(round_metrics_path):
                with open(round_metrics_path, 'r') as f:
                    self.round_metrics = json.load(f)
            
            # Load node metrics - find all node metrics files
            node_files = [f for f in os.listdir(metrics_dir) if f.startswith("node_") and f.endswith("_metrics.csv")]
            self.node_metrics = {}
            
            for node_file in node_files:
                # Extract node ID from filename
                node_id = node_file.replace("node_", "").replace("_metrics.csv", "")
                node_path = os.path.join(metrics_dir, node_file)
                node_df = pd.read_csv(node_path)
                self.node_metrics[node_id] = node_df.to_dict(orient='list')
            
            self.logger.info(f"Loaded metrics from {metrics_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading metrics from {metrics_dir}: {str(e)}")
            return False

    def reset_metrics(self) -> None:
        """
        Reset all collected metrics.
        """
        # Reset global metrics
        for key in self.global_metrics:
            self.global_metrics[key] = []
        
        # Reset node metrics
        self.node_metrics = {}
        
        # Reset round metrics
        self.round_metrics = {}
        
        self.logger.info("Metrics have been reset")