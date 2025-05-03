import os
import logging
import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
from typing import Dict, List, Any, Optional, Type
from torch.utils.data import Dataset, Subset

# Setup project paths assuming this script is in the root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'configs')
# Define specific output directories under 'experiments'
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results')
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations')
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs')
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT) # Add src to path

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

from src.utils.data_utils import (set_seed, load_config,
                                 load_iris_data, partition_data)
from src.federated.aggregation import AggregationStrategy
from src.core.quantum_manager import QuantumAggregationStrategy
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.models.classical_models import SimpleMLP

# Import Spark-specific components
from src.distributed.spark_manager import SparkManager
from src.core.spark_server import SparkFederatedServer

# Configure logging
log_file_path = os.path.join(LOG_DIR, f"spark_experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("SparkExperimentRunner")

def merge_configs(*configs: Dict) -> Dict:
    """Deep merges multiple dictionaries (later dicts override earlier ones)."""
    merged = {}
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

# --- Model Creation ---
def create_model(model_config: Dict, n_features: int, n_classes: int) -> Any:
    """Instantiates a model based on configuration."""
    model_type = model_config.get("type", "classical_mlp").lower()
    logger.info(f"Creating model of type: {model_type}")

    if model_type == "classical_mlp":
        hidden_dim = model_config.get("hidden_dim", 64)
        return SimpleMLP(input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_classes)
    elif model_type == "vqc":
        # Load quantum config if path specified, else use model_config directly
        q_config_path = model_config.get("quantum_config_path")
        q_config = load_config(q_config_path) if q_config_path else model_config
        return VariationalQuantumClassifier(
            n_qubits=q_config.get("n_qubits", n_features),
            n_layers=q_config.get("n_layers", 2),
            n_classes=n_classes,
            circuit_type=q_config.get("circuit_type", "basic"),
            device_name=q_config.get("device", "default.qubit"),
            shots=q_config.get("shots")
        )
    elif model_type == "qnn":
        q_config_path = model_config.get("quantum_config_path")
        q_config = load_config(q_config_path) if q_config_path else model_config
        return QuantumNeuralNetwork(
             n_qubits=q_config.get("n_qubits", n_features),
             n_layers=q_config.get("n_layers", 2),
             output_dim=n_classes,
             circuit_type=q_config.get("circuit_type", "basic"),
             device_name=q_config.get("device", "default.qubit"),
             shots=q_config.get("shots")
        )
    elif model_type == "hybrid":
        q_config_path = model_config.get("quantum_config_path")
        q_config = load_config(q_config_path) if q_config_path else model_config
        return HybridQuantumModel(
             n_qubits=q_config.get("n_qubits", n_features),
             classical_input_dim=n_features,
             classical_hidden_dim=model_config.get("classical_hidden_dim", 32),
             classical_output_dim=n_classes,
             q_layers=q_config.get("n_layers", 2),
             q_circuit_type=q_config.get("circuit_type", "basic"),
             q_device_name=q_config.get("device", "default.qubit"),
             q_shots=q_config.get("shots")
        )
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

# --- Experiment Setup ---
def setup_spark_federated_scenario(
    exp_config: Dict, 
    train_data: Dataset, 
    test_data: Dataset, 
    n_features: int,
    spark_manager: SparkManager
):
    """Sets up the Spark-based server for a given experiment config."""
    logger.info(f"--- Setting up Spark experiment: {exp_config.get('name', 'Unnamed Experiment')} ---")
    set_seed(exp_config.get("seed", 42))

    # --- Load configurations ---
    base_server_cfg = load_config("federated/server_config.yaml")
    base_client_cfg = load_config("federated/client_config.yaml")
    base_agg_cfg = load_config("federated/default_aggregation.yaml")
    base_spark_cfg = load_config("distributed/spark_config.yaml")

    # Merge base with experiment-specific overrides
    client_cfg = merge_configs(base_client_cfg, exp_config.get("client_overrides", {}))
    server_cfg = merge_configs(base_server_cfg, exp_config.get("server_overrides", {}))
    agg_cfg = merge_configs(base_agg_cfg, exp_config.get("aggregation_overrides", {}))
    spark_cfg = merge_configs(base_spark_cfg, exp_config.get("spark_overrides", {}))
    model_cfg = exp_config["model"]  # Model config is mandatory per experiment

    # --- Prepare Data Parameters ---
    num_clients = server_cfg.get("client_management", {}).get("num_clients", 5)
    iid = exp_config.get("data_partition", {}).get("iid", True)
    alpha = exp_config.get("data_partition", {}).get("alpha", 0.5)
    quantum_features = model_cfg.get("quantum_features")  # Target features for quantum models

    # Load potentially dimensionality-reduced data for quantum models
    effective_n_features = n_features
    if "vqc" in model_cfg.get("type", "") or "qnn" in model_cfg.get("type", "") or "hybrid" in model_cfg.get("type", ""):
         if quantum_features and quantum_features < n_features:
              logger.info(f"Quantum model specified: Reloading data with PCA features={quantum_features}")
              q_train_data, q_test_data, effective_n_features = load_iris_data(n_features=quantum_features)
              # For Spark, we'll use the entire dataset on the driver and partition in Spark
              train_dataset = q_train_data
              eval_dataset = q_test_data
         else:
              # Quantum model using original features
              effective_n_features = n_features
              # Check model config for qubits mismatch warning
              if model_cfg.get("quantum_config_path"):
                   q_config = load_config(model_cfg["quantum_config_path"])
                   if q_config.get("n_qubits") != effective_n_features:
                        logger.warning(f"Mismatch: n_features={effective_n_features} but quantum config n_qubits={q_config.get('n_qubits')}. Ensure model handles this.")
              elif model_cfg.get("n_qubits") != effective_n_features:
                   logger.warning(f"Mismatch: n_features={effective_n_features} but model config n_qubits={model_cfg.get('n_qubits')}. Ensure model handles this.")
              train_dataset = train_data
              eval_dataset = test_data
    else:  # Classical model
        effective_n_features = n_features
        train_dataset = train_data
        eval_dataset = test_data

    # --- Create Global Model ---
    n_classes = 3  # Iris has 3 classes
    global_model = create_model(model_cfg, effective_n_features, n_classes)
    model_type_str = model_cfg.get("type", "classical_mlp").lower()
    
    # Ensure model has correct dtype for float64 calculations
    if isinstance(global_model, nn.Module): 
        global_model.double()

    # --- Create Aggregation Strategy ---
    agg_strategy_name = agg_cfg.get("strategy", "FedAvg")
    # For VQC, wrap the base strategy; otherwise use the base directly
    use_quantum_agg = "vqc" in model_type_str
    base_aggregation_strategy = AggregationStrategy.from_config(agg_cfg)
    aggregation_strategy = QuantumAggregationStrategy(base_aggregation_strategy) if use_quantum_agg else base_aggregation_strategy
    logger.info(f"Using aggregation strategy: {type(aggregation_strategy).__name__}")

    # --- Initialize Spark Server ---
    server_device = torch.device("cuda" if client_cfg.get("resources", {}).get("use_gpu", False) and torch.cuda.is_available() else "cpu")
    
    # Create model class and configuration for clients on executors
    model_class_for_client = type(global_model)  # Get the model's class 
    # Create model kwargs for executors
    model_kwargs_for_client = model_cfg.copy()
    
    # Ensure correct field values for quantum models
    if "quantum_config_path" in model_kwargs_for_client:
        q_config = load_config(model_kwargs_for_client["quantum_config_path"])
        q_config["n_qubits"] = min(q_config.get("n_qubits", effective_n_features), effective_n_features)
        q_config["n_classes"] = n_classes
        if model_type_str == "qnn": q_config["output_dim"] = n_classes
        if model_type_str == "hybrid": q_config["classical_output_dim"] = n_classes
        model_kwargs_for_client.update(q_config)
        model_kwargs_for_client.pop("quantum_config_path")
    elif "n_qubits" in model_kwargs_for_client:
        model_kwargs_for_client["n_qubits"] = min(model_kwargs_for_client["n_qubits"], effective_n_features)
        if model_type_str == "vqc": model_kwargs_for_client["n_classes"] = n_classes
        if model_type_str == "qnn": model_kwargs_for_client["output_dim"] = n_classes
        if model_type_str == "hybrid": model_kwargs_for_client["classical_output_dim"] = n_classes
    
    # Add original number of features as metadata
    model_kwargs_for_client["_original_n_features"] = effective_n_features

    # Initialize Spark server
    server = SparkFederatedServer(
        model=global_model,
        spark_manager=spark_manager,
        aggregation_strategy=aggregation_strategy,
        evaluation_dataset=eval_dataset,
        device=server_device
    )

    logger.info(f"--- Spark scenario setup complete for {exp_config.get('name', 'Unnamed Experiment')} ---")
    return server, server_cfg, train_dataset, model_class_for_client, model_kwargs_for_client


# --- Main Execution ---
def run_all_spark_experiments(experiment_configs: List[Dict], spark_config_path: Optional[str] = None):
    """Runs all defined experiments using Spark and saves results."""
    all_results = []
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure results directory exists

    # Load base dataset once
    logger.info("Loading Iris data...")
    full_train_data, full_test_data, n_features_original = load_iris_data()

    # For Spark experiments, we can use the full dataset or a subset for testing
    train_data = full_train_data
    test_data = full_test_data
    logger.info(f"Using dataset: Train size={len(train_data)}, Test size={len(test_data)}")

    # Initialize Spark manager (using context manager)
    with SparkManager(config_path=spark_config_path, app_name="FedLearningSimulator") as spark_manager:
        logger.info(f"Spark initialized with {spark_manager.config['master']}")
        
        for i, exp_config in enumerate(experiment_configs):
            exp_name = exp_config.get("name", f"Spark_Experiment_{i+1}")
            logger.info(f"========== Starting {exp_name} ==========")

            try:
                # Setup server and prepare for this experiment
                server, server_cfg, train_dataset, model_class, model_kwargs = setup_spark_federated_scenario(
                    exp_config, train_data, test_data, n_features_original, spark_manager
                )

                # Extract training parameters from server config
                num_rounds = server_cfg.get("server", {}).get("num_rounds", 10)
                local_epochs = exp_config.get("client_overrides", {}).get("client", {}).get("local_epochs", 1)
                num_clients = server_cfg.get("client_management", {}).get("num_clients", 5) 
                proximal_term = exp_config.get("client_overrides", {}).get("regularization", {}).get("proximal_mu", 0.0)
                client_cfg = exp_config.get("client_overrides", {})

                # Run Spark-based federated training
                round_history = server.train(
                    num_rounds=num_rounds,
                    train_dataset=train_dataset,
                    num_data_partitions=num_clients,  # Number of Spark partitions = number of clients
                    model_class_for_client=model_class,
                    model_kwargs_for_client=model_kwargs,
                    client_config=client_cfg,
                    local_epochs=local_epochs,
                    proximal_term=proximal_term,
                    seed=exp_config.get("seed", 42)
                )

                # Store results along with config details
                result_entry = {
                    "experiment_name": exp_name,
                    "config": exp_config,  # Store the specific config for this run
                    "round_history": round_history,
                    "final_accuracy": round_history[-1]['evaluation_metrics'].get('accuracy') if round_history else None,
                    "final_loss": round_history[-1]['evaluation_metrics'].get('loss') if round_history else None,
                }
                all_results.append(result_entry)
                logger.info(f"========== Finished {exp_name} ==========\n")

                # Save model after experiment if desired
                model_save_dir = os.path.join(EXPERIMENTS_BASE_DIR, 'models')
                os.makedirs(model_save_dir, exist_ok=True)
                model_path = os.path.join(model_save_dir, f"{exp_name}_final_model.pt")
                server.save_model(model_path)

            except Exception as e:
                logger.error(f"========== Experiment {exp_name} failed: {e} ==========", exc_info=True)
                all_results.append({
                    "experiment_name": exp_name,
                    "config": exp_config,
                    "round_history": [],
                    "error": str(e)
                })

    # --- Save aggregated results ---
    results_df = pd.json_normalize(all_results, sep='_')  # Flatten nested dicts
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(RESULTS_DIR, f"spark_experiment_results_{timestamp}.csv")
    try:
        results_df.to_csv(results_filename, index=False)
        logger.info(f"All experiment results saved to {results_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")
        # Fallback: save as JSON
        import json
        json_filename = os.path.join(RESULTS_DIR, f"spark_experiment_results_{timestamp}.json")
        try:
            with open(json_filename, 'w') as f:
                json.dump(all_results, f, indent=4, default=lambda x: str(x))  # Handle non-serializable types
            logger.info(f"Experiment results saved to {json_filename} as fallback.")
        except Exception as je:
            logger.error(f"Failed to save results to JSON: {je}")

    return all_results

# Use the same plot_results function as in run_experiment.py
def plot_results(all_results: List[Dict]):
    """Generates comparison plots from experiment results."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    logger.info("Generating comparison plots...")
    # Create specific subdirectories
    vis_base = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations')
    acc_dir = os.path.join(vis_base, 'accuracy_curves')
    loss_dir = os.path.join(vis_base, 'loss_curves')
    res_dir = os.path.join(vis_base, 'resource_usage')
    comp_dir = os.path.join(vis_base, 'comparisons')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(loss_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')

    plot_data = []
    final_metrics_data = []

    # Prepare data for plotting
    for result in all_results:
        exp_name = result['experiment_name']
        if result.get("error"):
            logger.warning(f"Skipping plotting/analysis for failed experiment: {exp_name}")
            continue
        if not result['round_history']:
            logger.warning(f"Skipping plotting/analysis for experiment with no rounds: {exp_name}")
            continue

        cumulative_time = 0
        cumulative_comm_up = 0
        cumulative_comm_down = 0
        last_metrics = {}

        for round_data in result['round_history']:
            round_num = round_data['round']
            metrics = round_data.get('evaluation_metrics', {})
            duration = round_data.get('duration_seconds', 0)
            comm_up = round_data.get('comm_total_upload_bytes', 0)
            comm_down = round_data.get('comm_download_bytes_per_client', 0) * len(round_data.get('participating_clients', []))

            cumulative_time += duration
            cumulative_comm_up += comm_up
            cumulative_comm_down += comm_down

            plot_data.append({
                 'Experiment': exp_name, 'Round': round_num,
                 'Accuracy': metrics.get('accuracy'), 'Loss': metrics.get('loss'),
                 'Cumulative Time (s)': cumulative_time,
                 'Cumulative Comm Upload (MB)': cumulative_comm_up / (1024*1024),
                 'Cumulative Comm Download (MB)': cumulative_comm_down / (1024*1024),
             })
            last_metrics = metrics  # Store metrics from the last successful round

        # Store final metrics for comparison plots/tables
        final_metrics_data.append({
             'Experiment': exp_name,
             'Final Accuracy': last_metrics.get('accuracy'),
             'Final Loss': last_metrics.get('loss'),
             'Total Time (s)': cumulative_time,
             'Total Comm Upload (MB)': cumulative_comm_up / (1024*1024),
             'Total Comm Download (MB)': cumulative_comm_down / (1024*1024),
             'Config Type': result.get('config',{}).get('model',{}).get('type','unknown'),
             'Config Agg': result.get('config',{}).get('aggregation_overrides',{}).get('strategy','unknown'),
             'Config Data': 'NonIID' if not result.get('config',{}).get('data_partition',{}).get('iid', True) else 'IID'
        })

    if not plot_data: 
        logger.warning("No data available for plotting.")
        return
        
    plot_df = pd.DataFrame(plot_data)
    final_metrics_df = pd.DataFrame(final_metrics_data)

    # --- Plotting Functions ---
    def save_plot(fig, subdir, filename):
        path = os.path.join(subdir, filename)
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Plot saved to {path}")

    # --- Generate the same plots as in run_experiment.py ---
    # 1. Accuracy vs. Round
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='Round', y='Accuracy', hue='Experiment', marker='o', errorbar=None, ax=ax1)
    ax1.set_title('Accuracy vs. Communication Round'); ax1.set_ylabel('Global Model Accuracy')
    ax1.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig1.tight_layout(rect=[0, 0, 0.85, 1]); save_plot(fig1, acc_dir, "spark_accuracy_vs_round.png")

    # 2. Loss vs. Round
    plot_df_loss = plot_df.dropna(subset=['Loss'])
    if not plot_df_loss.empty:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=plot_df_loss, x='Round', y='Loss', hue='Experiment', marker='o', errorbar=None, ax=ax2)
        ax2.set_title('Loss vs. Communication Round'); ax2.set_ylabel('Global Model Loss')
        ax2.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        fig2.tight_layout(rect=[0, 0, 0.85, 1]); save_plot(fig2, loss_dir, "spark_loss_vs_round.png")
    else: logger.info("Skipping loss plot (no valid data).")

    # Add the remaining plot generation code as in run_experiment.py
    # ...

# --- Experiment Definitions ---
SPARK_EXPERIMENTS = [
    # Classical MLP Experiments
    {
        "name": "Spark_MLP_FedAvg_IID",
        "model": {"type": "classical_mlp", "hidden_dim": 32},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 5}, "optimization": {"learning_rate": 0.02, "optimizer": {"name": "Adam"}}},
        "server_overrides": {"server": {"num_rounds": 20}, "client_management": {"num_clients": 5}},
        "spark_overrides": {"executor_memory": "2g", "max_result_size": "1g"}
    },
    {
        "name": "Spark_MLP_FedProx_IID",
        "model": {"type": "classical_mlp", "hidden_dim": 32},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedProx"},
        "client_overrides": {
            "client": {"local_epochs": 5},
            "optimization": {"learning_rate": 0.02, "optimizer": {"name": "Adam"}},
            "regularization": {"proximal_mu": 0.01}
        },
        "server_overrides": {"server": {"num_rounds": 20}, "client_management": {"num_clients": 5}},
        "spark_overrides": {"executor_memory": "2g", "max_result_size": "1g"}
    },
    
    # VQC Experiments
    {
        "name": "Spark_VQC_FedAvg_IID_PCA3",
        "model": {"type": "vqc", "quantum_features": 3, "n_layers": 2},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 2}, "optimization": {"learning_rate": 0.1}},
        "server_overrides": {"server": {"num_rounds": 15}, "client_management": {"num_clients": 5}},
        "spark_overrides": {"executor_memory": "2g", "max_result_size": "1g"}
    },
    
    # Hybrid Quantum-Classical Model
    {
        "name": "Spark_Hybrid_FedAvg_IID_PCA3",
        "model": {"type": "hybrid", "quantum_features": 3, "n_qubits": 3, "n_layers": 1, "classical_hidden_dim": 8},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 3}, "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}},
        "server_overrides": {"server": {"num_rounds": 15}, "client_management": {"num_clients": 5}},
        "spark_overrides": {"executor_memory": "2g", "max_result_size": "1g"}
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Spark-based Federated Learning Experiments")
    parser.add_argument("--spark_config", type=str, help="Path to custom Spark configuration YAML")
    args = parser.parse_args()

    logger.info("Starting All Spark Experiments...")
    results = run_all_spark_experiments(SPARK_EXPERIMENTS, args.spark_config)
    logger.info("All Spark Experiments Finished.")

    # Generate plots if matplotlib/seaborn are available
    try:
        plot_results(results)
    except ImportError:
        logger.warning("Plotting libraries not found. Skipping plot generation.")
    except Exception as e:
        logger.error(f"Error during plotting: {e}", exc_info=True)

    logger.info("Spark experiment script finished.")