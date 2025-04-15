# run_experiment.py
import os
import logging
import argparse
import time
from typing import Dict, List, Any, Optional, Type
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
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
                                 load_mnist_data, partition_data)
from src.federated.aggregation import AggregationStrategy
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.core.quantum_manager import (QuantumFederatedServer, FederatedQuantumManager,
                                      QuantumAggregationStrategy)
from src.core.quantum_client import QuantumFederatedClient
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.models.classical_models import SimpleMLP

# Configure logging
log_file_path = os.path.join(LOG_DIR, f"experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to file in experiments/logs
        logging.StreamHandler(sys.stdout)    # Log to console
    ]
)
logger = logging.getLogger("ExperimentRunner")

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
            n_classes=n_classes, # Ensure output matches task
            circuit_type=q_config.get("circuit_type", "basic"),
            device_name=q_config.get("device", "default.qubit"),
            shots=q_config.get("shots") # Can be None
        )
    elif model_type == "qnn":
        q_config_path = model_config.get("quantum_config_path")
        q_config = load_config(q_config_path) if q_config_path else model_config
        return QuantumNeuralNetwork(
             n_qubits=q_config.get("n_qubits", n_features),
             n_layers=q_config.get("n_layers", 2),
             output_dim=n_classes, # QNN output should match n_classes
             circuit_type=q_config.get("circuit_type", "basic"),
             device_name=q_config.get("device", "default.qubit"),
             shots=q_config.get("shots")
        )
    elif model_type == "hybrid":
        q_config_path = model_config.get("quantum_config_path")
        q_config = load_config(q_config_path) if q_config_path else model_config
        return HybridQuantumModel(
             n_qubits=q_config.get("n_qubits", n_features),
             classical_input_dim=n_features, # Hybrid input matches original features
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
def setup_federated_scenario(exp_config: Dict, train_data: Dataset, test_data: Dataset, n_features: int):
    """Sets up the server and clients for a given experiment config."""
    logger.info(f"--- Setting up experiment: {exp_config.get('name', 'Unnamed Experiment')} ---")
    set_seed(exp_config.get("seed", 42))

    # --- Load configurations ---
    base_server_cfg = load_config("federated/server_config.yaml")
    base_client_cfg = load_config("federated/client_config.yaml")
    base_agg_cfg = load_config("federated/default_aggregation.yaml")

    # Merge base with experiment-specific overrides
    client_cfg = merge_configs(base_client_cfg, exp_config.get("client_overrides", {}))
    server_cfg = merge_configs(base_server_cfg, exp_config.get("server_overrides", {}))
    agg_cfg = merge_configs(base_agg_cfg, exp_config.get("aggregation_overrides", {}))
    model_cfg = exp_config["model"] # Model config is mandatory per experiment

    # --- Prepare Data ---
    num_clients = server_cfg.get("client_management", {}).get("num_clients", 5) # Get num_clients from server config or default
    iid = exp_config.get("data_partition", {}).get("iid", True)
    alpha = exp_config.get("data_partition", {}).get("alpha", 0.5)
    quantum_features = model_cfg.get("quantum_features") # Target features for quantum models

    # Load potentially dimensionality-reduced data for quantum models
    effective_n_features = n_features
    if "vqc" in model_cfg.get("type","") or "qnn" in model_cfg.get("type","") or "hybrid" in model_cfg.get("type",""):
         if quantum_features and quantum_features < n_features:
              logger.info(f"Quantum model specified: Reloading data with PCA features={quantum_features}")
              q_train_data, q_test_data, effective_n_features = load_mnist_data(n_features=quantum_features)
              client_datasets = partition_data(q_train_data, num_clients, iid, alpha)
              eval_dataset = q_test_data
         else:
              # Quantum model using original features (ensure n_qubits matches n_features)
              effective_n_features = n_features
              if model_cfg.get("quantum_config_path"):
                   q_config = load_config(model_cfg["quantum_config_path"])
                   if q_config.get("n_qubits") != effective_n_features:
                        logger.warning(f"Mismatch: n_features={effective_n_features} but quantum config n_qubits={q_config.get('n_qubits')}. Ensure model handles this.")
              elif model_cfg.get("n_qubits") != effective_n_features:
                   logger.warning(f"Mismatch: n_features={effective_n_features} but model config n_qubits={model_cfg.get('n_qubits')}. Ensure model handles this.")
              client_datasets = partition_data(train_data, num_clients, iid, alpha)
              eval_dataset = test_data
    else: # Classical model
        effective_n_features = n_features
        client_datasets = partition_data(train_data, num_clients, iid, alpha)
        eval_dataset = test_data


    # --- Create Global Model ---
    n_classes = 10 # MNIST has 10 classes
    global_model = create_model(model_cfg, effective_n_features, n_classes)
    model_type_str = model_cfg.get("type", "classical_mlp").lower()

    # --- Create Aggregation Strategy ---
    agg_strategy_name = agg_cfg.get("strategy", "FedAvg")
    # For VQC, wrap the base strategy; otherwise use the base directly
    use_quantum_agg = "vqc" in model_type_str
    base_aggregation_strategy = AggregationStrategy.from_config(agg_cfg)
    aggregation_strategy = QuantumAggregationStrategy(base_aggregation_strategy) if use_quantum_agg else base_aggregation_strategy
    logger.info(f"Using aggregation strategy: {type(aggregation_strategy).__name__}")


    # --- Initialize Server ---
    server_device = torch.device("cuda" if client_cfg.get("resources",{}).get("use_gpu", False) and torch.cuda.is_available() else "cpu")
    ServerClass = QuantumFederatedServer if "vqc" in model_type_str or "qnn" in model_type_str or "hybrid" in model_type_str else FederatedServer
    server = ServerClass(
        model=global_model,
        aggregation_strategy=aggregation_strategy,
        evaluation_dataset=eval_dataset,
        device=server_device
    )

    # --- Create Clients ---
    ClientClass = QuantumFederatedClient if "vqc" in model_type_str or "qnn" in model_type_str or "hybrid" in model_type_str else FederatedClient
    client_ids = [f"{model_type_str}_client_{i}" for i in range(num_clients)]
    client_setup_kwargs = {
        "batch_size": client_cfg.get("client", {}).get("batch_size", 32),
        "learning_rate": client_cfg.get("optimization", {}).get("learning_rate", 0.01),
        "device": server_device, # Clients use the same device logic as server
        # Pass optimizer/loss only if model is PyTorch-based
        "optimizer_class": getattr(torch.optim, client_cfg.get("optimization", {}).get("optimizer", {}).get("name", "SGD")) if model_type_str != "vqc" else None,
        "optimizer_kwargs": {k:v for k,v in client_cfg.get("optimization", {}).get("optimizer", {}).items() if k != 'name'} if model_type_str != "vqc" else None,
        "loss_fn": getattr(nn, client_cfg.get("optimization", {}).get("loss_function", "CrossEntropyLoss"))() if model_type_str != "vqc" else None,
    }

    # Ensure quantum models get correct number of features if PCA was used
    client_model_kwargs = model_cfg.copy()
    if "quantum_config_path" in client_model_kwargs: # Load quantum config for detailed args
         q_config = load_config(client_model_kwargs["quantum_config_path"])
         q_config["n_qubits"] = min(q_config.get("n_qubits", effective_n_features), effective_n_features) # Don't use more qubits than features
         # Ensure model uses correct n_classes
         q_config["n_classes"] = n_classes
         if model_type_str == "qnn": q_config["output_dim"] = n_classes
         if model_type_str == "hybrid": q_config["classical_output_dim"] = n_classes
         # Need to pass these potentially modified args back for model creation
         client_model_kwargs.update(q_config) # Update with loaded/modified q_config
         client_model_kwargs.pop("quantum_config_path") # Remove path after loading

    elif "n_qubits" in client_model_kwargs: # If args are directly in model_cfg
        client_model_kwargs["n_qubits"] = min(client_model_kwargs["n_qubits"], effective_n_features)
        if model_type_str == "vqc": client_model_kwargs["n_classes"]=n_classes
        if model_type_str == "qnn": client_model_kwargs["output_dim"]=n_classes
        if model_type_str == "hybrid": client_model_kwargs["classical_output_dim"]=n_classes

    # Create clients using the potentially modified model kwargs
    for i, client_id in enumerate(client_ids):
         client_model_instance = create_model(client_model_kwargs, effective_n_features, n_classes)
         # Set dtype for PyTorch models
         if isinstance(client_model_instance, nn.Module): client_model_instance.double()
         client = ClientClass(client_id=client_id, model=client_model_instance, dataset=client_datasets[i], **client_setup_kwargs)
         server.register_client(client)

    # Ensure server global model also has correct dtype
    if isinstance(server.model, nn.Module): server.model.double()

    logger.info(f"--- Scenario setup complete for {exp_config.get('name', 'Unnamed Experiment')} ---")
    return server, server_cfg # Return server and its config for training params


# --- Main Execution ---
def run_all_experiments(experiment_configs: List[Dict]):
    """Runs all defined experiments and saves results."""
    all_results = []
    os.makedirs(RESULTS_DIR, exist_ok=True) # Ensure results directory exists

    # Load base dataset once
    logger.info("Loading SMALL SUBSET of MNIST data for testing...")
    full_train_data, full_test_data, n_features_original = load_mnist_data()

    subset_train_size = 1000 # e.g., use 1000 samples for training
    subset_test_size = 200  # e.g., use 200 samples for testing
    train_indices = torch.randperm(len(full_train_data))[:subset_train_size]
    test_indices = torch.randperm(len(full_test_data))[:subset_test_size]
    train_data = Subset(full_train_data, train_indices)
    test_data = Subset(full_test_data, test_indices)
    logger.info(f"Using subset: Train size={len(train_data)}, Test size={len(test_data)}")

    for i, exp_config in enumerate(experiment_configs):
        exp_name = exp_config.get("name", f"Experiment_{i+1}")
        logger.info(f"========== Starting {exp_name} ==========")

        try:
            # Setup server and clients for this experiment
            server, server_cfg = setup_federated_scenario(exp_config, train_data, test_data, n_features_original)

            # Extract training parameters from server config
            num_rounds = server_cfg.get("server", {}).get("num_rounds", 10)
            local_epochs = exp_config.get("client_overrides",{}).get("client",{}).get("local_epochs", 1) # Get from client overrides or default
            client_fraction = server_cfg.get("client_management", {}).get("client_fraction", 1.0)
            min_clients = server_cfg.get("client_management", {}).get("min_clients_per_round", 1)
            proximal_term = exp_config.get("client_overrides",{}).get("regularization",{}).get("proximal_mu", 0.0)

            # Run federated training
            round_history = server.train(
                num_rounds=num_rounds,
                local_epochs=local_epochs,
                client_fraction=client_fraction,
                min_clients=min_clients,
                proximal_term=proximal_term
            )

            # Store results along with config details
            result_entry = {
                "experiment_name": exp_name,
                "config": exp_config, # Store the specific config for this run
                "round_history": round_history,
                "final_accuracy": round_history[-1]['evaluation_metrics'].get('accuracy') if round_history else None,
                "final_loss": round_history[-1]['evaluation_metrics'].get('loss') if round_history else None,
            }
            all_results.append(result_entry)
            logger.info(f"========== Finished {exp_name} ==========\n")

        except Exception as e:
            logger.error(f"========== Experiment {exp_name} failed: {e} ==========", exc_info=True)
            all_results.append({
                "experiment_name": exp_name,
                "config": exp_config,
                "round_history": [],
                "error": str(e)
            })

    # --- Save aggregated results ---
    results_df = pd.json_normalize(all_results, sep='_') # Flatten nested dicts
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename = os.path.join(RESULTS_DIR, f"experiment_results_{timestamp}.csv")
    try:
        results_df.to_csv(results_filename, index=False)
        logger.info(f"All experiment results saved to {results_filename}")
    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")
        # Fallback: save as JSON
        import json
        json_filename = os.path.join(RESULTS_DIR, f"experiment_results_{timestamp}.json")
        try:
             with open(json_filename, 'w') as f:
                  json.dump(all_results, f, indent=4, default=lambda x: str(x)) # Handle non-serializable types
             logger.info(f"Experiment results saved to {json_filename} as fallback.")
        except Exception as je:
             logger.error(f"Failed to save results to JSON: {je}")

    return all_results

# --- Visualization ---
def plot_results(all_results: List[Dict]):
    """Generates comparison plots from experiment results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    logger.info("Generating comparison plots...")
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)
    plt.style.use('seaborn-v0_8-darkgrid')

    metrics_to_plot = ['accuracy', 'loss']
    plot_data = []

    # Prepare data for plotting
    for result in all_results:
        exp_name = result['experiment_name']
        if result.get("error"):
            logger.warning(f"Skipping plotting for failed experiment: {exp_name}")
            continue
        for round_data in result['round_history']:
            round_num = round_data['round']
            metrics = round_data['evaluation_metrics']
            if metrics and 'accuracy' in metrics: # Check if metrics exist
                 plot_data.append({
                     'Experiment': exp_name,
                     'Round': round_num,
                     'Accuracy': metrics.get('accuracy'),
                     'Loss': metrics.get('loss') # Might be None for VQC
                 })

    if not plot_data:
         logger.warning("No data available for plotting.")
         return

    plot_df = pd.DataFrame(plot_data)

    # Plot Accuracy vs. Round
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=plot_df, x='Round', y='Accuracy', hue='Experiment', marker='o', errorbar=None) # Use errorbar=None if no std dev
    plt.title('Federated Learning Accuracy Comparison')
    plt.xlabel('Communication Round')
    plt.ylabel('Global Model Accuracy')
    plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend
    plot_filename = os.path.join(RESULTS_DIR, "plots", "accuracy_vs_round.png")
    plt.savefig(plot_filename)
    plt.close()
    logger.info(f"Accuracy plot saved to {plot_filename}")

    # Plot Loss vs. Round (handle None values)
    plot_df_loss = plot_df.dropna(subset=['Loss']) # Only plot where loss exists
    if not plot_df_loss.empty:
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=plot_df_loss, x='Round', y='Loss', hue='Experiment', marker='o', errorbar=None)
        plt.title('Federated Learning Loss Comparison')
        plt.xlabel('Communication Round')
        plt.ylabel('Global Model Loss')
        plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_filename = os.path.join(RESULTS_DIR, "plots", "loss_vs_round.png")
        plt.savefig(plot_filename)
        plt.close()
        logger.info(f"Loss plot saved to {plot_filename}")
    else:
         logger.info("Skipping loss plot as no valid loss data was found (e.g., only VQC runs).")


# --- Experiment Definitions ---
EXPERIMENTS = [
    {
        "name": "Classical_MLP_FedAvg_IID",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
             "client": {"local_epochs": 2},
             "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}
        },
        "server_overrides": {"server": {"num_rounds": 10}} # Fewer rounds for testing
    },
    {
        "name": "VQC_FedAvg_IID_PCA4",
        "model": {"type": "vqc", "quantum_features": 4}, # Use PCA to 4 features
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"}, # Will be wrapped by QuantumAggregationStrategy
        "client_overrides": {
             "client": {"local_epochs": 1}, # VQC training can be slow
             "optimization": {"learning_rate": 0.1} # VQC uses LR directly
        },
        "server_overrides": {"server": {"num_rounds": 5}} # Fewer rounds for VQC
    },
    {
        "name": "QNN_FedAvg_IID_PCA4",
        "model": {"type": "qnn", "quantum_features": 4}, # Use PCA to 4 features
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"}, # Base FedAvg for QNN
        "client_overrides": {
             "client": {"local_epochs": 1},
             "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}
        },
        "server_overrides": {"server": {"num_rounds": 5}}
    },
    {
        "name": "Hybrid_FedAvg_IID_PCA4",
        "model": {"type": "hybrid", "quantum_features": 4, "classical_hidden_dim": 16}, # Use PCA to 4 features
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"}, # Base FedAvg for Hybrid
        "client_overrides": {
             "client": {"local_epochs": 1},
             "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}
        },
        "server_overrides": {"server": {"num_rounds": 5}}
    },
    # --- Add Non-IID experiments ---
    {
        "name": "Classical_MLP_FedAvg_NonIID",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_partition": {"iid": False, "alpha": 0.5}, # Dirichlet Non-IID
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
             "client": {"local_epochs": 2},
             "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}
        },
        "server_overrides": {"server": {"num_rounds": 10}}
    },
     {
        "name": "VQC_FedAvg_NonIID_PCA4",
        "model": {"type": "vqc", "quantum_features": 4},
        "data_partition": {"iid": False, "alpha": 0.5},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 1}, "optimization": {"learning_rate": 0.1}},
        "server_overrides": {"server": {"num_rounds": 5}}
    },
    # Add more experiments (e.g., different aggregation, FedProx, different quantum circuits via config paths)
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    # Add arguments later if needed, e.g., --experiments_file config.json
    args = parser.parse_args()

    logger.info("Starting All Experiments...")
    results = run_all_experiments(EXPERIMENTS)
    logger.info("All Experiments Finished.")

    # Generate plots if matplotlib/seaborn are available
    try:
        plot_results(results)
    except ImportError:
         logger.warning("Plotting libraries not found. Skipping plot generation.")
    except Exception as e:
         logger.error(f"Error during plotting: {e}", exc_info=True)

    logger.info("Script finished.")