# run_feddive_experiment.py
import os
import logging
import argparse
import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sys
import yaml
import json

# --- Setup project paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'configs')
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results_feddive')
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations_feddive')
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs_feddive')

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Imports from your project modules ---
from src.utils.data_utils import (set_seed, load_and_preprocess_data)
from src.federated.utils import FederatedDataset
from src.federated.aggregation import AggregationStrategy
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.models.classical_models import SimpleMLP, SimpleCNN
from src.visualization.integrate_visualizations import enhanced_plot_bc_results as plot_results # Re-use plotting

# --- Configure logging for this experiment script ---
log_file_path = os.path.join(LOG_DIR, f"feddive_experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("FedDive_ExperimentRunner")

# --- Utility Functions (simplified from your script) ---

def load_config(config_path_relative: str) -> Dict[str, Any]:
    full_path = os.path.join(CONFIG_ROOT, config_path_relative)
    if not os.path.exists(full_path):
        logger.error(f"Configuration file not found: {full_path}")
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    with open(full_path, 'r') as file:
        config = yaml.safe_load(file)
    logger.info(f"Loaded configuration from {full_path}")
    return config if config is not None else {}

def merge_configs(*configs: Dict) -> Dict:
    merged = {}
    for config in configs:
        if not isinstance(config, dict): continue
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

# --- Model Creation (Simplified for Classical Models Only) ---
def create_model(model_config: Dict, n_features: int, n_classes: int) -> nn.Module:
    model_type = model_config.get("type", "classical_mlp").lower()
    logger.info(f"Creating model of type: {model_type} with n_features={n_features}, n_classes={n_classes}")

    if model_type == "classical_mlp":
        hidden_dim = model_config.get("hidden_dim", 128)
        model_instance = SimpleMLP(input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_classes)
        model_instance.to(torch.float32) # Standard for classical models
    elif model_type == "cnn":
        # n_features for a CNN is the number of input channels (e.g., 3 for RGB)
        model_instance = SimpleCNN(input_channels=n_features, num_classes=n_classes)
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")
    return model_instance

# --- Custom Noisy Client for Robustness Experiments ---
class NoisyFederatedClient(FederatedClient):
    """A client that adds Gaussian noise to its parameters after training."""
    def __init__(self, noise_level: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level
        logger.info(f"Client {self.client_id} is a NOISY client with noise level: {self.noise_level}")

    def train(self, *args, **kwargs) -> Tuple[Dict[str, torch.Tensor], int]:
        # Perform standard training
        updated_params, num_samples = super().train(*args, **kwargs)

        if self.noise_level > 0:
            logger.warning(f"Client {self.client_id} is adding noise to its parameters.")
            # Add noise to the parameters before returning them
            with torch.no_grad():
                for name in updated_params:
                    noise = torch.randn_like(updated_params[name]) * self.noise_level
                    updated_params[name] += noise
        
        return updated_params, num_samples

# --- Experiment Setup ---
def setup_federated_scenario(exp_config: Dict):
    exp_name = exp_config.get("name", "Unnamed Experiment")
    logger.info(f"--- Setting up experiment: {exp_name} ---")
    set_seed(exp_config.get("seed", 42))
    model_cfg = exp_config["model"]
    model_type_for_data = model_cfg.get("type", "mlp")

    data_cfg = exp_config.get("data_config", {})
    train_data, test_data, n_features, n_classes = load_and_preprocess_data(
        dataset_name=data_cfg.get("name", "iris"),
        n_features_pca=data_cfg.get("pca_features"),
        test_size=data_cfg.get("test_size", 0.2),
        random_state=exp_config.get("seed", 42),
        scaling_strategy=data_cfg.get("scaling", "standard"),
        data_subset_size=data_cfg.get("subset_size"),
        tensor_dtype=torch.float32,
        model_type=model_type_for_data,
    )

    # --- Load and merge configurations ---
    base_client_cfg = load_config("federated/client_config.yaml")
    base_server_cfg = load_config("federated/server_config.yaml")
    base_agg_cfg = load_config("federated/default_aggregation.yaml")

    client_cfg = merge_configs(base_client_cfg, exp_config.get("client_overrides", {}))
    server_cfg = merge_configs(base_server_cfg, exp_config.get("server_overrides", {}))
    agg_cfg = merge_configs(base_agg_cfg, exp_config.get("aggregation_overrides", {}))

    # --- Data Partitioning ---
    num_clients = server_cfg.get("client_management", {}).get("num_clients", 5)
    partition_type = exp_config.get("data_partition", {}).get("type", "iid")
    if partition_type == "iid":
        client_datasets = FederatedDataset.iid_partition(train_data, num_clients)
    elif partition_type == "dirichlet":
        alpha = exp_config.get("data_partition", {}).get("alpha", 0.5)
        client_datasets = FederatedDataset.dirichlet_partition(
            train_data, num_clients, n_classes, alpha
        )
    elif partition_type == "pathological": # NEW CASE
        classes_per_client = exp_config.get("data_partition", {}).get("classes_per_client", 2)
        client_datasets = FederatedDataset.pathological_non_iid_partition(
            train_data, num_clients, classes_per_client
        )
    else:
        raise ValueError(f"Unknown partition type: {partition_type}")
    
    # --- Create Server and Aggregation Strategy ---
    global_model = create_model(model_cfg, n_features, n_classes)
    aggregation_strategy = AggregationStrategy.from_config(agg_cfg)
    server = FederatedServer(model=global_model, aggregation_strategy=aggregation_strategy, evaluation_dataset=test_data)

    # --- Create Clients ---
    client_ids = [f"client_{i}" for i in range(num_clients)]
    for i, client_id in enumerate(client_ids):
        if len(client_datasets[i]) == 0:
            logger.warning(f"Client {client_id} has 0 samples. Skipping client creation.")
            continue
        
        client_model_instance = create_model(model_cfg, n_features, n_classes)
        opt_config = client_cfg.get("optimization", {}).get("optimizer", {})
        opt_class = getattr(torch.optim, opt_config.get("name", "Adam"))
        opt_kwargs = {k: v for k, v in opt_config.items() if k != 'name'}
        loss_fn_instance = getattr(nn, client_cfg.get("optimization", {}).get("loss_function", "CrossEntropyLoss"))()

        client_params = {
            "client_id": client_id,
            "model": client_model_instance,
            "dataset": client_datasets[i],
            "batch_size": client_cfg.get("client", {}).get("batch_size", 32),
            "learning_rate": client_cfg.get("optimization", {}).get("learning_rate", 0.01),
            "optimizer_class": opt_class,
            "optimizer_kwargs": opt_kwargs,
            "loss_fn": loss_fn_instance
        }
        
        # Check if this client should be a noisy one
        if client_cfg.get("client", {}).get("type") == "noisy" and i == 0: # Make the first client noisy
             client_params["noise_level"] = client_cfg.get("client", {}).get("noise_level", 0.1)
             client = NoisyFederatedClient(**client_params)
        else:
             client = FederatedClient(**client_params)
             
        server.register_client(client)

    logger.info(f"--- Scenario setup complete for {exp_name} ---")
    return server, server_cfg

# --- Main Execution Logic ---
def run_experiments(experiment_configs: List[Dict]):
    all_results_summary = []
    for i, exp_config in enumerate(experiment_configs):
        exp_name = exp_config.get("name", f"Experiment_{i+1}")
        logger.info(f"========== Starting: {exp_name} ==========")
        try:
            server, server_runtime_cfg = setup_federated_scenario(exp_config)
            if not server.clients:
                logger.error(f"Experiment {exp_name} has no clients. Aborting.")
                all_results_summary.append({"experiment_name": exp_name, "config": exp_config, "round_history": [], "error": "No clients registered."})
                continue

            num_rounds = server_runtime_cfg.get("server", {}).get("num_rounds", 10)
            local_epochs = exp_config.get("client_overrides", {}).get("client", {}).get("local_epochs", 1)
            proximal_term = exp_config.get("client_overrides", {}).get("regularization", {}).get("proximal_mu", 0.0)
            
            round_history = server.train(
                num_rounds=num_rounds, local_epochs=local_epochs, proximal_term=proximal_term
            )
            final_metrics = round_history[-1]['evaluation_metrics'] if round_history and 'evaluation_metrics' in round_history[-1] else {}
            all_results_summary.append({
                "experiment_name": exp_name, "config": exp_config,
                "round_history": round_history,
                "final_accuracy": final_metrics.get('accuracy'),
                "final_loss": final_metrics.get('loss'),
            })
            logger.info(f"========== Finished {exp_name} ==========\n")
        except Exception as e:
            logger.error(f"========== Experiment {exp_name} failed: {e} ==========", exc_info=True)
            all_results_summary.append({"experiment_name": exp_name, "config": exp_config, "round_history": [], "error": str(e)})

    # --- Save results ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename_csv = os.path.join(RESULTS_DIR, f"feddive_experiment_results_{timestamp}.csv")
    results_filename_json = os.path.join(RESULTS_DIR, f"feddive_experiment_results_{timestamp}.json")
    try:
        # Flatten the complex dictionary for easier CSV export
        flat_results = []
        for res in all_results_summary:
            flat_res = {
                'experiment_name': res['experiment_name'],
                'final_accuracy': res['final_accuracy'],
                'final_loss': res['final_loss'],
                'error': res.get('error'),
                'agg_type': res['config'].get('aggregation_overrides', {}).get('type'),
                'data_set': res['config'].get('data_config', {}).get('name'),
                'data_iid': res['config'].get('data_partition', {}).get('iid'),
                'data_alpha': res['config'].get('data_partition', {}).get('alpha'),
                'feddive_momentum': res['config'].get('aggregation_overrides', {}).get('momentum'),
                'feddive_temp': res['config'].get('aggregation_overrides', {}).get('temperature'),
            }
            flat_results.append(flat_res)
        pd.DataFrame(flat_results).to_csv(results_filename_csv, index=False)
        logger.info(f"All experiment results saved to {results_filename_csv}")
        # Also save the full data as JSON
        with open(results_filename_json, 'w') as f:
            json.dump(all_results_summary, f, indent=4, default=convert_numpy)
        logger.info(f"Full experiment data saved to {results_filename_json}")

    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        
    return all_results_summary

# --- Experiment Definitions ---

EXPERIMENTS_TO_RUN = [
    {
    "name": "FMNIST_Pathological_FedAvg",
    "model": {"type": "classical_mlp", "hidden_dim": 256},
    "data_config": {"name": "fashion-mnist", "scaling": "standard"},
    "data_partition": {"type": "pathological", "classes_per_client": 2}, # Each client gets 2 of 10 classes
    "aggregation_overrides": {"type": "fedavg"},
    "client_overrides": {
        "client": {"local_epochs": 10}, # Train longer to become a true specialist
        "optimization": {"learning_rate": 0.005, "optimizer": {"name": "Adam"}}
    },
    "server_overrides": {"server": {"num_rounds": 50}, "client_management": {"num_clients": 5}} # 5 clients, 2 classes each = all 10 classes covered
},
{
    "name": "FMNIST_Pathological_FedProx",
    "model": {"type": "classical_mlp", "hidden_dim": 256},
    "data_config": {"name": "fashion-mnist", "scaling": "standard"},
    "data_partition": {"type": "pathological", "classes_per_client": 2},
    "aggregation_overrides": {"type": "fedprox"},
    "client_overrides": {
        "client": {"local_epochs": 10},
        "optimization": {"learning_rate": 0.005, "optimizer": {"name": "Adam"}},
        "regularization": {"proximal_mu": 0.01}
    },
    "server_overrides": {"server": {"num_rounds": 50}, "client_management": {"num_clients": 5}}
},
{
    "name": "FMNIST_Pathological_FedDive",
    "model": {"type": "classical_mlp", "hidden_dim": 256},
    "data_config": {"name": "fashion-mnist", "scaling": "standard"},
    "data_partition": {"type": "pathological", "classes_per_client": 2},
    "aggregation_overrides": {"type": "feddive", "momentum": 0.9, "temperature": 1.0}, # Standard temp should work best here
    "client_overrides": {
        "client": {"local_epochs": 10},
        "optimization": {"learning_rate": 0.005, "optimizer": {"name": "Adam"}}
    },
    "server_overrides": {"server": {"num_rounds": 50}, "client_management": {"num_clients": 5}}
}
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments for FedDive")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["breast_cancer", "digits", "all"],
        help="Which dataset's experiments to run."
    )
    args = parser.parse_args()

    experiments_to_run_filtered = EXPERIMENTS_TO_RUN
    if args.dataset != "all":
        experiments_to_run_filtered = [
            exp for exp in EXPERIMENTS_TO_RUN if exp["data_config"]["name"] == args.dataset
        ]
        logger.info(f"Running experiments only for dataset: {args.dataset}")

    logger.info(f"Starting {len(experiments_to_run_filtered)} experiments...")
    results = run_experiments(experiments_to_run_filtered)
    logger.info("All selected experiments finished.")

    if results:
        try:
            logger.info("Generating visualization plots...")
            plot_results(results) # The plotting function from your file is generic enough to be reused
        except Exception as e:
            logger.error(f"Error during plotting: {e}", exc_info=True)
    
    logger.info("Script finished.")

def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError