# run_experiments.py

import os
import logging
import argparse
import time
import sys
import json
from typing import Dict, List, Any
from copy import deepcopy

import numpy as np
import torch

# --- Setup project paths and logging ---
# (Assuming your project structure from the original script)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'paper_experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results')
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

log_file_path = os.path.join(LOG_DIR, f"run_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PaperExperimentRunner")

# --- Import your project modules ---
# These are placeholder imports. Replace with your actual module paths.
from src.utils.data_utils import set_seed, load_and_preprocess_data
from src.federated.utils import FederatedDataset
from src.federated.aggregation import AggregationStrategy, FedAvg, FedProx, Median, FedDive, FedDiveR
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient, NoisyFederatedClient # Assume Noisy client is defined
from src.models.classical_models import SimpleMLP

# --- Core Experiment Functions ---
def setup_federated_scenario(config: Dict, seed: int):
    """Sets up the entire federated learning scenario for a single run."""
    set_seed(seed)

    # Load data and create federated partitions
    train_data, test_data, n_features, n_classes = load_and_preprocess_data(dataset_name="digits")
    num_clients = config["federated"]["num_clients"]
    client_datasets = FederatedDataset.dirichlet_partition(
        train_data, num_clients, n_classes, config["data"]["alpha"]
    )

    # Setup Server
    global_model = SimpleMLP(input_dim=n_features, output_dim=n_classes)
    aggregation_strategy = AggregationStrategy.from_config(config["aggregation"])
    server = FederatedServer(
        model=global_model,
        aggregation_strategy=aggregation_strategy,
        evaluation_dataset=test_data
    )

    # Setup Clients
    noisy_client_ids = config.get("federated", {}).get("noisy_clients", {}).get("ids", [])
    for i in range(num_clients):
        client_model = SimpleMLP(input_dim=n_features, output_dim=n_classes)
        client_params = {
            "client_id": f"client_{i}", "model": client_model,
            "dataset": client_datasets[i], "batch_size": 32,
            "learning_rate": 0.01, "optimizer_class": torch.optim.SGD
        }
        if i in noisy_client_ids:
            client = NoisyFederatedClient(noise_level=config["federated"]["noisy_clients"]["noise_level"], **client_params)
        else:
            client = FederatedClient(**client_params)
        server.register_client(client)
        
    return server

def run_single_experiment(config: Dict, seed: int) -> Dict:
    """Runs a single instance of a federated experiment and returns detailed results."""
    logger.info(f"--- Running '{config['name']}' with seed {seed} ---")
    server = setup_federated_scenario(config, seed)
    
    # This history should be detailed, containing per-round global metrics
    # and aggregator state (like client weights from FedDive).
    round_history = server.train(
        num_rounds=config["federated"]["rounds"],
        local_epochs=config["federated"]["local_epochs"]
    )
    
    # This evaluation should return per-client accuracies on their local data.
    fairness_metrics = server.evaluate_fairness()
    
    final_global_metrics = round_history[-1]['evaluation_metrics'] if round_history else {}
    
    # Calculate convergence speed
    target_acc = final_global_metrics.get('accuracy', 0) * 0.95
    rounds_to_95_pct = -1
    for item in round_history:
        if item['evaluation_metrics'].get('accuracy', 0) >= target_acc:
            rounds_to_95_pct = item['round']
            break

    return {
        "seed": seed,
        "final_accuracy": final_global_metrics.get('accuracy'),
        "final_loss": final_global_metrics.get('loss'),
        "convergence_rounds": rounds_to_95_pct,
        "fairness_metrics": fairness_metrics, # e.g., {'client_0': 0.9, 'client_1': 0.85, ...}
        "round_history": round_history # Contains [{round: r, metrics: {...}, agg_state: {...}}, ...]
    }

def run_experiment_suite(config: Dict):
    """Runs an experiment multiple times with different seeds."""
    all_runs_results = []
    num_runs = config.get("num_runs", 3) # Default to 3 runs for statistical significance
    base_seed = config.get("seed", 42)

    for i in range(num_runs):
        seed = base_seed + i
        try:
            run_result = run_single_experiment(config, seed)
            all_runs_results.append(run_result)
        except Exception as e:
            logger.error(f"Experiment '{config['name']}' seed {seed} crashed: {e}", exc_info=True)
            all_runs_results.append({"seed": seed, "error": str(e)})
            
    return {"experiment_config": config, "runs": all_runs_results}

# --- Experiment Definition Generator ---
def generate_all_paper_experiments() -> List[Dict]:
    """Generates the configuration for every experiment needed for the paper."""
    experiments = []
    base_config = {
        "num_runs": 5, "seed": 42,
        "federated": {"num_clients": 10, "rounds": 50, "local_epochs": 5}
    }

    # === Objective 1: IID Baseline ===
    for agg_type in ["fedavg", "feddive"]:
        cfg = deepcopy(base_config)
        cfg.update({
            "name": f"IID_{agg_type.upper()}", "group": "IID",
            "data": {"alpha": 1000.0},
            "aggregation": {"type": agg_type, "momentum": 0.9, "temperature": 1.0},
            "federated": {**cfg["federated"], "rounds": 30}
        })
        experiments.append(cfg)

    # === Objective 2: Non-IID Dominance ===
    for agg_type in ["fedavg", "fedprox", "feddive"]:
        cfg = deepcopy(base_config)
        cfg.update({
            "name": f"NonIID_{agg_type.upper()}", "group": "Non-IID",
            "data": {"alpha": 0.1}
        })
        if agg_type == "fedprox":
            cfg["aggregation"] = {"type": "fedprox", "mu": 0.01}
        elif agg_type == "feddive":
            # Using the optimal temperature found in ablation
            cfg["aggregation"] = {"type": "feddive", "momentum": 0.9, "temperature": 1.5}
        else:
            cfg["aggregation"] = {"type": "fedavg"}
        experiments.append(cfg)

    # === Objective 3: Temperature Ablation ===
    for temp in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0]:
        cfg = deepcopy(base_config)
        cfg.update({
            "name": f"Ablation_Temp_{temp}", "group": "Ablation",
            "data": {"alpha": 0.1},
            "aggregation": {"type": "feddive", "momentum": 0.9, "temperature": temp},
            "num_runs": 3 # Fewer runs for ablation is fine
        })
        experiments.append(cfg)
        
    # === Objective 4: Adversarial Resilience ===
    for agg_type in ["fedavg", "median", "feddive", "feddiver"]:
        cfg = deepcopy(base_config)
        cfg.update({
            "name": f"Robustness_{agg_type.upper()}", "group": "Robustness",
            "data": {"alpha": 0.5},
            "federated": {**cfg["federated"], "rounds": 40, "noisy_clients": {"ids": [0], "noise_level": 10.0}}
        })
        if agg_type == "feddive":
            cfg["aggregation"] = {"type": "feddive", "momentum": 0.9, "temperature": 10.0} # Cautious
        elif agg_type == "feddiver":
            cfg["aggregation"] = {"type": "feddiver", "momentum": 0.9, "temperature": 1.5} # Optimal Temp
        else:
            cfg["aggregation"] = {"type": agg_type}
        experiments.append(cfg)
        
    return experiments

# --- Main Execution Block ---
def main():
    all_configs = generate_all_paper_experiments()
    logger.info(f"Generated {len(all_configs)} experiment configurations for the paper.")
    
    all_results = []
    for config in all_configs:
        suite_results = run_experiment_suite(config)
        all_results.append(suite_results)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = os.path.join(RESULTS_DIR, f"paper_results_detailed_{timestamp}.json")
    with open(json_path, 'w') as f:
        # Custom converter for numpy types
        def json_converter(o):
            if isinstance(o, (np.integer, np.floating, np.bool_)):
                return o.item()
            if isinstance(o, torch.Tensor):
                return o.cpu().numpy().tolist()
            if isinstance(o, (dict, list, str, int, float, bool, type(None))):
                return o
            return str(o)
        json.dump(all_results, f, indent=2, default=json_converter)
        
    logger.info(f"SUCCESS: All experiment results saved to {json_path}")
    logger.info(f"You can now generate plots using: python visualize_results.py {json_path}")

if __name__ == "__main__":
    main()