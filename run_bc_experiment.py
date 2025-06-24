# run_bc_experiment.py
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
import yaml # Import yaml for load_config_exp
import json

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'configs')
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results_bc') # Specific results for BC
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations_bc') # Specific viz for BC
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs_bc') # Specific logs for BC

if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Imports from your project
from src.utils.data_utils import (set_seed, load_and_preprocess_data)
from src.federated.utils import FederatedDataset # For data partitioning
from src.federated.aggregation import AggregationStrategy
from src.federated.server import FederatedServer
from src.federated.client import FederatedClient
from src.core.quantum_manager import (QuantumFederatedServer, QuantumAggregationStrategy)
from src.core.quantum_client import QuantumFederatedClient
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.models.classical_models import SimpleMLP
from src.visualization.integrate_visualizations import enhanced_plot_bc_results

# Configure logging for this specific experiment script
log_file_path = os.path.join(LOG_DIR, f"bc_experiment_run_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BC_ExperimentRunner")


def load_config_exp(config_path_relative: str) -> Dict[str, Any]:
    """Loads config relative to the global CONFIG_ROOT."""
    full_path = os.path.join(CONFIG_ROOT, config_path_relative)
    if not os.path.exists(full_path):
        logger.error(f"Configuration file not found: {full_path}")
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    try:
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {full_path}")
        return config if config is not None else {}
    except Exception as e:
        logger.error(f"Error loading/parsing YAML file {full_path}: {e}", exc_info=True)
        raise

def merge_configs(*configs: Dict) -> Dict:
    merged = {}
    for config in configs:
        if not isinstance(config, dict):
            logger.warning(f"Skipping non-dictionary item in merge_configs: {type(config)}")
            continue
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = merge_configs(merged[key], value)
            else:
                merged[key] = value
    return merged

# --- Model Creation (Identical to the one in the combined script) ---
def create_model(model_config: Dict, n_features: int, n_classes: int, model_dtype: torch.dtype) -> Any:
    model_type = model_config.get("type", "classical_mlp").lower()
    logger.info(f"Creating model of type: {model_type} with n_features={n_features}, n_classes={n_classes}, initial_dtype_request={model_dtype}")

    # Number of qubits for the quantum model part.
    # If n_features (e.g., after PCA) is less than configured n_qubits, use n_features.
    num_qubits_for_model = model_config.get("n_qubits", n_features) # Default to n_features if not specified
    if model_type in ["vqc", "qnn", "hybrid"] and num_qubits_for_model > n_features and n_features > 0 :
        logger.warning(f"Model config n_qubits ({num_qubits_for_model}) > n_features_from_data ({n_features}). Adjusting n_qubits to {n_features}.")
        num_qubits_for_model = n_features
    elif model_type in ["vqc", "qnn", "hybrid"] and n_features == 0: # Should not happen with PCA>0
        logger.error(f"n_features is 0. Cannot create quantum model for type {model_type}.")
        raise ValueError("n_features cannot be 0 for quantum models.")


    model_instance: Optional[nn.Module] = None # Ensure model_instance is defined
    q_cfg = {} # For storing loaded quantum specific config if any
    q_cfg_path = model_config.get("quantum_config_path")
    if q_cfg_path:
        q_cfg = load_config_exp(q_cfg_path) # Load from specified path

    # Merge model_config directly into q_config, with model_config taking precedence for shared keys
    # This allows overriding YAML specifics directly in the experiment definition's model block.
    # For keys only in q_config (from YAML), they remain. For keys in both, model_config wins.
    # For keys only in model_config, they are added.
    final_q_params = {**q_cfg, **model_config}


    if model_type == "classical_mlp":
        hidden_dim = model_config.get("hidden_dim", 64)
        model_instance = SimpleMLP(input_dim=n_features, hidden_dim=hidden_dim, output_dim=n_classes)
    elif model_type == "vqc":
        model_instance = VariationalQuantumClassifier(
            n_qubits=num_qubits_for_model, # This is critical: use the adjusted num_qubits
            n_layers=final_q_params.get("n_layers", 2),
            n_classes=n_classes,
            circuit_type=final_q_params.get("circuit_type", "basic"),
            device_name=final_q_params.get("device", "lightning.qubit"),
            shots=final_q_params.get("shots"), # Can be None
            config_path=q_cfg_path, # Pass the original path if it exists for loading inside VQC if needed
            diff_method=final_q_params.get("diff_method", "best")
        )
    elif model_type == "qnn":
        # QNN output_dim is typically for the quantum layer's direct output.
        # If it's part of a larger network (like Hybrid), this might not be the final n_classes.
        # For standalone QNN for classification, its output_dim should align with n_classes.
        qnn_output_dim = model_config.get("output_dim") # User's desired output from QNN layer
        if qnn_output_dim is None: # If not specified, assume it should output n_classes for classification tasks
            qnn_output_dim = n_classes
            logger.info(f"QNN output_dim not specified, defaulting to n_classes: {n_classes}")

        model_instance = QuantumNeuralNetwork(
             n_qubits=num_qubits_for_model, # Adjusted
             n_layers=final_q_params.get("n_layers", 2),
             output_dim=qnn_output_dim, # This tells QNN what its target output dim is (can be None for auto-infer)
             circuit_type=final_q_params.get("circuit_type", "basic"),
             device_name=final_q_params.get("device", "lightning.qubit"),
             shots=final_q_params.get("shots"),
             config_path=q_cfg_path,
             diff_method=final_q_params.get("diff_method", "best")
        )
    elif model_type == "hybrid":
        model_instance = HybridQuantumModel(
             n_qubits=num_qubits_for_model, # Adjusted
             classical_input_dim=n_features, # Actual features from data
             classical_hidden_dim=model_config.get("classical_hidden_dim", 32),
             classical_output_dim=n_classes, # Final output of the hybrid model
             q_layers=final_q_params.get("n_layers", 2),
             q_circuit_type=final_q_params.get("circuit_type", "basic"),
             q_device_name=final_q_params.get("device", "lightning.qubit"),
             q_shots=final_q_params.get("shots"),
             q_config_path=q_cfg_path,
             q_diff_method=final_q_params.get("diff_method", "best")
        )
    else:
        raise ValueError(f"Unknown model type in config: {model_type}")

    # Dtype handling: PyTorch models usually work with float32.
    # Quantum layers (VQC, QNN inside Hybrid) internally cast to float64 for PennyLane.
    if isinstance(model_instance, nn.Module):
        # For purely classical MLP, set to requested experiment_tensor_dtype (likely float32)
        # For Hybrid, its classical parts will be set. QNN/VQC handle their internal dtype.
        if model_type == "classical_mlp":
            model_instance.to(model_dtype)
            logger.info(f"Model '{model_type}' parameters set to dtype: {model_dtype}")
        # For VQC and QNN, their parameters are managed by TorchLayer, which we set to float64.
        # For Hybrid, its classical parts should match common PyTorch practice (float32 usually)
        # but its internal QNN will handle float64.
        # So, let's ensure the overall Hybrid model's *buffers and non-quantum params* are float32
        # if that's the experiment_tensor_dtype for classical components.
        elif model_type == "hybrid" and model_dtype == torch.float32:
             model_instance.classical_pre.to(torch.float32)
             model_instance.classical_post.to(torch.float32)
             logger.info(f"Hybrid model's classical parts set to float32. Quantum part uses float64 internally.")
        # VQC and standalone QNN are already handled (their TorchLayer params are set to float64)
    return model_instance

# --- Experiment Setup (Identical to the one in the combined script) ---
def setup_federated_scenario(exp_config: Dict,
                             dataset_name_to_load: str, # Explicitly pass dataset name
                             expected_n_classes: int): # For a sanity check
    exp_name = exp_config.get("name", "Unnamed Experiment")
    logger.info(f"--- Setting up experiment: {exp_name} for dataset {dataset_name_to_load} ---")
    set_seed(exp_config.get("seed", 42))

    model_type_str_lc = exp_config.get("model", {}).get("type", "classical_mlp").lower()
    experiment_tensor_dtype = torch.float64 if model_type_str_lc in ["qnn", "hybrid"] else torch.float32
    logger.info(f"Experiment '{exp_name}' will use tensor_dtype: {experiment_tensor_dtype}")

    data_cfg = exp_config.get("data_config", {}) # data_config must exist in experiment definition
    effective_train_data, effective_test_data, effective_n_features, data_n_classes = \
        load_and_preprocess_data(
            dataset_name=dataset_name_to_load, # Use the passed name
            n_features_pca=data_cfg.get("pca_features"),
            test_size=data_cfg.get("test_size", 0.2),
            random_state=exp_config.get("seed", 42),
            scaling_strategy=data_cfg.get("scaling", "standard"),
            data_subset_size=data_cfg.get("subset_size"),
            tensor_dtype=experiment_tensor_dtype
        )

    if expected_n_classes != data_n_classes:
        logger.warning(f"Sanity Check Mismatch: Expected n_classes={expected_n_classes} but loaded data has {data_n_classes}. Using {data_n_classes} from data.")
    current_n_classes = data_n_classes


    base_server_cfg = load_config_exp("federated/server_config.yaml")
    base_client_cfg = load_config_exp("federated/client_config.yaml")
    base_agg_cfg = load_config_exp("federated/default_aggregation.yaml")

    client_cfg = merge_configs(base_client_cfg, exp_config.get("client_overrides", {}))
    server_cfg = merge_configs(base_server_cfg, exp_config.get("server_overrides", {}))
    agg_cfg = merge_configs(base_agg_cfg, exp_config.get("aggregation_overrides", {}))
    model_cfg = exp_config["model"]

    num_clients = server_cfg.get("client_management", {}).get("num_clients", 5)
    iid = exp_config.get("data_partition", {}).get("iid", True)
    alpha = exp_config.get("data_partition", {}).get("alpha", 0.5)
    min_samples_per_client = client_cfg.get("data_partitioning", {}).get("min_samples_per_client", 1)

    if iid:
        client_datasets = FederatedDataset.iid_partition(effective_train_data, num_clients)
    else:
        client_datasets = FederatedDataset.dirichlet_partition(
            effective_train_data, num_clients, current_n_classes,
            alpha
        )
    eval_dataset = effective_test_data

    global_model_config = model_cfg.copy() # Work on a copy for global model
    if global_model_config.get("type") in ["vqc", "qnn", "hybrid"]:
        if "n_qubits" not in global_model_config or global_model_config["n_qubits"] is None: # Ensure n_qubits is set
             global_model_config["n_qubits"] = effective_n_features
        elif global_model_config["n_qubits"] > effective_n_features:
             global_model_config["n_qubits"] = effective_n_features
    if global_model_config.get("type") == "hybrid":
        global_model_config["classical_input_dim"] = effective_n_features

    global_model = create_model(global_model_config, effective_n_features, current_n_classes, experiment_tensor_dtype)

    base_aggregation_strategy = AggregationStrategy.from_config(agg_cfg)
    aggregation_strategy = QuantumAggregationStrategy(base_aggregation_strategy) if model_type_str_lc == "vqc" else base_aggregation_strategy
    logger.info(f"Using aggregation strategy: {type(aggregation_strategy).__name__} (base: {type(base_aggregation_strategy).__name__})")

    server_device_str = "cuda" if client_cfg.get("resources",{}).get("use_gpu", False) and torch.cuda.is_available() else "cpu"
    server_device = torch.device(server_device_str)
    ServerClass = QuantumFederatedServer if model_type_str_lc in ["vqc", "qnn", "hybrid"] else FederatedServer
    server = ServerClass(model=global_model, aggregation_strategy=aggregation_strategy, evaluation_dataset=eval_dataset, device=server_device)
    if isinstance(server.model, nn.Module): server.model.to(server_device)


    ClientClass = QuantumFederatedClient if model_type_str_lc in ["vqc", "qnn", "hybrid"] else FederatedClient
    client_ids = [f"{model_type_str_lc}_client_{i}" for i in range(num_clients)]
    client_model_config_template = model_cfg.copy() # Use this template for all clients
    if client_model_config_template.get("type") in ["vqc", "qnn", "hybrid"]:
        if "n_qubits" not in client_model_config_template or client_model_config_template["n_qubits"] is None:
             client_model_config_template["n_qubits"] = effective_n_features
        elif client_model_config_template["n_qubits"] > effective_n_features:
            client_model_config_template["n_qubits"] = effective_n_features
    if client_model_config_template.get("type") == "hybrid":
        client_model_config_template["classical_input_dim"] = effective_n_features

    clients_for_server = []
    for i, client_id in enumerate(client_ids):
        if len(client_datasets[i]) == 0:
            logger.warning(f"Client {client_id} has 0 samples. Skipping client creation.")
            continue
        client_model_instance = create_model(client_model_config_template, effective_n_features, current_n_classes, experiment_tensor_dtype)
        opt_kwargs, loss_fn_instance, opt_class = None, None, None
        opt_config = client_cfg.get("optimization", {}).get("optimizer", {})
        opt_name = opt_config.get("name", "SGD") # Default to SGD
        try:
            opt_class = getattr(torch.optim, opt_name)
        except AttributeError:
            logger.error(f"Optimizer '{opt_name}' not found in torch.optim. Defaulting to SGD.")
            opt_class = torch.optim.SGD
        # Filter out 'name' from kwargs passed to optimizer constructor
        opt_kwargs = {k:v for k,v in opt_config.items() if k != 'name'}

        loss_name = client_cfg.get("optimization", {}).get("loss_function", "CrossEntropyLoss")
        try:
            loss_fn_instance = getattr(nn, loss_name)()
        except AttributeError:
            logger.error(f"Loss function '{loss_name}' not found in torch.nn. Defaulting to CrossEntropyLoss.")
            loss_fn_instance = nn.CrossEntropyLoss()

        client = ClientClass(
            client_id=client_id, model=client_model_instance, dataset=client_datasets[i],
            batch_size=client_cfg.get("client", {}).get("batch_size", 32),
            learning_rate=client_cfg.get("optimization", {}).get("learning_rate", 0.01),
            optimizer_class=opt_class, # Now passed for VQC too
            optimizer_kwargs=opt_kwargs, # Now passed for VQC too
            loss_fn=loss_fn_instance, # Now passed for VQC too
            device=server_device # Assuming clients run on same device as server for simulation
        )
        clients_for_server.append(client)

    for client_obj in clients_for_server: # Then register them
        server.register_client(client_obj)
    logger.info(f"--- Scenario setup complete for {exp_name} ---")
    return server, server_cfg

# --- Main Execution ---
def run_all_bc_experiments(experiment_configs: List[Dict]):
    all_results_summary = []
    N_CLASSES_BC = 2 # Breast Cancer is binary

    for i, exp_config in enumerate(experiment_configs):
        exp_name = exp_config.get("name", f"BC_Experiment_{i+1}")
        logger.info(f"========== Starting {exp_name} ==========")
        try:
            server, server_runtime_cfg = setup_federated_scenario(
                exp_config,
                dataset_name_to_load="breast_cancer", # Explicitly Breast Cancer
                expected_n_classes=N_CLASSES_BC
            )
            if not server.clients:
                 logger.error(f"Experiment {exp_name} has no clients. Aborting.")
                 all_results_summary.append({"experiment_name": exp_name, "config": exp_config, "round_history": [], "error": "No clients registered."})
                 continue

            num_rounds = server_runtime_cfg.get("server", {}).get("num_rounds", 10)
            local_epochs = exp_config.get("client_overrides",{}).get("client",{}).get("local_epochs", 1)
            client_fraction = server_runtime_cfg.get("client_management", {}).get("client_fraction", 1.0)
            min_clients = server_runtime_cfg.get("client_management", {}).get("min_clients_per_round", 1)
            proximal_term = exp_config.get("client_overrides",{}).get("regularization",{}).get("proximal_mu", 0.0)

            round_history = server.train(
                num_rounds=num_rounds, local_epochs=local_epochs,
                client_fraction=client_fraction, min_clients=min_clients,
                proximal_term=proximal_term
            )
            final_metrics = round_history[-1]['evaluation_metrics'] if round_history and round_history[-1].get('evaluation_metrics') else {}
            result_entry = {
                "experiment_name": exp_name, "config": exp_config,
                "round_history": round_history,
                "final_accuracy": final_metrics.get('accuracy'),
                "final_loss": final_metrics.get('loss'),
            }
            all_results_summary.append(result_entry)
            logger.info(f"========== Finished {exp_name} ==========\n")
        except Exception as e:
            logger.error(f"========== Experiment {exp_name} failed: {e} ==========", exc_info=True)
            all_results_summary.append({"experiment_name": exp_name, "config": exp_config, "round_history": [], "error": str(e)})

    results_df = pd.json_normalize(all_results_summary, sep='_')
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_filename_csv = os.path.join(RESULTS_DIR, f"bc_experiment_results_{timestamp}.csv")
    results_filename_json = os.path.join(RESULTS_DIR, f"bc_experiment_results_{timestamp}.json")
    try:
        results_df.to_csv(results_filename_csv, index=False)
        logger.info(f"All BC experiment results saved to {results_filename_csv}")
    except Exception as e:
        logger.error(f"Failed to save BC results to CSV: {e}. Trying JSON.")
        try:
             with open(results_filename_json, 'w') as f:
                  json.dump(all_results_summary, f, indent=4, default=lambda o: str(o) if isinstance(o, (np.ndarray, np.generic)) else o.__dict__ if hasattr(o, '__dict__') else str(o))
             logger.info(f"BC Experiment results saved to {results_filename_json} as fallback.")
        except Exception as je:
             logger.error(f"Failed to save BC results to JSON: {je}")
    return all_results_summary

# --- Plotting (Identical to the one in the combined script, just ensure paths are correct) ---
def plot_bc_results(all_results: List[Dict]):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # VIS_DIR for bc is already defined globally

    logger.info("Generating comparison plots for Breast Cancer experiments...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plot_data = []
    final_metrics_data = []

    for result in all_results:
        exp_name = result['experiment_name']
        if result.get("error") or not result.get('round_history'):
            logger.warning(f"Skipping plotting for incomplete/failed experiment: {exp_name}")
            continue
        cumulative_time, cumulative_comm_up, cumulative_comm_down, last_metrics = 0,0,0,{}
        for round_data in result['round_history']:
            metrics = round_data.get('evaluation_metrics', {})
            cumulative_time += round_data.get('duration_seconds', 0)
            cumulative_comm_up += round_data.get('comm_total_upload_bytes', 0)
            cumulative_comm_down += round_data.get('comm_download_bytes_per_client', 0) * len(round_data.get('participating_clients', []))
            plot_data.append({
                 'Experiment': exp_name, 'Round': round_data['round'],
                 'Accuracy': metrics.get('accuracy'), 'Loss': metrics.get('loss'),
                 'Cumulative Time (s)': cumulative_time,
                 'Cumulative Comm Upload (MB)': cumulative_comm_up / (1024*1024),
                 'Cumulative Comm Download (MB)': cumulative_comm_down / (1024*1024),
             })
            last_metrics = metrics
        final_metrics_data.append({
             'Experiment': exp_name, 'Final Accuracy': last_metrics.get('accuracy'),
             'Final Loss': last_metrics.get('loss'), 'Total Time (s)': cumulative_time,
             'Total Comm Upload (MB)': cumulative_comm_up / (1024*1024),
             'Total Comm Download (MB)': cumulative_comm_down / (1024*1024),
             'Config Type': result.get('config',{}).get('model',{}).get('type','unknown'),
             'Config Agg': result.get('config',{}).get('aggregation_overrides',{}).get('strategy','FedAvg'), # Default if not found
             'Config Data': 'NonIID' if not result.get('config',{}).get('data_partition',{}).get('iid', True) else 'IID'
        })

    if not plot_data: logger.warning("No data available for BC plotting."); return
    plot_df = pd.DataFrame(plot_data)
    final_metrics_df = pd.DataFrame(final_metrics_data)

    def save_plot(fig, subdir_name, filename):
        # VIS_DIR is global and specific to BC (_bc suffix)
        path = os.path.join(VIS_DIR, subdir_name, filename)
        os.makedirs(os.path.join(VIS_DIR, subdir_name), exist_ok=True)
        fig.savefig(path, bbox_inches='tight'); plt.close(fig)
        logger.info(f"BC Plot saved to {path}")

    if not plot_df.empty:
        fig1, ax1 = plt.subplots(figsize=(12, 7)); sns.lineplot(data=plot_df, x='Round', y='Accuracy', hue='Experiment', marker='o', ax=ax1); ax1.set_title('BC: Accuracy vs. Round'); ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); fig1.tight_layout(rect=[0,0,0.8,1]); save_plot(fig1, "accuracy_curves", "bc_accuracy_vs_round.png")
        plot_df_loss = plot_df.dropna(subset=['Loss'])
        if not plot_df_loss.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 7)); sns.lineplot(data=plot_df_loss, x='Round', y='Loss', hue='Experiment', marker='o', ax=ax2); ax2.set_title('BC: Loss vs. Round'); ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); fig2.tight_layout(rect=[0,0,0.8,1]); save_plot(fig2, "loss_curves", "bc_loss_vs_round.png")
        fig3, ax3 = plt.subplots(figsize=(12, 7)); sns.lineplot(data=plot_df, x='Cumulative Time (s)', y='Accuracy', hue='Experiment', marker='.', ax=ax3); ax3.set_title('BC: Accuracy vs. Time'); ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); fig3.tight_layout(rect=[0,0,0.8,1]); save_plot(fig3, "resource_usage", "bc_accuracy_vs_time.png")
        fig4, ax4 = plt.subplots(figsize=(12, 7)); sns.lineplot(data=plot_df, x='Cumulative Comm Upload (MB)', y='Accuracy', hue='Experiment', marker='.', ax=ax4); ax4.set_title('BC: Accuracy vs. Comm Upload'); ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); fig4.tight_layout(rect=[0,0,0.8,1]); save_plot(fig4, "resource_usage", "bc_accuracy_vs_comm_upload.png")

    if not final_metrics_df.empty:
        fig5, ax5 = plt.subplots(figsize=(10, max(6, len(final_metrics_df)*0.5) )); sns.barplot(data=final_metrics_df, x='Final Accuracy', y='Experiment', ax=ax5, orient='h'); ax5.set_title('BC: Final Accuracy Comparison'); ax5.set_xlim(0,1.05); fig5.tight_layout(); save_plot(fig5, "comparisons", "bc_final_accuracy_comparison.png")
        fig6, ax6 = plt.subplots(figsize=(10, max(6, len(final_metrics_df)*0.5) )); sns.barplot(data=final_metrics_df, x='Total Time (s)', y='Experiment', ax=ax6, orient='h'); ax6.set_title('BC: Total Time Comparison'); fig6.tight_layout(); save_plot(fig6, "comparisons", "bc_total_time_comparison.png")
        fig7, ax7 = plt.subplots(figsize=(10, max(6, len(final_metrics_df)*0.5) )); sns.barplot(data=final_metrics_df, x='Total Comm Upload (MB)', y='Experiment', ax=ax7, orient='h'); ax7.set_title('BC: Total Comm Upload Comparison'); fig7.tight_layout(); save_plot(fig7, "comparisons", "bc_total_comm_comparison.png")


# --- Experiment Definitions for Breast Cancer ---
EXPERIMENTS_BREAST_CANCER = [
    {
        "name": "BC_MLP_FedAvg_IID_FullFeat",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_config": {"name": "breast_cancer", "pca_features": None, "scaling": "standard", "subset_size": None},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 3}, "optimization": {"learning_rate": 0.001, "optimizer": {"name": "Adam"}, "loss_function": "CrossEntropyLoss"}},
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 5}}
    },
    {
        "name": "BC_MLP_FedAvg_NonIID_a0.3_FullFeat",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_config": {"name": "breast_cancer", "pca_features": None, "scaling": "standard"},
        "data_partition": {"iid": False, "alpha": 0.3},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {"client": {"local_epochs": 3}, "optimization": {"learning_rate": 0.001, "optimizer": {"name": "Adam"}, "loss_function": "CrossEntropyLoss"}},
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 5}}
    },
    {
        "name": "BC_MLP_FedProx_NonIID_a0.3_FullFeat",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_config": {"name": "breast_cancer", "pca_features": None, "scaling": "standard"},
        "data_partition": {"iid": False, "alpha": 0.3},
        "aggregation_overrides": {"strategy": "FedProx"},
        "client_overrides": {
            "client": {"local_epochs": 3},
            "optimization": {"learning_rate": 0.001, "optimizer": {"name": "Adam"}, "loss_function": "CrossEntropyLoss"},
            "regularization": {"proximal_mu": 0.01}
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 5}}
    },
    {
        "name": "BC_MLP_FedDive_IID_FullFeat",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_config": {"name": "breast_cancer", "pca_features": None, "scaling": "standard"},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"type": "feddive", "momentum": 0.9, "temperature": 1.0, "normalize_distances": True},
        "client_overrides": {"client": {"local_epochs": 3}, "optimization": {"learning_rate": 0.001, "optimizer": {"name": "Adam"}, "loss_function": "CrossEntropyLoss"}},
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 5}}
    },
    {
        "name": "BC_MLP_FedDive_NonIID_a0.3_FullFeat",
        "model": {"type": "classical_mlp", "hidden_dim": 64},
        "data_config": {"name": "breast_cancer", "pca_features": None, "scaling": "standard"},
        "data_partition": {"iid": False, "alpha": 0.3},
        "aggregation_overrides": {"type": "feddive", "momentum": 0.9, "temperature": 1.0, "normalize_distances": True},
        "client_overrides": {"client": {"local_epochs": 3}, "optimization": {"learning_rate": 0.001, "optimizer": {"name": "Adam"}, "loss_function": "CrossEntropyLoss"}},
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 5}}
    },

    # === VQC Experiments (Updated) ===
    {
        "name": "BC_VQC_FedAvg_IID_PCA4_L2_Basic",
        "model": {
            "type": "vqc", "n_qubits": 4, "n_layers": 2, "circuit_type": "basic",
            "device": "lightning.qubit" # USE FAST SIMULATOR
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"}, # minmax for angle encoding often good
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
            "client": {"local_epochs": 3}, # Can increase if training is faster
            "optimization": {
                "learning_rate": 0.005, # Start with a moderate LR for Adam + quantum
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss" # VQC head now outputs 2 logits
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    {
        "name": "BC_VQC_FedAvg_NonIID_a0.5_PCA4_L2_Complex",
        "model": {
            "type": "vqc", "n_qubits": 4, "n_layers": 2, "circuit_type": "complex",
            "device": "lightning.qubit"
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"},
        "data_partition": {"iid": False, "alpha": 0.5},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
            "client": {"local_epochs": 3},
            "optimization": {
                "learning_rate": 0.005,
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss"
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    {
        "name": "BC_VQC_FedProx_IID_PCA4_L2_Basic", # FedProx example
        "model": {
            "type": "vqc", "n_qubits": 4, "n_layers": 2, "circuit_type": "basic",
            "device": "lightning.qubit"
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedProx"},
        "client_overrides": {
            "client": {"local_epochs": 3},
            "optimization": {
                "learning_rate": 0.005,
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss"
            },
            "regularization": {"proximal_mu": 0.01}
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    {
        "name": "BC_VQC_FedDive_IID_PCA4_L2_Basic",
        "model": {
            "type": "vqc", "n_qubits": 4, "n_layers": 2, "circuit_type": "basic",
            "device": "lightning.qubit" # USE FAST SIMULATOR
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"}, # minmax for angle encoding often good
        "data_partition": {"iid": True},
        "aggregation_overrides": {"type": "feddive", "momentum": 0.9, "temperature": 1.0, "normalize_distances": True},
        "client_overrides": {
            "client": {"local_epochs": 3}, # Can increase if training is faster
            "optimization": {
                "learning_rate": 0.005, # Start with a moderate LR for Adam + quantum
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss" # VQC head now outputs 2 logits
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    {
        "name": "BC_VQC_FedDive_NonIID_a0.5_PCA4_L2_Complex",
        "model": {
            "type": "vqc", "n_qubits": 4, "n_layers": 2, "circuit_type": "complex",
            "device": "lightning.qubit"
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"},
        "data_partition": {"iid": False, "alpha": 0.5},
        "aggregation_overrides": {"type": "feddive", "momentum": 0.9, "temperature": 1.0, "normalize_distances": True},
        "client_overrides": {
            "client": {"local_epochs": 3},
            "optimization": {
                "learning_rate": 0.005,
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss"
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    # === Hybrid Model Experiments (Updated) ===
    {
        "name": "BC_Hybrid_FedAvg_IID_PCA8_Q4L2_Basic",
        "model": {
            "type": "hybrid", "classical_hidden_dim": 16, 
            "n_qubits": 4, "q_layers": 2, "q_circuit_type": "basic",
            "q_device_name": "lightning.qubit" # Note: Hybrid model takes q_device_name
        },
        "data_config": {"name": "breast_cancer", "pca_features": 8, "scaling": "standard"},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
            "client": {"local_epochs": 3}, 
            "optimization": {
                "learning_rate": 0.001, # Hybrid might need smaller LR due to more params
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss"
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
    
    # === QNN Model Experiments (Updated) ===
    {
        "name": "BC_QNN_FedAvg_IID_PCA4_L2_Basic", # Standalone QNN for classification
        "model": {
            "type": "qnn", "n_qubits": 4, "n_layers": 2, "circuit_type": "basic",
            "output_dim": 2, # Request QNN to output 2 logits for CrossEntropyLoss
            "device": "lightning.qubit"
        },
        "data_config": {"name": "breast_cancer", "pca_features": 4, "scaling": "minmax"},
        "data_partition": {"iid": True},
        "aggregation_overrides": {"strategy": "FedAvg"},
        "client_overrides": {
            "client": {"local_epochs": 3},
            "optimization": {
                "learning_rate": 0.005,
                "optimizer": {"name": "Adam"},
                "loss_function": "CrossEntropyLoss"
            }
        },
        "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}
    },
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments for Breast Cancer Dataset")
    # You can add arguments here if needed, e.g., to select specific experiments by name/tag
    args = parser.parse_args()

    logger.info("Starting All Breast Cancer Experiments...")
    # results = run_all_bc_experiments(EXPERIMENTS_BREAST_CANCER)
    results = run_all_bc_experiments(EXPERIMENTS_BREAST_CANCER) # Uncomment to run all experimentsy
    logger.info("All Breast Cancer Experiments Finished.")

    try:
        enhanced_plot_bc_results(results)
    except ImportError:
        logger.warning("Advanced visualization modules not available. Falling back to basic visualization.")
        plot_bc_results(results)
    except Exception as e:
        logger.error(f"Error during advanced plotting: {e}", exc_info=True)
        logger.info("Falling back to basic visualizations...")
        plot_bc_results(results)

    logger.info("Script for Breast Cancer experiments finished.")