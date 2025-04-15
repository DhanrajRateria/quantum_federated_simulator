# run_spark_experiment.py
import os
import logging
import argparse
import time
import sys
from typing import Dict, List, Any, Optional, Type
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# --- Path Setup ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
CONFIG_ROOT = os.path.join(PROJECT_ROOT, 'configs')
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results')
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations')
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs')
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(VIS_DIR, exist_ok=True); os.makedirs(LOG_DIR, exist_ok=True)

# --- Import project components ---
from src.federated.utils import (set_seed, load_config, load_mnist_data, partition_data) # Use load_config from utils
from src.federated.aggregation import AggregationStrategy, FedAvg
from src.distributed.spark_manager import SparkManager
from src.core.spark_server import SparkFederatedServer # Import the Spark server
from src.core.quantum_manager import QuantumAggregationStrategy # Keep for VQC aggregation
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.models.classical_models import SimpleMLP

# --- Configure Logging ---
log_file_path = os.path.join(LOG_DIR, f"experiment_run_spark_{time.strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("SparkExperimentRunner")


# --- Keep merge_configs, create_model ---
def merge_configs(*configs: Dict) -> Dict: # ... (implementation as before) ...
    merged = {};
    for config in configs:
        for key, value in config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict): merged[key] = merge_configs(merged[key], value)
            else: merged[key] = value
    return merged

def create_model(model_config: Dict, n_features: int, n_classes: int) -> Any: # ... (implementation as before) ...
    model_type = model_config.get("type", "classical_mlp").lower(); logger.info(f"Driver: Creating model of type: {model_type}")
    effective_n_features = n_features
    if "quantum_features" in model_config and model_config["quantum_features"] < n_features: effective_n_features = model_config["quantum_features"]
    q_config = {}; q_config_path = model_config.get("quantum_config_path")
    if q_config_path: q_config = load_config(q_config_path)
    else: q_config = model_config
    if model_type == "classical_mlp": hidden_dim = model_config.get("hidden_dim", 64); return SimpleMLP(input_dim=effective_n_features, hidden_dim=hidden_dim, output_dim=n_classes)
    elif model_type == "vqc": n_qubits = min(q_config.get("n_qubits", effective_n_features), effective_n_features); return VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=q_config.get("n_layers", 1), n_classes=n_classes, circuit_type=q_config.get("circuit_type", "basic"), device_name=q_config.get("device", "default.qubit"), shots=q_config.get("shots"))
    elif model_type == "qnn": n_qubits = min(q_config.get("n_qubits", effective_n_features), effective_n_features); return QuantumNeuralNetwork(n_qubits=n_qubits, n_layers=q_config.get("n_layers", 1), output_dim=n_classes, circuit_type=q_config.get("circuit_type", "basic"), device_name=q_config.get("device", "default.qubit"), shots=q_config.get("shots"))
    elif model_type == "hybrid": n_qubits = min(q_config.get("n_qubits", effective_n_features), effective_n_features); original_n_features = model_config.get("_original_n_features", n_features); return HybridQuantumModel(n_qubits=n_qubits, classical_input_dim=original_n_features, classical_hidden_dim=model_config.get("classical_hidden_dim", 32), classical_output_dim=n_classes, q_layers=q_config.get("n_layers", 1), q_circuit_type=q_config.get("circuit_type", "basic"), q_device_name=q_config.get("device", "default.qubit"), q_shots=q_config.get("shots"))
    else: raise ValueError(f"Unknown model type: {model_type}")

# --- Visualization (Keep as before) ---
def plot_results(all_results: List[Dict]): # ... (implementation as before, using VIS_DIR) ...
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import seaborn as sns
    logger.info("Generating comparison plots..."); os.makedirs(VIS_DIR, exist_ok=True); plt.style.use('seaborn-v0_8-darkgrid'); plot_data = []
    for result in all_results:
        exp_name = result['experiment_name']; round_history = result.get('round_history', [])
        if result.get("error") or not round_history: logger.warning(f"Skipping plot for {exp_name}"); continue
        for round_data in round_history: round_num = round_data['round']; metrics = round_data.get('evaluation_metrics', {});
        if metrics: plot_data.append({'Experiment': exp_name, 'Round': round_num, 'Accuracy': metrics.get('accuracy'), 'Loss': metrics.get('loss')})
    if not plot_data: logger.warning("No data for plotting."); return
    plot_df = pd.DataFrame(plot_data); plt.figure(figsize=(12, 7)); sns.lineplot(data=plot_df, x='Round', y='Accuracy', hue='Experiment', marker='o', errorbar=None)
    plt.title('Federated Learning Accuracy Comparison'); plt.xlabel('Comm Round'); plt.ylabel('Global Accuracy'); plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1]); plot_filename = os.path.join(VIS_DIR, "accuracy_vs_round_spark.png"); plt.savefig(plot_filename); plt.close(); logger.info(f"Accuracy plot saved to {plot_filename}")
    plot_df_loss = plot_df.dropna(subset=['Loss']);
    if not plot_df_loss.empty: plt.figure(figsize=(12, 7)); sns.lineplot(data=plot_df_loss, x='Round', y='Loss', hue='Experiment', marker='o', errorbar=None); plt.title('Federated Learning Loss Comparison'); plt.xlabel('Comm Round'); plt.ylabel('Global Loss'); plt.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left'); plt.tight_layout(rect=[0, 0, 0.85, 1]); plot_filename = os.path.join(VIS_DIR, "loss_vs_round_spark.png"); plt.savefig(plot_filename); plt.close(); logger.info(f"Loss plot saved to {plot_filename}")
    else: logger.info("Skipping loss plot (no valid data).")

# --- Experiment Definitions (Keep as before) ---
EXPERIMENTS = [
     # ... (Your experiment definitions as before) ...
    {   "name": "Classical_MLP_FedAvg_IID", "model": {"type": "classical_mlp", "hidden_dim": 64}, "data_partition": {"iid": True}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 2}, "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}}, "server_overrides": {"server": {"num_rounds": 10}, "client_management": {"num_clients": 5}}},
    {   "name": "VQC_FedAvg_IID_PCA4", "model": {"type": "vqc", "quantum_features": 4, "n_qubits": 4, "n_layers": 1}, "data_partition": {"iid": True}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 1}, "optimization": {"learning_rate": 0.05}}, "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}},
    {   "name": "QNN_FedAvg_IID_PCA4", "model": {"type": "qnn", "quantum_features": 4, "n_qubits": 4, "n_layers": 1}, "data_partition": {"iid": True}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 1}, "optimization": {"learning_rate": 0.005, "optimizer": {"name": "Adam"}}}, "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}},
    {   "name": "Hybrid_FedAvg_IID_PCA4", "model": {"type": "hybrid", "quantum_features": 4, "classical_hidden_dim": 16, "n_qubits": 4, "n_layers": 1}, "data_partition": {"iid": True}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 1}, "optimization": {"learning_rate": 0.005, "optimizer": {"name": "Adam"}}}, "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}},
    {   "name": "Classical_MLP_FedAvg_NonIID", "model": {"type": "classical_mlp", "hidden_dim": 64}, "data_partition": {"iid": False, "alpha": 0.3}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 2}, "optimization": {"learning_rate": 0.01, "optimizer": {"name": "Adam"}}}, "server_overrides": {"server": {"num_rounds": 10}, "client_management": {"num_clients": 5}}},
     {  "name": "VQC_FedAvg_NonIID_PCA4", "model": {"type": "vqc", "quantum_features": 4, "n_qubits": 4, "n_layers": 1}, "data_partition": {"iid": False, "alpha": 0.3}, "aggregation_overrides": {"strategy": "FedAvg"}, "client_overrides": {"client": {"local_epochs": 1}, "optimization": {"learning_rate": 0.05}}, "server_overrides": {"server": {"num_rounds": 5}, "client_management": {"num_clients": 3}}},
]

# --- Main Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments using Spark")
    args = parser.parse_args()

    logger.info("Starting All Spark Experiments...")
    all_results = []
    spark_manager = None # Define in outer scope for finally block

    try:
        # --- Initialize Spark ---
        spark_config_path = os.path.join(CONFIG_ROOT, "distributed", "spark_config.yaml")
        spark_manager = SparkManager(config_path=spark_config_path)
        spark_manager.start()

        # --- Load Data Once ---
        full_train_data, full_test_data, n_features_original = load_mnist_data()

        # --- Run Experiments ---
        for i, exp_config in enumerate(EXPERIMENTS):
            exp_name = exp_config.get("name", f"Experiment_{i+1}")
            logger.info(f"========== Starting Spark Experiment: {exp_name} ==========")
            server = None # Ensure server is defined
            try:
                set_seed(exp_config.get("seed", 42))
                # --- Load experiment specific configs ---
                base_client_cfg = load_config("federated/client_config.yaml"); base_agg_cfg = load_config("federated/default_aggregation.yaml"); server_cfg_overrides = exp_config.get("server_overrides", {}); client_cfg = merge_configs(base_client_cfg, exp_config.get("client_overrides", {})); agg_cfg = merge_configs(base_agg_cfg, exp_config.get("aggregation_overrides", {})); model_cfg = exp_config["model"]; model_cfg["_original_n_features"] = n_features_original
                model_type_str = model_cfg.get("type", "classical_mlp").lower(); is_quantum = "vqc" in model_type_str or "qnn" in model_type_str or "hybrid" in model_type_str

                # --- Prepare Data (Partitioning logic for RDD) ---
                num_clients = server_cfg_overrides.get("client_management", {}).get("num_clients", 5)
                iid = exp_config.get("data_partition", {}).get("iid", True); alpha = exp_config.get("data_partition", {}).get("alpha", 0.5)
                quantum_features = model_cfg.get("quantum_features"); effective_n_features = n_features_original; eval_dataset = full_test_data
                train_data_to_partition = full_train_data

                if is_quantum and quantum_features and quantum_features < n_features_original:
                     logger.info(f"Getting PCA data (features={quantum_features}) for {exp_name}")
                     q_train_data, q_test_data, effective_n_features = load_mnist_data(n_features=quantum_features); train_data_to_partition = q_train_data; eval_dataset = q_test_data
                else: effective_n_features = n_features_original

                # Create data list for parallelization
                data_list = [(train_data_to_partition[i][0], train_data_to_partition[i][1]) for i in range(len(train_data_to_partition))]
                data_rdd = spark_manager.get_context().parallelize(data_list, numSlices=num_clients).cache()
                num_partitions = data_rdd.getNumPartitions(); data_rdd.count(); logger.info(f"Data parallelized into {num_partitions} partitions.")


                # --- Create Global Model ---
                n_classes = 10; global_model = create_model(model_cfg, effective_n_features, n_classes)
                ModelClassForClient = type(global_model)
                client_model_kwargs = model_cfg.copy(); # ... (update client_model_kwargs for effective_n_features/classes) ...
                if "n_qubits" in client_model_kwargs: client_model_kwargs["n_qubits"] = min(client_model_kwargs.get("n_qubits", effective_n_features), effective_n_features)
                if model_type_str == "vqc": client_model_kwargs["n_classes"]=n_classes
                if model_type_str == "qnn": client_model_kwargs["output_dim"]=n_classes
                if model_type_str == "hybrid": client_model_kwargs["classical_output_dim"]=n_classes; client_model_kwargs["classical_input_dim"]=n_features_original


                # --- Aggregation Strategy ---
                use_quantum_agg = "vqc" in model_type_str; base_agg_strategy = AggregationStrategy.from_config(agg_cfg); aggregation_strategy = QuantumAggregationStrategy(base_agg_strategy) if use_quantum_agg else base_agg_strategy; logger.info(f"Using aggregation: {type(aggregation_strategy).__name__}")

                # --- Initialize Server ---
                server_device = torch.device("cuda" if client_cfg.get("resources",{}).get("use_gpu", False) and torch.cuda.is_available() else "cpu")
                server = SparkFederatedServer(model=global_model, spark_manager=spark_manager, aggregation_strategy=aggregation_strategy, evaluation_dataset=eval_dataset, device=server_device)
                if isinstance(server.model, nn.Module): server.model.double()

                # --- Run Training ---
                num_rounds = server_cfg_overrides.get("server", {}).get("num_rounds", 5)
                local_epochs = client_cfg.get("client", {}).get("local_epochs", 1)
                proximal_term = client_cfg.get("regularization",{}).get("proximal_mu", 0.0)

                round_history = server.train(
                    num_rounds=num_rounds, train_dataset=None, client_data_rdd=data_rdd,
                    num_data_partitions=num_partitions, model_class_for_client=ModelClassForClient,
                    model_kwargs_for_client=client_model_kwargs, client_config=client_cfg,
                    local_epochs=local_epochs, proximal_term=proximal_term, seed=exp_config.get("seed", 42)
                )

                result_entry = {
                    "experiment_name": exp_name, "config": exp_config, "round_history": round_history,
                    "final_accuracy": round_history[-1]['evaluation_metrics'].get('accuracy') if round_history else None,
                    "final_loss": round_history[-1]['evaluation_metrics'].get('loss') if round_history else None,
                }; all_results.append(result_entry)
                logger.info(f"========== Finished Spark Experiment: {exp_name} ==========\n")

            except Exception as e:
                logger.error(f"========== Spark Experiment {exp_name} failed: {e} ==========", exc_info=True)
                all_results.append({"experiment_name": exp_name, "config": exp_config, "round_history": [], "error": str(e)})
            finally:
                 if 'data_rdd' in locals() and data_rdd is not None:
                      data_rdd.unpersist()

    finally:
        # --- Stop Spark ---
        if spark_manager:
            spark_manager.stop()

    # --- Save & Plot Results ---
    # (Saving/Plotting logic using RESULTS_DIR, VIS_DIR as before)
    results_df=pd.json_normalize(all_results, sep='_'); timestamp=time.strftime("%Y%m%d-%H%M%S"); results_filename=os.path.join(RESULTS_DIR, f"experiment_results_spark_{timestamp}.csv")
    try: results_df.to_csv(results_filename, index=False); logger.info(f"Spark results saved to {results_filename}")
    except Exception as e: logger.error(f"Failed to save Spark results CSV: {e}"); # ... (JSON fallback) ...
    try: plot_results(all_results)
    except ImportError: logger.warning("Plotting libraries not found.")
    except Exception as e: logger.error(f"Error plotting: {e}", exc_info=True)

    logger.info("Spark script finished.")