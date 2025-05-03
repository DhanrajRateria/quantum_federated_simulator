# src/distributed/spark_executor_task.py

import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Iterator, Type, Union, Optional # Added Optional

# Imports needed on the executor node
# Assuming the 'src' directory is packaged and distributed to executors
try:
    from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
    from src.core.quantum_client import QuantumFederatedClient
    from src.federated.client import FederatedClient
    from src.federated.utils import set_seed
    IMPORT_SUCCESS = True
except ImportError as e:
    # This allows the driver to import this file without crashing if dependencies
    # are only on the cluster. Proper logging is tricky here as it runs on executor.
    print(f"[Executor Error] Failed to import necessary modules: {e}. Ensure 'src' is packaged or PYTHONPATH is set on executors.")
    IMPORT_SUCCESS = False
    # Define dummy classes/functions if needed to allow file parsing on driver
    class QuantumFederatedClient: pass
    class FederatedClient: pass
    class VariationalQuantumClassifier: pass
    class QuantumNeuralNetwork(nn.Module): pass
    class HybridQuantumModel(nn.Module): pass
    def set_seed(s): pass


# Logger setup needs to be careful on executors, might default to basic config
logger = logging.getLogger(__name__)
# Basic config for executor logs if not configured by Spark/runner
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] ExecutorTask: %(message)s')


def run_client_training_partition(
    partition_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    *, # Keyword-only arguments below
    model_info: Dict[str, Any], # Contains class name string, kwargs
    client_config: Dict[str, Any],
    global_params_bc: Any, # Spark Broadcast variable
    local_epochs: int,
    proximal_term: float,
    partition_id: int,
    seed: int
    ) -> Iterator[Tuple[Dict[str, torch.Tensor], int]]:
    """
    Executes local training for a data partition on a Spark executor.

    Args:
        partition_iterator: Iterator yielding (data, target) tuples.
        model_info: Dict containing 'class_name' (str) and 'kwargs' (dict).
        client_config: Configuration dictionary for the client (LR, batch size, etc.).
        global_params_bc: Spark broadcast variable (holds state_dict or VQC format).
        local_epochs: Number of local epochs.
        proximal_term: FedProx mu value.
        partition_id: Spark partition ID.
        seed: Random seed for the task.

    Yields:
        Single Tuple: (updated_local_params_cpu, num_samples_trained)
    """
    if not IMPORT_SUCCESS:
        logger.error(f"Task on partition {partition_id} cannot run due to import errors.")
        return iter([])

    process_start_time = time.time()
    task_seed = seed + partition_id
    set_seed(task_seed)
    client_id = f"spark_client_{partition_id}"
    logger.info(f"Task {client_id}: Starting on partition {partition_id} (Seed: {task_seed}).")

    # --- 1. Reconstruct Data ---
    partition_data = list(partition_iterator)
    if not partition_data: logger.warning(f"Task {client_id}: Partition empty."); return iter([])
    try:
        features_list, targets_list = zip(*partition_data)
        features_tensor = torch.stack(features_list).to(torch.float64) # Use float64
        targets_tensor = torch.stack(targets_list)
        partition_dataset = TensorDataset(features_tensor, targets_tensor)
        num_samples = len(partition_dataset)
        logger.info(f"Task {client_id}: Reconstructed dataset ({num_samples} samples).")
    except Exception as e: logger.error(f"Task {client_id}: Failed dataset reconstruction: {e}", exc_info=True); return iter([])

    # --- 2. Get Model Class and Instantiate ---
    try:
        model_class_name = model_info['class_name']
        model_kwargs_received = model_info['kwargs']
        model_init_kwargs = {
            k: v for k, v in model_kwargs_received.items()
            if k not in ['type', 'quantum_features', 'quantum_config_path', '_original_n_features'] # Add any other meta-keys used only in run_experiment
        }
        # Find the actual class object based on its name string
        # This requires the relevant modules to be imported above
        if model_class_name == "VariationalQuantumClassifier": model_class = VariationalQuantumClassifier
        elif model_class_name == "QuantumNeuralNetwork": model_class = QuantumNeuralNetwork
        elif model_class_name == "HybridQuantumModel": model_class = HybridQuantumModel
        elif model_class_name == "SimpleMLP":
             # Need to import SimpleMLP if used
             from src.models.classical_models import SimpleMLP
             model_class = SimpleMLP
        # Add other models here...
        else: raise ValueError(f"Unknown model class name: {model_class_name}")

        model_instance = model_class(**model_init_kwargs)
        is_pytorch_model = isinstance(model_instance, nn.Module)
        if is_pytorch_model: model_instance.double() # Ensure dtype
        logger.info(f"Task {client_id}: Instantiated model type: {model_class_name}")
    except Exception as e: logger.error(f"Task {client_id}: Failed model instantiation ({model_class_name}): {e}", exc_info=True); return iter([])

    # --- 3. Setup Client (Quantum or Base) ---
    try:
        ClientClass = QuantumFederatedClient if model_class_name != "SimpleMLP" else FederatedClient # Choose client type
        client_init_kwargs = {
            "batch_size": client_config.get("client", {}).get("batch_size", 32),
            "learning_rate": client_config.get("optimization", {}).get("learning_rate", 0.01),
            "device": torch.device("cuda" if client_config.get("resources",{}).get("use_gpu", False) and torch.cuda.is_available() else "cpu"),
            "optimizer_class": getattr(torch.optim, client_config.get("optimization", {}).get("optimizer", {}).get("name", "SGD"), torch.optim.SGD) if is_pytorch_model else None,
            "optimizer_kwargs": {k:v for k,v in client_config.get("optimization", {}).get("optimizer", {}).items() if k != 'name'} if is_pytorch_model else None,
            "loss_fn": getattr(nn, client_config.get("optimization", {}).get("loss_function", "CrossEntropyLoss"), nn.CrossEntropyLoss)() if is_pytorch_model else None,
        }
        client = ClientClass(client_id=client_id, model=model_instance, dataset=partition_dataset, **client_init_kwargs)
    except Exception as e: logger.error(f"Task {client_id}: Failed client setup: {e}", exc_info=True); return iter([])

    # --- 4. Load Global Parameters ---
    try:
        global_params = global_params_bc.value # Access broadcasted value
        client.set_parameters(global_params)
        logger.debug(f"Task {client_id}: Loaded global parameters.")
    except Exception as e: logger.error(f"Task {client_id}: Failed loading broadcasted params: {e}", exc_info=True); return iter([])

    # --- 5. Perform Local Training ---
    updated_params_dict = {}; samples_trained = 0 # Defaults
    try:
        logger.info(f"Task {client_id}: Starting {local_epochs} epochs training.")
        updated_params_dict, samples_trained = client.train(
            epochs=local_epochs,
            proximal_term=proximal_term,
            global_params=global_params # Pass dict for FedProx
        )
        logger.info(f"Task {client_id}: Training finished. Samples={samples_trained}.")
    except Exception as e: logger.error(f"Task {client_id}: Error during training: {e}", exc_info=True); return iter([])

    # --- 6. Yield Result ---
    process_end_time = time.time()
    logger.info(f"Task {client_id}: Completed in {process_end_time - process_start_time:.2f}s.")
    # Parameters are already on CPU from client.get_parameters()
    yield (updated_params_dict, samples_trained)

# Enhanced error handling
def enhanced_error_handling(partition_id, phase, exception):
    """Structured error reporting with classification and context"""
    error_code = classify_error(exception)
    context = {"partition_id": partition_id, "phase": phase, "timestamp": time.time()}
    error_report = {"error_code": error_code, "message": str(exception), "context": context}
    return error_report

# Memory metrics collection
def collect_memory_metrics():
    """Collect memory usage during training"""
    import psutil
    import gc
    
    # Force garbage collection before measurement
    gc.collect()
    
    process = psutil.Process()
    metrics = {
        "rss_mb": process.memory_info().rss / (1024 * 1024),
        "vms_mb": process.memory_info().vms / (1024 * 1024),
        "percent": process.memory_percent(),
        "cpu_percent": process.cpu_percent()
    }
    
    # For quantum models, collect circuit metrics if available
    try:
        if hasattr(current_model, "circuit"):
            metrics["q_depth"] = current_model.circuit.depth
            metrics["q_gates"] = len(current_model.circuit.operations)
    except:
        pass
        
    return metrics

# Non-IID partitioning within executor
def create_non_iid_local_partition(dataset, alpha=0.5, min_size=10):
    """Create a non-IID partition from dataset on executor"""
    import numpy as np
    
    # Extract labels
    all_labels = [y for _, y in dataset]
    unique_labels = sorted(set(all_labels))
    label_indices = {l: [i for i, (_, y) in enumerate(dataset) if y == l] 
                     for l in unique_labels}
    
    # Use Dirichlet to determine proportion of each class to include
    proportions = np.random.dirichlet([alpha] * len(unique_labels))
    
    # Ensure minimum samples per class
    min_proportion = min_size / len(dataset)
    proportions = np.clip(proportions, min_proportion, 1.0)
    proportions = proportions / proportions.sum()
    
    # Select indices based on proportions
    selected_indices = []
    for l_idx, proportion in enumerate(proportions):
        label_count = int(proportion * len(dataset))
        # Pick randomly from this label's indices
        indices = np.random.choice(
            label_indices[unique_labels[l_idx]], 
            size=min(label_count, len(label_indices[unique_labels[l_idx]])),
            replace=False
        )
        selected_indices.extend(indices)
    
    # Create subset
    from torch.utils.data import Subset
    return Subset(dataset, selected_indices)