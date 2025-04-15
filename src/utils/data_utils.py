# src/utils/data_utils.py (or src/federated/utils.py)

import logging
import random
import os # Import os for path joining
import yaml # Import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List, Optional, Tuple # Added List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# --- Configuration Loading ---
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file relative to the CONFIG_ROOT.

    Args:
        config_path: Path to the YAML configuration file *relative* to the 'configs' directory
                     (e.g., "federated/server.yaml", "quantum/basic_circuit.yaml").

    Returns:
        Dictionary containing configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If the file cannot be parsed.
    """
    # Assume CONFIG_ROOT is defined globally or passed appropriately.
    # For standalone use, define it relative to this file or expect absolute path.
    # In the context of run_experiment.py, CONFIG_ROOT was defined there.
    # If running this utils file directly, adjust this:
    if 'CONFIG_ROOT' not in globals():
        # Define relative to this file's location if run directly
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up directories until 'configs' is found or reach project root
        project_root_marker = 'src' # Or another marker like '.git'
        current_dir = script_dir
        while project_root_marker not in os.path.basename(current_dir) and os.path.dirname(current_dir) != current_dir:
             current_dir = os.path.dirname(current_dir)
        # If marker found, assume configs is sibling to src
        if project_root_marker in os.path.basename(current_dir):
             project_root = os.path.dirname(current_dir)
        else: # Fallback: assume configs is relative to script_dir if marker not found
             project_root = script_dir # This might be incorrect depending on structure
        CONFIG_ROOT = os.path.join(project_root, 'configs')
        logger.debug(f"CONFIG_ROOT automatically determined as: {CONFIG_ROOT}")


    full_path = os.path.join(CONFIG_ROOT, config_path)
    logger.debug(f"Attempting to load config from: {full_path}")

    if not os.path.exists(full_path):
        logger.error(f"Configuration file not found: {full_path}")
        raise FileNotFoundError(f"Configuration file not found: {full_path}")

    try:
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {full_path}")
        if config is None: # Handle empty YAML file case
             logger.warning(f"Config file {full_path} is empty.")
             return {}
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {full_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading config file {full_path}: {e}", exc_info=True)
        raise


# --- Seed setting ---
def set_seed(seed: int):
    """Sets random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


# --- Data Loading ---
def load_mnist_data(data_path: str = "./data", n_features: Optional[int] = None, normalize: bool = True) -> Tuple[TensorDataset, TensorDataset, int]:
    """
    Loads MNIST dataset, applies transformations, and optionally reduces dimensionality.

    Args:
        data_path: Path to download/load MNIST data.
        n_features: If not None, reduce features to this dimension using PCA. Must be <= 784.
        normalize: If True, scale features using StandardScaler then to [-1, 1] using MinMaxScaler.

    Returns:
        Tuple: (train_dataset, test_dataset, actual_n_features)
    """
    logger.info(f"Loading MNIST data from {data_path}. Target features: {n_features}. Normalize: {normalize}")
    # Basic transform to get Tensor data first
    basic_transform = transforms.Compose([transforms.ToTensor()])

    try:
        train_data_raw = datasets.MNIST(data_path, train=True, download=True, transform=basic_transform)
        test_data_raw = datasets.MNIST(data_path, train=False, download=True, transform=basic_transform)
    except Exception as e:
        logger.error(f"Failed to download or load MNIST data from {data_path}: {e}", exc_info=True)
        raise

    # Flatten images and convert labels
    # Ensure data is float32 for sklearn compatibility initially
    X_train = train_data_raw.data.view(len(train_data_raw), -1).numpy().astype(np.float32)
    y_train = train_data_raw.targets.numpy()
    X_test = test_data_raw.data.view(len(test_data_raw), -1).numpy().astype(np.float32)
    y_test = test_data_raw.targets.numpy()

    original_n_features = X_train.shape[1]
    actual_n_features = original_n_features

    # Apply StandardScaler (important for PCA and general ML)
    logger.debug("Applying StandardScaler to MNIST features.")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test) # Use transform only on test data

    # Apply PCA if requested
    if n_features is not None and 0 < n_features < original_n_features:
        logger.info(f"Applying PCA to reduce features from {original_n_features} to {n_features}")
        if n_features > original_n_features:
             logger.warning(f"Requested n_features ({n_features}) > original ({original_n_features}). PCA cannot increase dimensions. Using original features.")
        else:
            pca = PCA(n_components=n_features, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test) # Use transform only on test data
            actual_n_features = n_features
            logger.info(f"PCA completed. Explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    elif n_features is not None and n_features >= original_n_features:
         logger.info(f"Requested n_features ({n_features}) >= original ({original_n_features}). Skipping PCA.")
    else:
        logger.info(f"Using original {actual_n_features} features (PCA not requested).")


    # Apply normalization to [-1, 1] range if requested
    if normalize:
         logger.info("Applying MinMaxScaler to scale features to [-1, 1] range.")
         minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
         X_train = minmax_scaler.fit_transform(X_train)
         X_test = minmax_scaler.transform(X_test) # Use transform only

    # Convert back to tensors - use float64 for Pennylane compatibility
    X_train_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float64)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    logger.info(f"MNIST data loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}, Features: {actual_n_features}")
    return train_dataset, test_dataset, actual_n_features


# --- Data Partitioning ---
def partition_data(dataset: Dataset, num_clients: int, iid: bool = True, alpha: float = 0.5, seed: int = 42) -> List[Subset]:
    """Partitions a dataset among clients (IID or Non-IID Dirichlet)."""
    np.random.seed(seed)
    num_samples = len(dataset)
    indices = np.arange(num_samples)

    if num_clients <= 0:
         raise ValueError("Number of clients must be positive.")
    if num_samples == 0:
         logger.warning("Dataset is empty, returning empty subsets.")
         return [Subset(dataset, []) for _ in range(num_clients)]

    client_datasets = []

    if iid:
        logger.info(f"Partitioning data IID among {num_clients} clients.")
        np.random.shuffle(indices) # Shuffle for IID split
        split_indices = np.array_split(indices, num_clients)
        # Ensure all indices are assigned even if not perfectly divisible
        client_datasets = [Subset(dataset, idx.tolist()) for idx in split_indices if len(idx) > 0]
        # If partitioning resulted in fewer subsets than clients (e.g., samples < clients)
        while len(client_datasets) < num_clients:
             logger.warning(f"Adding empty subset for client {len(client_datasets)} due to few samples.")
             client_datasets.append(Subset(dataset, []))

    else: # Non-IID Dirichlet distribution
        logger.info(f"Partitioning data Non-IID (Dirichlet alpha={alpha}) among {num_clients} clients.")

        # Extract targets efficiently
        if isinstance(dataset, TensorDataset):
            targets_np = dataset.tensors[1].numpy()
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, TensorDataset):
            targets_np = dataset.dataset.tensors[1][dataset.indices].numpy()
        else:
            logger.warning("Attempting to get targets by iterating dataset for Non-IID split (might be slow).")
            try:
                targets_np = np.array([dataset[i][1] for i in range(num_samples)])
            except Exception as e:
                logger.error(f"Failed to extract targets for Non-IID split: {e}. Cannot proceed with Non-IID.", exc_info=True)
                raise ValueError("Could not extract targets for Non-IID split.") from e

        num_classes = len(np.unique(targets_np))
        if num_classes <= 1 and num_clients > 1:
             logger.warning(f"Only {num_classes} class(es) found in data. Non-IID Dirichlet split might behave like IID.")

        # Map targets to indices
        indices_per_class = {cls: indices[targets_np[indices] == cls] for cls in range(num_classes)}
        client_indices_map = {i: [] for i in range(num_clients)}

        # Distribute indices for each class based on Dirichlet proportions
        for cls in range(num_classes):
            class_indices = indices_per_class.get(cls, np.array([]))
            num_class_samples = len(class_indices)
            if num_class_samples == 0: continue

            # Ensure class indices are shuffled
            np.random.shuffle(class_indices)

            # Generate proportions for this class
            try:
                proportions = np.random.dirichlet([alpha] * num_clients)
            except ValueError as e: # Handle alpha <= 0 case if needed
                 logger.error(f"Invalid alpha value {alpha} for Dirichlet distribution: {e}")
                 raise

            target_samples_per_client = (proportions * num_class_samples).astype(int)
            # Distribute remainder due to rounding
            remainder = num_class_samples - target_samples_per_client.sum()
            add_indices = np.random.choice(num_clients, size=remainder, replace=True) # Distribute randomly
            for client_idx in add_indices: target_samples_per_client[client_idx] += 1

            # Assign indices
            current_idx = 0
            for client_id in range(num_clients):
                take = target_samples_per_client[client_id]
                assigned_indices = class_indices[current_idx : current_idx + take]
                client_indices_map[client_id].extend(assigned_indices)
                current_idx += take

        client_datasets = [Subset(dataset, client_indices_map[i]) for i in range(num_clients)]

    # Log distribution summary
    logger.info("Data partitioning complete.")
    for i, d in enumerate(client_datasets): logger.debug(f"Client {i} final data size: {len(d)}")

    return client_datasets


# --- FederatedDataset Wrapper (Optional but can be useful) ---
class FederatedDataset:
    """ Helper class to manage partitioned datasets """
    def __init__(self, base_dataset: Dataset, num_clients: int, iid: bool = True, alpha: float = 0.5, seed: int = 42):
        self.base_dataset = base_dataset
        self.num_clients = num_clients
        self.client_datasets = partition_data(base_dataset, num_clients, iid, alpha, seed)

    def get_client_dataset(self, client_id: int) -> Subset:
        """ Returns the dataset for a specific client ID """
        if 0 <= client_id < self.num_clients:
            return self.client_datasets[client_id]
        else:
            raise IndexError(f"Client ID {client_id} out of range (0-{self.num_clients-1})")

    def __len__(self):
        return self.num_clients

    def __getitem__(self, client_id: int):
        return self.get_client_dataset(client_id)