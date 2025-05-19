# src/utils/data_utils.py

import logging
import random
import os
import yaml
import numpy as np
import torch
import sys
from torch.utils.data import Dataset, Subset, TensorDataset
# torchvision not strictly needed if only using sklearn datasets here
# from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List, Optional, Tuple
from sklearn.datasets import load_iris, load_digits, load_breast_cancer # Ensure all are imported
from sklearn.model_selection import train_test_split

logger_data_utils = logging.getLogger(__name__) # Use a specific logger for this module

# --- Configuration Loading (can be kept if you use it within this file) ---

class FederatedTensorDataset(TensorDataset):
    def __init__(self, data_tensor, target_tensor):
        super().__init__(data_tensor, target_tensor)
        self.targets = target_tensor  # Required for non-IID partitioning

def load_config_from_utils(config_path: str, config_root_override: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    If config_root_override is provided, it's used. Otherwise, it tries to determine
    a 'configs' directory relative to this file's project structure.
    """
    if config_root_override:
        CONFIG_ROOT_UTILS = config_root_override
    elif 'CONFIG_ROOT' in globals(): # If run_experiment.py set a global one
        CONFIG_ROOT_UTILS = globals()['CONFIG_ROOT']
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root_marker = 'src'
        current_dir = script_dir
        while project_root_marker not in os.path.basename(current_dir) and os.path.dirname(current_dir) != current_dir:
             current_dir = os.path.dirname(current_dir)
        if project_root_marker in os.path.basename(current_dir):
             project_root = os.path.dirname(current_dir)
        else:
             project_root = script_dir
        CONFIG_ROOT_UTILS = os.path.join(project_root, 'configs')
        logger_data_utils.debug(f"data_utils.py determined CONFIG_ROOT as: {CONFIG_ROOT_UTILS}")

    if not os.path.isabs(config_path):
        full_path = os.path.join(CONFIG_ROOT_UTILS, config_path)
    else:
        full_path = config_path

    logger_data_utils.debug(f"Attempting to load config from: {full_path}")

    if not os.path.exists(full_path):
        logger_data_utils.error(f"Configuration file not found: {full_path}")
        raise FileNotFoundError(f"Configuration file not found: {full_path}")
    try:
        with open(full_path, 'r') as file:
            config = yaml.safe_load(file)
        logger_data_utils.info(f"Successfully loaded configuration from {full_path}")
        return config if config is not None else {}
    except yaml.YAMLError as e:
        logger_data_utils.error(f"Error parsing YAML file {full_path}: {e}", exc_info=True)
        raise
    except Exception as e:
        logger_data_utils.error(f"Unexpected error loading config file {full_path}: {e}", exc_info=True)
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
    logger_data_utils.info(f"Set all random seeds to {seed}")


# --- Data Loading and Preprocessing ---
def load_and_preprocess_data(
    dataset_name: str = "iris",
    n_features_pca: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaling_strategy: Optional[str] = "standard",
    data_subset_size: Optional[int] = None,
    tensor_dtype: torch.dtype = torch.float32
) -> Tuple[TensorDataset, TensorDataset, int, int]:
    """
    Loads, preprocesses, and splits a specified dataset from sklearn.datasets.

    Args:
        dataset_name: Name of the dataset ("iris", "digits", "breast_cancer").
        n_features_pca: Number of features to reduce to using PCA. None for no PCA.
        test_size: Proportion of dataset for the test split.
        random_state: Random seed for reproducibility.
        scaling_strategy: "standard" for StandardScaler, "minmax" for MinMaxScaler, or None.
        data_subset_size: If not None, use a random subset of this many total samples.
        tensor_dtype: PyTorch dtype for feature tensors.

    Returns:
        Tuple of (train_dataset, test_dataset, num_features, num_classes).
    """
    logger_data_utils.info(
        f"Loading dataset: {dataset_name}, PCA features={n_features_pca}, "
        f"Scaling={scaling_strategy}, Subset size={data_subset_size}, Dtype={tensor_dtype}"
    )
    set_seed(random_state)

    if dataset_name == "iris":
        data = load_iris()
    elif dataset_name == "digits":
        data = load_digits()
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    X, y = data.data, data.target
    n_classes = len(np.unique(y))
    logger_data_utils.debug(f"Original data shape: X={X.shape}, y={y.shape}. Num classes: {n_classes}")

    if data_subset_size is not None and 0 < data_subset_size < len(X):
        logger_data_utils.info(f"Subsampling dataset to {data_subset_size} total samples.")
        indices = np.random.choice(len(X), data_subset_size, replace=False)
        X, y = X[indices], y[indices]
        logger_data_utils.debug(f"Subsampled data shape: X={X.shape}, y={y.shape}")

    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if scaling_strategy == "standard":
        scaler = StandardScaler()
    elif scaling_strategy == "minmax":
        scaler = MinMaxScaler(feature_range=(-1, 1)) # Common for quantum encodings
    else: # None or other
        scaler = None

    if scaler:
        X_train_scaled = scaler.fit_transform(X_train_orig)
        X_test_scaled = scaler.transform(X_test_orig)
    else:
        X_train_scaled = X_train_orig.astype(np.float32) # Ensure consistent type if no scaling
        X_test_scaled = X_test_orig.astype(np.float32)

    X_train_final = X_train_scaled
    X_test_final = X_test_scaled
    num_features = X_train_final.shape[1]

    if n_features_pca is not None and 0 < n_features_pca < num_features:
        logger_data_utils.info(f"Applying PCA to reduce features to {n_features_pca}")
        pca = PCA(n_components=n_features_pca, random_state=random_state)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
        explained_variance = np.sum(pca.explained_variance_ratio_)
        num_features = X_train_final.shape[1]
        logger_data_utils.info(f"PCA applied. New feature count: {num_features}. Explained variance: {explained_variance:.4f}")

    X_train_tensor = torch.tensor(X_train_final, dtype=tensor_dtype)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_final, dtype=tensor_dtype)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = FederatedTensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = FederatedTensorDataset(X_test_tensor, y_test_tensor)

    logger_data_utils.info(
        f"Data preparation complete for '{dataset_name}'. Train size: {len(train_dataset)}, "
        f"Test size: {len(test_dataset)}, Features: {num_features}, Classes: {n_classes}"
    )
    return train_dataset, test_dataset, num_features, n_classes


# --- FederatedDataset (contains partitioning logic) ---
# (This is the FederatedDataset class from src/federated/utils.py,
# ensure it's correctly defined and imported in run_experiment.py or defined here if this is the sole utils file)
# For now, I will assume it's imported correctly in run_experiment.py.
# If you want to consolidate it here, copy the FederatedDataset class definition from
# your `src/federated/utils.py` (the one with iid_partition and dirichlet_partition).

# --- Example Usage (for testing this file) ---
if __name__ == '__main__':
    # Setup basic logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print("--- Testing Iris ---")
    train_ds_iris, test_ds_iris, n_feat_iris, n_class_iris = load_and_preprocess_data(
        dataset_name="iris", n_features_pca=3, scaling_strategy="minmax", tensor_dtype=torch.float32
    )
    print(f"Iris: Train: {len(train_ds_iris)}, Test: {len(test_ds_iris)}, Feat: {n_feat_iris}, Class: {n_class_iris}\n")

    print("--- Testing Digits ---")
    train_ds_digits, test_ds_digits, n_feat_digits, n_class_digits = load_and_preprocess_data(
        dataset_name="digits", n_features_pca=16, scaling_strategy="standard", tensor_dtype=torch.float64
    )
    print(f"Digits: Train: {len(train_ds_digits)}, Test: {len(test_ds_digits)}, Feat: {n_feat_digits}, Class: {n_class_digits}\n")

    print("--- Testing Breast Cancer (PCA, subset, float32) ---")
    train_ds_bc, test_ds_bc, n_feat_bc, n_class_bc = load_and_preprocess_data(
        dataset_name="breast_cancer", n_features_pca=10, scaling_strategy="standard",
        data_subset_size=300, tensor_dtype=torch.float32
    )
    print(f"Breast Cancer: Train: {len(train_ds_bc)}, Test: {len(test_ds_bc)}, Feat: {n_feat_bc}, Class: {n_class_bc}\n")

    print("--- Testing Breast Cancer (No PCA, full, float64) ---")
    train_ds_bc_full, test_ds_bc_full, n_feat_bc_full, n_class_bc_full = load_and_preprocess_data(
        dataset_name="breast_cancer", n_features_pca=None, scaling_strategy="standard", tensor_dtype=torch.float64
    )
    print(f"Breast Cancer (Full): Train: {len(train_ds_bc_full)}, Test: {len(test_ds_bc_full)}, Feat: {n_feat_bc_full}, Class: {n_class_bc_full}\n")

    # Test config loading (assuming a dummy config exists)
    # Create a dummy configs/test_cfg.yaml if it doesn't exist for this test
    # dummy_config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs')
    # os.makedirs(dummy_config_dir, exist_ok=True)
    # with open(os.path.join(dummy_config_dir, 'test_cfg.yaml'), 'w') as f:
    # f.write('key: value\n')
    # try:
    # cfg = load_config_from_utils("test_cfg.yaml")
    # print(f"Loaded test config: {cfg}")
    # except Exception as e:
    # print(f"Could not test load_config_from_utils: {e}")