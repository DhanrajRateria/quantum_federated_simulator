import logging
import random
import os
import yaml
import numpy as np
import torch
import sys
from torch.utils.data import Dataset, Subset, TensorDataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Dict, Any, List, Optional, Tuple
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split

# NEW: Import torchvision for image datasets like CIFAR-10
from torchvision import datasets, transforms

logger_data_utils = logging.getLogger(__name__)

# --- Custom TensorDataset that holds a 'targets' attribute for compatibility ---
class FederatedTensorDataset(TensorDataset):
    """A custom TensorDataset that also stores targets for non-IID partitioning."""
    def __init__(self, *tensors):
        super(FederatedTensorDataset, self).__init__(*tensors)
        # Assumes the last tensor is the target tensor
        self.targets = tensors[-1]

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
    model_type: str = "mlp",
    n_features_pca: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scaling_strategy: Optional[str] = "standard",
    data_subset_size: Optional[int] = None,
    tensor_dtype: torch.dtype = torch.float32
) -> Tuple[Dataset, Dataset, int, int]:
    """
    Loads, preprocesses, and splits a specified dataset.
    Supports sklearn datasets and torchvision's CIFAR-10.

    Args:
        dataset_name: Name of the dataset ("iris", "digits", "breast_cancer", "cifar10").
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
        f"Scaling={scaling_strategy}, Subset size={data_subset_size}"
    )
    set_seed(random_state)

    X, y = None, None
    X_train_orig, X_test_orig, y_train, y_test = None, None, None, None
    num_features = 0

    if dataset_name in ["iris", "digits", "breast_cancer"]:
        if dataset_name == "iris":
            data = load_iris()
        elif dataset_name == "digits":
            data = load_digits()
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
        
        X, y = data.data, data.target
        
        if data_subset_size is not None and 0 < data_subset_size < len(X):
            logger_data_utils.info(f"Subsampling dataset to {data_subset_size} total samples.")
            indices = np.random.choice(len(X), data_subset_size, replace=False)
            X, y = X[indices], y[indices]

        # Split before scaling/PCA
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if np.any(y) else None
        )
        if scaling_strategy == "standard": scaler = StandardScaler()
        else: scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train_orig)
        X_test_scaled = scaler.transform(X_test_orig)
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
        num_features = X_train_final.shape[1]
        
        X_train_tensor = torch.tensor(X_train_final, dtype=tensor_dtype)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_final, dtype=tensor_dtype)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        train_dataset = FederatedTensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = FederatedTensorDataset(X_test_tensor, y_test_tensor)
        n_classes = len(np.unique(y_train))

    elif dataset_name in ["cifar10", "fashion-mnist"]: # Add fashion-mnist here
        logger_data_utils.info(f"Loading and preprocessing {dataset_name} for model type: {model_type}")
        data_path = './data'
        
        # Select the correct dataset from torchvision
        if dataset_name == "cifar10":
            DatasetClass = datasets.CIFAR10
            n_classes = 10
        else: # fashion-mnist
            DatasetClass = datasets.FashionMNIST
            n_classes = 10
        if model_type == 'cnn':
            # For CNN, we normalize per-channel, as is standard
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            num_features = 3 # Input channels for CNN
            
            train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        else: # model_type == 'mlp'
            # For MLP, flatten the image and convert to a TensorDataset
            train_set_raw = datasets.CIFAR10(root=data_path, train=True, download=True)
            test_set_raw = datasets.CIFAR10(root=data_path, train=False, download=True)
            
            X_train_flat = train_set_raw.data.reshape(len(train_set_raw), -1).astype(np.float32) / 255.0
            y_train_flat = torch.tensor(train_set_raw.targets, dtype=torch.long)
            X_test_flat = test_set_raw.data.reshape(len(test_set_raw), -1).astype(np.float32) / 255.0
            y_test_flat = torch.tensor(test_set_raw.targets, dtype=torch.long)
            
            num_features = X_train_flat.shape[1] # Will be 3072
            
            # Apply standard scaling to the flattened data
            if scaling_strategy == 'standard':
                scaler = StandardScaler()
                X_train_flat = scaler.fit_transform(X_train_flat)
                X_test_flat = scaler.transform(X_test_flat)
            
            train_dataset = FederatedTensorDataset(torch.tensor(X_train_flat, dtype=tensor_dtype), y_train_flat)
            test_dataset = FederatedTensorDataset(torch.tensor(X_test_flat, dtype=tensor_dtype), y_test_flat)

    elif dataset_name == "synthetic_multi_modal":
        return load_multi_modal_synthetic_data(random_state=random_state)
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    logger_data_utils.info(
        f"Data preparation complete. Train size: {len(train_dataset)}, "
        f"Test size: {len(test_dataset)}, Features/Channels: {num_features}, Classes: {n_classes}"
    )
    # The return type is now a generic Dataset
    return train_dataset, test_dataset, num_features, n_classes


# --- Example Usage (for testing this file) ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    print("\n--- Testing Iris ---")
    train_ds_iris, test_ds_iris, n_feat_iris, n_class_iris = load_and_preprocess_data(
        dataset_name="iris", n_features_pca=3, scaling_strategy="minmax"
    )
    print(f"Iris: Train: {len(train_ds_iris)}, Test: {len(test_ds_iris)}, Feat: {n_feat_iris}, Class: {n_class_iris}")

    print("\n--- Testing Digits ---")
    train_ds_digits, test_ds_digits, n_feat_digits, n_class_digits = load_and_preprocess_data(
        dataset_name="digits", n_features_pca=16, scaling_strategy="standard"
    )
    print(f"Digits: Train: {len(train_ds_digits)}, Test: {len(test_ds_digits)}, Feat: {n_feat_digits}, Class: {n_class_digits}")

    print("\n--- Testing Breast Cancer ---")
    train_ds_bc, test_ds_bc, n_feat_bc, n_class_bc = load_and_preprocess_data(
        dataset_name="breast_cancer", n_features_pca=10, scaling_strategy="standard",
        data_subset_size=300
    )
    print(f"Breast Cancer: Train: {len(train_ds_bc)}, Test: {len(test_ds_bc)}, Feat: {n_feat_bc}, Class: {n_class_bc}")
    
    print("\n--- Testing CIFAR-10 (with subset and PCA for speed) ---")
    train_ds_cifar, test_ds_cifar, n_feat_cifar, n_class_cifar = load_and_preprocess_data(
        dataset_name="cifar10", 
        data_subset_size=5000, # Use a small subset for quick testing
        n_features_pca=128,      # Reduce features for quick testing
        scaling_strategy="standard"
    )
    print(f"CIFAR-10: Train: {len(train_ds_cifar)}, Test: {len(test_ds_cifar)}, Feat: {n_feat_cifar}, Class: {n_class_cifar}")


def load_multi_modal_synthetic_data(
    n_samples: int = 2000,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[TensorDataset, TensorDataset, int, int]:
    """
    Generates a synthetic dataset with a multi-modal class distribution.
    - Class 0: A single cluster.
    - Class 1: Two spatially separated clusters.
    
    This creates a scenario where averaging models trained on different modes
    of Class 1 will fail.
    """
    logger_data_utils.info("Generating multi-modal synthetic dataset...")
    set_seed(random_state)
    
    # Class 0: One central cluster
    X0, y0 = make_blobs(n_samples=n_samples // 2, centers=[(0, 0)], n_features=2, cluster_std=0.8, random_state=random_state)
    y0[:] = 0
    
    # Class 1: Two separated clusters
    X1a, y1a = make_blobs(n_samples=n_samples // 4, centers=[(-4, 4)], n_features=2, cluster_std=0.8, random_state=random_state)
    y1a[:] = 1
    
    X1b, y1b = make_blobs(n_samples=n_samples // 4, centers=[(4, -4)], n_features=2, cluster_std=0.8, random_state=random_state)
    y1b[:] = 1
    
    # Combine the data
    X = np.vstack((X0, X1a, X1b))
    y = np.hstack((y0, y1a, y1b))
    
    # --- Standard split and tensor conversion ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    num_features = 2
    num_classes = 2
    
    return train_dataset, test_dataset, num_features, num_classes

def pathological_synthetic_partition(full_dataset: TensorDataset, num_clients: int) -> List[Subset]:
        """
        Splits the synthetic multi-modal dataset pathologically.
        Even clients get Class 0 + top-left Class 1.
        Odd clients get Class 0 + bottom-right Class 1.
        """
        logger_data_utils.info("Partitioning synthetic data pathologically...")
        X, y = full_dataset.tensors
        
        # Identify the indices for each cluster
        class0_indices = np.where(y == 0)[0]
        # Identify Class 1 modes by their original location (before scaling)
        # This is a simplification; we can also do it by a simple split
        class1_indices = np.where(y == 1)[0]
        # Simple split: first half of class 1 indices go to one mode, second to other
        class1_mode_a_indices = class1_indices[:len(class1_indices)//2]
        class1_mode_b_indices = class1_indices[len(class1_indices)//2:]
        
        client_indices = [[] for _ in range(num_clients)]
        for i in range(num_clients):
            # All clients get a share of Class 0
            client_indices[i].extend(np.array_split(class0_indices, num_clients)[i])
            
            # Distribute the two modes of Class 1
            if i % 2 == 0: # Even clients get mode A
                client_indices[i].extend(np.array_split(class1_mode_a_indices, (num_clients + 1) // 2)[i // 2])
            else: # Odd clients get mode B
                client_indices[i].extend(np.array_split(class1_mode_b_indices, num_clients // 2)[i // 2])
                
        return [Subset(full_dataset, indices) for indices in client_indices]