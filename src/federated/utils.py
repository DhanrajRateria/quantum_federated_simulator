"""
Utility functions for federated learning.

This module provides helper functions for data partitioning, model manipulation,
metrics calculation, and other utilities used across the federated learning system.
"""

import logging
import random
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, random_split
import yaml

logger = logging.getLogger(__name__)


class FederatedDataset:
    """Utility class for partitioning datasets among clients in various ways."""
    
    @staticmethod
    def iid_partition(dataset: Dataset, num_clients: int) -> List[Subset]:
        """
        Partition dataset in IID (Independent and Identically Distributed) fashion.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients to distribute data to
            
        Returns:
            List of dataset subsets, one for each client
        """
        total_size = len(dataset)
        partition_size = total_size // num_clients
        sizes = [partition_size] * num_clients
        
        # Adjust for any remaining samples
        sizes[-1] += total_size - sum(sizes)
        
        return random_split(dataset, sizes)
        
    @staticmethod
    def non_iid_label_partition(
        dataset: Dataset, 
        num_clients: int, 
        num_classes: int,
        classes_per_client: int = 2,
        balance_factor: float = 1.0
    ) -> List[Subset]:
        """
        Partition dataset in non-IID fashion based on class labels.
        Each client gets a subset of classes with potentially uneven distribution.
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients to distribute data to
            num_classes: Total number of classes in the dataset
            classes_per_client: Number of classes each client should have
            balance_factor: Controls how balanced the classes are (1.0 is balanced)
            
        Returns:
            List of dataset subsets, one for each client
        """
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

        if not hasattr(dataset, 'targets') and not hasattr(dataset, 'train_labels'):
            raise ValueError("Dataset must have a 'targets' or 'train_labels' attribute for non-IID partitioning")
        
        labels = dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels
            
        # Get indices for each class
        class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
            
        # Assign classes to clients
        client_class_indices = [[] for _ in range(num_clients)]
        for client_id in range(num_clients):
            # Select classes for this client
            client_classes = random.sample(
                range(num_classes), 
                min(classes_per_client, num_classes)
            )
            
            # Get indices for these classes
            for class_id in client_classes:
                # Apply balance factor
                if balance_factor < 1.0:
                    # Take a smaller fraction of the class indices
                    sample_size = int(len(class_indices[class_id]) * balance_factor)
                    sampled_indices = random.sample(class_indices[class_id], sample_size)
                    client_class_indices[client_id].extend(sampled_indices)
                else:
                    client_class_indices[client_id].extend(class_indices[class_id])
        
        # Create subsets
        return [Subset(dataset, indices) for indices in client_class_indices]
        
    @staticmethod
    def dirichlet_partition(
        dataset: Dataset,
        num_clients: int,
        num_classes: int,
        alpha: float = 0.5
    ) -> List[Subset]:
        """
        Partition dataset using Dirichlet distribution to create unbalanced, non-IID splits.
        Lower alpha = more non-IID (e.g., 0.1), higher alpha = more balanced (e.g., 10.0)
        
        Args:
            dataset: The dataset to partition
            num_clients: Number of clients to distribute data to
            num_classes: Total number of classes in the dataset
            alpha: Concentration parameter for Dirichlet distribution
            
        Returns:
            List of dataset subsets, one for each client
        """

        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

        if not hasattr(dataset, 'targets') and not hasattr(dataset, 'train_labels'):
            raise ValueError("Dataset must have a 'targets' or 'train_labels' attribute for non-IID partitioning")
        
        labels = dataset.targets if hasattr(dataset, 'targets') else dataset.train_labels
        
        # Get indices for each class
        class_indices = [[] for _ in range(num_classes)]
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
            
        # Sample from Dirichlet distribution
        client_proportions = np.random.dirichlet(
            alpha=[alpha] * num_clients, 
            size=num_classes
        )
        
        # Get indices for each client
        client_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            # Distribute indices according to proportions
            class_size = len(class_indices[class_id])
            
            # Calculate how many samples of this class go to each client
            client_sample_sizes = np.round(
                client_proportions[class_id] * class_size
            ).astype(int)
            
            # Adjust to ensure we use all samples
            class_indices_copy = class_indices[class_id].copy()
            random.shuffle(class_indices_copy)
            
            # Distribute to clients
            start_idx = 0
            for client_id, size in enumerate(client_sample_sizes):
                if client_id == num_clients - 1:
                    client_indices[client_id].extend(
                        class_indices_copy[start_idx:]
                    )
                else:
                    client_indices[client_id].extend(
                        class_indices_copy[start_idx:start_idx + size]
                    )
                    start_idx += size
                
        # Create subsets
        return [Subset(dataset, indices) for indices in client_indices]


class ModelUtils:
    """Utilities for handling model parameters and operations."""
    
    @staticmethod
    def get_model_size(model: nn.Module) -> int:
        """
        Calculate the total number of parameters in the model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in model.parameters())
        
    @staticmethod
    def flatten_params(parameters: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Flatten model parameters into a single vector.
        
        Args:
            parameters: Dictionary of parameter name -> tensor
            
        Returns:
            Flattened parameters as a 1D tensor
        """
        return torch.cat([param.view(-1) for param in parameters.values()])
        
    @staticmethod
    def unflatten_params(
        flattened: torch.Tensor,
        like_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Reshape a flattened vector back to the original parameter shapes.
        
        Args:
            flattened: Flattened parameters as a 1D tensor
            like_params: Dictionary with the desired parameter shapes
            
        Returns:
            Dictionary of parameter name -> tensor
        """
        unflattened = {}
        index = 0
        for name, param in like_params.items():
            param_size = param.numel()
            unflattened[name] = flattened[index:index + param_size].view(param.size())
            index += param_size
        return unflattened
    
    @staticmethod
    def clone_model(model: nn.Module) -> nn.Module:
        """
        Create a deep copy of a model with the same architecture.
        
        Args:
            model: PyTorch model to clone
            
        Returns:
            A new instance of the model with copied parameters
        """
        clone = type(model)()  # Create new instance of the same class
        clone.load_state_dict(model.state_dict())
        return clone
    
    @staticmethod
    def serialize_model_params(parameters: Dict[str, torch.Tensor]) -> Dict[str, List]:
        """
        Serialize model parameters for transmission or storage.
        
        Args:
            parameters: Dictionary of parameter name -> tensor
            
        Returns:
            Serialized parameters as a dictionary of lists
        """
        return {name: param.detach().cpu().numpy().tolist() for name, param in parameters.items()}
    
    @staticmethod
    def deserialize_model_params(serialized: Dict[str, List]) -> Dict[str, torch.Tensor]:
        """
        Deserialize model parameters from transmission or storage.
        
        Args:
            serialized: Serialized parameters as a dictionary of lists
            
        Returns:
            Dictionary of parameter name -> tensor
        """
        return {name: torch.tensor(param) for name, param in serialized.items()}


class MetricsUtils:
    """Utilities for computing and tracking metrics."""
    
    @staticmethod
    def compute_accuracy(model: nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
        """
        Compute classification accuracy of a model on a dataset.
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader for the evaluation dataset
            
        Returns:
            Accuracy as a float between 0 and 1
        """
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
        
    @staticmethod
    def compute_loss(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    ) -> float:
        """
        Compute average loss of a model on a dataset.
        
        Args:
            model: PyTorch model to evaluate
            dataloader: DataLoader for the evaluation dataset
            loss_fn: Loss function
            
        Returns:
            Average loss as a float
        """
        model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return total_loss / total_samples


class ConfigUtils:
    """Utilities for handling configuration."""
    
    @staticmethod
    def load_yaml_config(config_path: str) -> Dict:
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration as a dictionary
        """
        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            try:
                config = yaml.safe_load(f)
                return config
            except yaml.YAMLError as e:
                logger.error(f"Error loading YAML config: {e}")
                return {}
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configuration dictionaries, with override taking precedence.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base values
            
        Returns:
            Merged configuration
        """
        merged = base_config.copy()
        
        def _merge_dicts(base, override):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    _merge_dicts(base[key], value)
                else:
                    base[key] = value
                    
        _merge_dicts(merged, override_config)
        return merged


def setup_logger(
    name: str = "federated",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure a logger for the federated learning system.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: File to log to (None for no file logging)
        console_output: Whether to output logs to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False