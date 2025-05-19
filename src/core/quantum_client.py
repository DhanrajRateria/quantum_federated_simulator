"""
Quantum-specific client implementations for federated learning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any, Union, Callable


# Use absolute imports assuming src is in PYTHONPATH or running from root
from src.federated.client import FederatedClient
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel

logger = logging.getLogger(__name__)

class QuantumFederatedClient(FederatedClient):
    """
    Client for federated learning with quantum models.
    Handles VQC, QNN, Hybrid models, all now nn.Modules.
    """

    def __init__(
        self,
        client_id: str,
        model: Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module],
        dataset: Dataset, # Expect PyTorch dataset
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_class: torch.optim.Optimizer = torch.optim.SGD, # Now required, with default
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss(), # Now required, with default
        device: Optional[torch.device] = None
    ):
        """
        Initialize a quantum federated client. All models are expected to be nn.Module.

        Args:
            client_id: Unique identifier.
            model: Quantum model instance (VQC, QNN, Hybrid, or standard nn.Module).
            dataset: Local PyTorch Dataset.
            batch_size: Batch size for training.
            learning_rate: Learning rate for the optimizer.
            optimizer_class: PyTorch optimizer *class*.
            optimizer_kwargs: Additional keyword arguments for PyTorch optimizer.
            loss_fn: PyTorch loss function instance.
            device: Device to run training on.
        """
        self.model_type = self._determine_model_type(model) # Determine type for logging/debug
        logger.info(f"QuantumClient {client_id}: Initializing with model type '{self.model_type}'.")

        # All model types (VQC, QNN, Hybrid) are now nn.Module and use PyTorch optimizers/loss
        super().__init__(
            client_id=client_id, model=model, dataset=dataset, batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class, # Pass directly
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn, # Pass directly
            device=device
        )
        # Optimizer and loss_fn are set by the parent __init__

        logger.debug(f"QuantumClient {client_id}: Initialization complete on device {self.device}. "
                     f"Optimizer: {type(self.optimizer).__name__}, Loss: {type(self.loss_fn).__name__}")


    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of quantum model for informational purposes."""
        # Order matters if classes inherit (e.g. Hybrid is an nn.Module)
        if isinstance(model, VariationalQuantumClassifier):
            return "vqc"
        elif isinstance(model, QuantumNeuralNetwork):
            return "qnn"
        elif isinstance(model, HybridQuantumModel):
            return "hybrid"
        elif isinstance(model, nn.Module):
            return "torch_generic" # Generic PyTorch model
        else:
            # This case should ideally not be hit if model is always nn.Module
            logger.warning(f"Unsupported model type passed to QuantumFederatedClient: {type(model)}")
            return "unknown"

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get model parameters as a state dictionary (on CPU).
        VQC now uses state_dict like other nn.Modules.
        """
        logger.debug(f"Client {self.client_id}: Getting {self.model_type} parameters (state_dict).")
        return super().get_parameters() # Parent handles nn.Module state_dict

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Set model parameters from a state dictionary.
        VQC now uses state_dict like other nn.Modules.
        """
        logger.debug(f"Client {self.client_id}: Setting {self.model_type} parameters (state_dict).")
        super().set_parameters(parameters) # Parent handles nn.Module state_dict

    def train(
        self,
        epochs: int = 1,
        proximal_term: float = 0.0,
        global_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train the local model. VQC now uses the standard PyTorch training loop from parent.
        """
        logger.info(f"Client {self.client_id}: Starting local training for {epochs} epochs (model type: {self.model_type}). "
                    f"Proximal term mu={proximal_term}.")
        # All models (VQC, QNN, Hybrid, generic Torch) use the parent's training loop
        return super().train(epochs, proximal_term, global_params)

    # _train_vqc_manual_batches method is REMOVED.

    def evaluate(self, val_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model. VQC now uses the standard PyTorch evaluation loop from parent.
        The parent's evaluate method calculates loss and accuracy.
        """
        logger.info(f"Client {self.client_id}: Starting evaluation (model type: {self.model_type}).")
        # All models use the parent's evaluate method
        return super().evaluate(val_dataset)