"""
Quantum-specific client implementations for federated learning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

# Use absolute imports assuming src is in PYTHONPATH or running from root
from src.federated.client import FederatedClient
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel

logger = logging.getLogger(__name__)

class QuantumFederatedClient(FederatedClient):
    """
    Client for federated learning with quantum models.

    Extends FederatedClient to handle VQC (NumPy params) vs PyTorch models.
    """

    def __init__(
        self,
        client_id: str,
        model: Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module],
        dataset: torch.utils.data.Dataset, # Expect PyTorch dataset
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_class: Optional[torch.optim.Optimizer] = None, # Made Optional
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Optional[nn.Module] = None, # Made Optional
        device: Optional[torch.device] = None
    ):
        """
        Initialize a quantum federated client.

        Args:
            client_id: Unique identifier.
            model: Quantum model instance (VQC, QNN, Hybrid, or standard nn.Module).
            dataset: Local PyTorch Dataset.
            batch_size: Batch size for training.
            learning_rate: Learning rate (used by VQC internal fit and PyTorch optimizers).
            optimizer_class: PyTorch optimizer *class* (e.g., torch.optim.Adam) for PyTorch models. Ignored for VQC.
            optimizer_kwargs: Additional keyword arguments for PyTorch optimizer.
            loss_fn: PyTorch loss function instance (e.g., nn.CrossEntropyLoss()). Ignored for VQC.
            device: Device to run training on.
        """
        # Determine model type *before* calling super().__init__
        self.model_type = self._determine_model_type(model)
        logger.info(f"QuantumClient {client_id}: Initializing with model type '{self.model_type}'.")

        # Handle args for super init based on model type
        if self.model_type == "vqc":
            # VQC uses internal optimizer/loss, pass None to parent
            super().__init__(
                client_id=client_id, model=model, dataset=dataset, batch_size=batch_size,
                learning_rate=learning_rate, # VQC uses this LR internally
                optimizer_class=None, optimizer_kwargs=None, loss_fn=None,
                device=device # Pass device even if VQC mainly uses numpy
            )
            # VQC specific: optimizer and loss_fn are managed internally or not applicable directly
            self.optimizer = None # No PyTorch optimizer for VQC
            self.loss_fn = None # Loss calculated within VQC methods
        else:
            # For PyTorch models, ensure optimizer and loss are provided or default
            opt_cls = optimizer_class or torch.optim.SGD # Default optimizer if none provided
            loss_func = loss_fn or nn.CrossEntropyLoss() # Default loss if none provided
            super().__init__(
                client_id=client_id, model=model, dataset=dataset, batch_size=batch_size,
                learning_rate=learning_rate,
                optimizer_class=opt_cls, optimizer_kwargs=optimizer_kwargs, loss_fn=loss_func,
                device=device
            )
            # Optimizer and loss_fn are now set by the parent __init__

        # Model is already moved to device by parent __init__
        logger.debug(f"QuantumClient {client_id}: Initialization complete on device {self.device}.")


    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of quantum model."""
        if isinstance(model, VariationalQuantumClassifier):
            return "vqc"
        elif isinstance(model, QuantumNeuralNetwork):
            # QNN is technically also nn.Module, check it first
            return "qnn"
        elif isinstance(model, HybridQuantumModel):
             # Hybrid is technically also nn.Module, check it first
            return "hybrid"
        elif isinstance(model, nn.Module):
            # Standard PyTorch model (non-quantum specific, but handled)
            return "torch"
        else:
            raise ValueError(f"Unsupported model type for QuantumFederatedClient: {type(model)}")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters, converting VQC NumPy params to Torch tensor."""
        if self.model_type == "vqc":
            # Convert VQC's NumPy params to a Torch tensor on CPU
            logger.debug(f"Client {self.client_id}: Getting VQC parameters and converting to Tensor.")
            if self.model.params is None:
                 logger.error(f"Client {self.client_id}: VQC model parameters are None.")
                 raise ValueError("VQC model parameters not initialized.")
            # Ensure dtype consistency, float64 often used with PennyLane backend
            params_tensor = torch.tensor(self.model.params, dtype=torch.float64).cpu()
            return {"quantum_params": params_tensor}
        else:
            # For PyTorch models, use the parent method (which gets state_dict on CPU)
            logger.debug(f"Client {self.client_id}: Getting {self.model_type} parameters (state_dict).")
            return super().get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set parameters, converting Torch tensor back to NumPy for VQC."""
        if self.model_type == "vqc":
            logger.debug(f"Client {self.client_id}: Setting VQC parameters from Tensor.")
            if "quantum_params" in parameters:
                try:
                    # Convert incoming tensor (likely CPU) to NumPy array
                    vqc_params_np = parameters["quantum_params"].numpy()
                    if self.model.params is not None and self.model.params.shape != vqc_params_np.shape:
                         logger.error(f"VQC parameter shape mismatch. Model: {self.model.params.shape}, Received: {vqc_params_np.shape}")
                         raise ValueError("VQC parameter shape mismatch during set_parameters.")
                    self.model.params = vqc_params_np
                    logger.debug(f"Client {self.client_id}: VQC parameters updated.")
                except Exception as e:
                     logger.error(f"Client {self.client_id}: Error converting/setting VQC parameters: {e}", exc_info=True)
                     raise
            else:
                logger.error(f"Client {self.client_id}: 'quantum_params' key not found in parameters dictionary for VQC.")
                raise KeyError("'quantum_params' key missing in parameters for VQC model.")
        else:
            # For PyTorch models, use the parent method (loads state_dict to model's device)
            logger.debug(f"Client {self.client_id}: Setting {self.model_type} parameters (state_dict).")
            super().set_parameters(parameters)

    def train(
        self,
        epochs: int = 1,
        proximal_term: float = 0.0,
        # Changed: Expect params dict now, not model obj
        global_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Train the local model."""
        logger.info(f"Client {self.client_id}: Starting local training (model type: {self.model_type}).")

        if self.model_type == "vqc":
            # Handle VQC training using modified train_step
            return self._train_vqc_manual_batches(epochs, proximal_term, global_params)
        elif self.model_type in ["qnn", "hybrid", "torch"]:
            # Use the parent's training loop which handles PyTorch models and FedProx
            # Need to pass global_params correctly
            return super().train(epochs, proximal_term, global_params) # Pass dict
        else:
            # Should not happen due to check in __init__
            raise TypeError(f"Training not implemented for model type: {self.model_type}")

    def _train_vqc_manual_batches(
        self,
        epochs: int = 1,
        proximal_term: float = 0.0,
        global_params_dict: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train VQC by iterating DataLoader and calling VQC.train_step.
        This allows correct handling of FedProx.
        """
        if proximal_term > 0 and global_params_dict is None:
            raise ValueError("global_params_dict must be provided for VQC FedProx.")

        global_params_numpy = None
        if proximal_term > 0:
             if "quantum_params" not in global_params_dict:
                  raise KeyError("global_params_dict missing 'quantum_params' key for VQC FedProx.")
             global_params_numpy = global_params_dict["quantum_params"].numpy()

        total_samples_processed = 0
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            samples_in_epoch = 0
            num_batches = len(self.data_loader)

            for batch_idx, (data, target) in enumerate(self.data_loader):
                # VQC train_step expects numpy arrays
                # Data might need flattening depending on VQC encoding/circuit
                batch_size = data.shape[0]
                if data.dim() > 2: # Example: Flatten image data
                    data = data.reshape(batch_size, -1)
                features_np = data.numpy()
                labels_np = target.numpy()
                samples_in_epoch += batch_size

                # Call the VQC's modified train_step
                batch_loss = self.model.train_step(
                    features_np,
                    labels_np,
                    self.learning_rate, # Use client's LR
                    proximal_term,
                    global_params_numpy
                )
                epoch_total_loss += batch_loss * batch_size # Use loss before update

                logger.debug(f"Client {self.client_id} [VQC] - Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{num_batches}, "
                             f"Batch Loss (before step): {batch_loss:.4f}")

            avg_epoch_loss = epoch_total_loss / samples_in_epoch if samples_in_epoch > 0 else 0.0
            total_samples_processed += samples_in_epoch
            logger.info(f"Client {self.client_id} [VQC] - Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

        final_params = self.get_parameters() # Get params as {"quantum_params": tensor}
        logger.info(f"Client {self.client_id} [VQC]: Local training finished. Processed {total_samples_processed} samples.")
        return final_params, total_samples_processed

    def evaluate(self, val_dataset: Optional[torch.utils.data.Dataset] = None) -> Dict[str, float]:
        """Evaluate the model, handling VQC separately."""
        logger.info(f"Client {self.client_id}: Starting evaluation (model type: {self.model_type}).")
        if self.model_type == "vqc":
            return self._evaluate_vqc(val_dataset)
        else:
            # Use parent evaluate for PyTorch models
            return super().evaluate(val_dataset)

    def _evaluate_vqc(self, eval_dataset: Optional[torch.utils.data.Dataset] = None) -> Dict[str, float]:
        """Evaluate the VQC model."""
        dataset_to_eval = eval_dataset if eval_dataset is not None else self.dataset
        if len(dataset_to_eval) == 0:
             logger.warning(f"Client {self.client_id} [VQC]: Evaluation dataset empty.")
             return {'loss': float('nan'), 'accuracy': 0.0}

        # VQC predict works sample by sample, iterate through dataset
        # (DataLoader is less convenient here unless we load all data first)
        logger.debug(f"Client {self.client_id} [VQC]: Evaluating on {len(dataset_to_eval)} samples.")
        correct = 0
        total = 0
        # Note: Calculating loss accurately would require running forward pass
        # again, which might be slow. Often only accuracy is reported from VQC eval.

        for i in range(len(dataset_to_eval)):
            try:
                # Dataset __getitem__ should return (data, target)
                data, target = dataset_to_eval[i]
                features_np = data.numpy()
                # Flatten if needed
                if features_np.ndim > 1:
                     features_np = features_np.flatten()
                label = target.item() if isinstance(target, torch.Tensor) else int(target)

                prediction = self.model.predict(features_np)
                if prediction == label:
                    correct += 1
                total += 1
            except Exception as e:
                 logger.error(f"Client {self.client_id} [VQC]: Error evaluating sample {i}: {e}", exc_info=False)
                 # Optionally break or continue

        accuracy = correct / total if total > 0 else 0.0
        metrics = {'accuracy': accuracy, 'loss': None} # Loss not easily calculated here
        logger.info(f"Client {self.client_id} [VQC]: Evaluation finished. Metrics: {metrics}")
        return metrics