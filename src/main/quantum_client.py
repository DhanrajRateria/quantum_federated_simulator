"""
Quantum-specific client implementations for federated learning.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union

from src.federated.client import FederatedClient
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.quantum.utils import set_random_seed

logger = logging.getLogger(__name__)


class QuantumFederatedClient(FederatedClient):
    """
    Client for federated learning with quantum models.
    
    This class extends the standard FederatedClient to handle quantum models,
    including parameter conversion and quantum-specific training routines.
    """
    
    def __init__(
        self,
        client_id: str,
        model: Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module],
        dataset: Any,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_class: Any = None,  # Will be handled differently based on model type
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: Any = None  # Will be handled differently based on model type
    ):
        """
        Initialize a quantum federated client.
        
        Args:
            client_id: Unique identifier for the client
            model: Quantum model to train
            dataset: Local dataset for training
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            optimizer_class: Optional PyTorch optimizer class (for PyTorch-based models)
            optimizer_kwargs: Additional keyword arguments for optimizer
            loss_fn: Optional loss function
        """
        # Call the parent constructor for common initialization
        super().__init__(
            client_id=client_id,
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer_class=optimizer_class if not isinstance(model, VariationalQuantumClassifier) else None,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn
        )
        
        # Specific handling for different quantum model types
        self.model_type = self._determine_model_type(model)
        logger.debug(f"Initialized quantum federated client with {self.model_type} model type")
    
    def _determine_model_type(self, model: Any) -> str:
        """
        Determine the type of quantum model.
        
        Args:
            model: The model to check
            
        Returns:
            String indicating the model type
        """
        if isinstance(model, VariationalQuantumClassifier):
            return "vqc"
        elif isinstance(model, QuantumNeuralNetwork):
            return "qnn"
        elif isinstance(model, HybridQuantumModel):
            return "hybrid"
        elif isinstance(model, nn.Module):
            return "torch"
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the current model parameters, handling different quantum model types.
        
        Returns:
            Dictionary of parameter name -> tensor
        """
        if self.model_type == "vqc":
            # For VariationalQuantumClassifier, convert numpy parameters to torch tensors
            params_tensor = torch.tensor(self.model.params)
            return {"quantum_params": params_tensor}
        elif self.model_type in ["qnn", "hybrid", "torch"]:
            # For PyTorch-based models, use the standard approach
            return {name: param.data.clone() for name, param in self.model.named_parameters()}
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Update local model with provided parameters, handling different quantum model types.
        
        Args:
            parameters: Dictionary of parameter name -> tensor
        """
        if self.model_type == "vqc":
            # For VariationalQuantumClassifier, convert torch tensor to numpy array
            if "quantum_params" in parameters:
                self.model.params = parameters["quantum_params"].numpy()
            else:
                logger.warning("No quantum_params found in parameters dictionary")
        elif self.model_type in ["qnn", "hybrid", "torch"]:
            # For PyTorch-based models, use the standard approach
            for name, param in self.model.named_parameters():
                if name in parameters:
                    param.data = parameters[name].clone()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(
        self, 
        epochs: int = 1, 
        proximal_term: float = 0.0,
        global_model: Optional[nn.Module] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Train the quantum model on local data.
        
        Args:
            epochs: Number of local epochs
            proximal_term: Coefficient for proximal term (FedProx)
            global_model: Global model for proximal term calculation
            
        Returns:
            Tuple containing:
                - Updated model parameters
                - Number of samples used for training
        """
        logger.debug(f"Training quantum model for {epochs} epochs")
        
        if self.model_type == "vqc":
            # Handle training for VariationalQuantumClassifier
            return self._train_vqc(epochs, proximal_term, global_model)
        elif self.model_type in ["qnn", "hybrid", "torch"]:
            # For PyTorch-based models, use the standard approach with some adaptations
            return super().train(epochs, proximal_term, global_model)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
    def _train_vqc(
        self, 
        epochs: int = 1, 
        proximal_term: float = 0.0,
        global_model: Optional[Any] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Train a VariationalQuantumClassifier model.
        
        Args:
            epochs: Number of local epochs
            proximal_term: Coefficient for proximal term (FedProx)
            global_model: Global model for proximal term calculation
            
        Returns:
            Tuple containing:
                - Updated model parameters
                - Number of samples used for training
        """
        # Save initial parameters if using proximal term
        if proximal_term > 0 and global_model is not None and isinstance(global_model, VariationalQuantumClassifier):
            global_params = global_model.params.copy()
        
        total_samples = 0
        
        # Convert dataset to numpy arrays for VQC training
        features_list = []
        labels_list = []
        
        # Extract features and labels from dataset
        for data, target in self.data_loader:
            batch_size = data.shape[0]
            total_samples += batch_size
            
            # Flatten input data if needed
            if len(data.shape) > 2:
                data = data.reshape(batch_size, -1)
            
            # Convert to numpy
            features_list.append(data.numpy())
            labels_list.append(target.numpy())
        
        # Concatenate batches
        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        # Train the model using VQC's native training method
        self.model.fit(
            features=features,
            labels=labels,
            epochs=epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=False
        )
        
        # Add proximal term calculation if needed
        if proximal_term > 0 and global_model is not None and isinstance(global_model, VariationalQuantumClassifier):
            # Apply proximal regularization - modify parameters to be closer to global model
            self.model.params = self.model.params - proximal_term * (self.model.params - global_params)
        
        # Return updated parameters
        return self.get_parameters(), total_samples