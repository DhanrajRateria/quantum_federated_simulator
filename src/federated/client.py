"""
Federated learning client implementation.

This module provides the client-side functionality for federated learning,
handling local model training and communication with the server.
"""

import logging
import copy
from typing import Dict, Optional, Tuple, List, Any, Union, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class FederatedClient:
    """
    Client in a federated learning system.
    
    Responsible for:
    - Training on local data
    - Computing model updates
    - Sending updates to server
    - Receiving global model updates
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        dataset: Dataset,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        optimizer_class: torch.optim.Optimizer = torch.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        loss_fn: nn.Module = nn.CrossEntropyLoss()
    ):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            model: PyTorch model to train
            dataset: Local dataset for training
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            optimizer_class: PyTorch optimizer class
            optimizer_kwargs: Additional keyword arguments for optimizer
            loss_fn: Loss function for training
        """
        self.client_id = client_id
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        
        # Set up optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(
            self.model.parameters(), 
            lr=learning_rate, 
            **optimizer_kwargs
        )
        
        # Setup data loader
        self.data_loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True
        )
        
        logger.debug(f"Client {client_id} initialized with {len(dataset)} samples")
        
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Get the current model parameters.
        
        Returns:
            Dictionary of parameter name -> tensor
        """
        return {name: param.data.clone() for name, param in self.model.named_parameters()}
        
    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """
        Update local model with provided parameters.
        
        Args:
            parameters: Dictionary of parameter name -> tensor
        """
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data = parameters[name].clone()
        
    def train(
        self, 
        epochs: int = 1, 
        proximal_term: float = 0.0,
        global_model: Optional[nn.Module] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Train the model on local data.
        
        Args:
            epochs: Number of local epochs
            proximal_term: Coefficient for proximal term (FedProx)
            global_model: Global model for proximal term calculation
            
        Returns:
            Tuple containing:
                - Updated model parameters
                - Number of samples used for training
        """
        self.model.train()
        
        # Save initial parameters if using proximal term
        if proximal_term > 0 and global_model is not None:
            global_params = {name: param.data.clone() for name, param in global_model.named_parameters()}
        
        total_samples = 0
        for epoch in range(epochs):
            running_loss = 0.0
            samples_this_epoch = 0
            
            for batch_idx, (data, target) in enumerate(self.data_loader):
                batch_size = data.shape[0]
                total_samples += batch_size
                samples_this_epoch += batch_size
                
                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                
                # Add proximal term if specified (for FedProx)
                if proximal_term > 0 and global_model is not None:
                    proximal_loss = 0
                    for name, param in self.model.named_parameters():
                        proximal_loss += torch.sum((param - global_params[name]) ** 2)
                    loss += (proximal_term / 2) * proximal_loss
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item() * batch_size
            
            epoch_loss = running_loss / samples_this_epoch
            logger.debug(f"Client {self.client_id} - Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        return self.get_parameters(), total_samples
    
    def evaluate(self, val_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            val_dataset: Dataset to use for evaluation (if None, uses training data)
        
        Returns:
            Dictionary with metrics (loss, accuracy)
        """
        if val_dataset is None:
            val_dataset = self.dataset
            
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                output = self.model(data)
                total_loss += self.loss_fn(output, target).item() * data.shape[0]
                
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
                total += target.size(0)
        
        metrics = {
            'loss': total_loss / total,
            'accuracy': correct / total
        }
        
        logger.debug(f"Client {self.client_id} - Evaluation: {metrics}")
        return metrics