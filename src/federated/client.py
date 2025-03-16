"""
Client implementation for quantum federated learning.
This module defines the client-side logic for participating in a federated
learning process using quantum machine learning models.
"""

import numpy as np
import json
import logging
import os
import time
from typing import List, Dict, Optional, Tuple, Union, Callable, Any

# Import quantum model
import sys
import os

# Add the parent directory to sys.path to resolve imports
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..quantum.models import create_quantum_model

logger = logging.getLogger(__name__)

class QuantumFederatedClient:
    """Client for quantum federated learning."""
    
    def __init__(self, client_id: str, config: Dict):
        """
        Initialize a federated learning client.
        
        Args:
            client_id: Unique identifier for this client
            config: Dictionary containing client configuration
        """
        self.client_id = client_id
        self.config = config
        
        # Initialize model
        self.model_config = config.get('model', {})
        self.model = create_quantum_model(self.model_config)
        
        # Training configuration
        self.local_epochs = config.get('local_epochs', 5)
        self.batch_size = config.get('batch_size', 4)
        self.learning_rate = config.get('learning_rate', 0.01)
        
        # Data attributes
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None
        
        # Communication configuration
        self.server_url = config.get('server_url', 'http://localhost:5000')
        
        # Privacy settings
        self.differential_privacy = config.get('differential_privacy', False)
        self.dp_epsilon = config.get('dp_epsilon', 1.0)
        
        # Metrics collection
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'train_time': [],
            'communication_overhead': []
        }
        
        logger.info(f"Initialized client {client_id} with {type(self.model).__name__}")
    
    def load_data(self, data: Dict[str, np.ndarray]) -> None:
        """
        Load data for training and validation.
        
        Args:
            data: Dictionary containing 'x_train', 'y_train', 'x_val', 'y_val'
        """
        self.train_data = data.get('x_train')
        self.train_labels = data.get('y_train')
        self.val_data = data.get('x_val')
        self.val_labels = data.get('y_val')
        
        n_train = len(self.train_data) if self.train_data is not None else 0
        n_val = len(self.val_data) if self.val_data is not None else 0
        logger.info(f"Client {self.client_id} loaded {n_train} training samples and {n_val} validation samples")
    
    def train(self, global_round: int) -> Dict:
        """
        Perform local training for one federated round.
        
        Args:
            global_round: Current global round number
            
        Returns:
            Training metrics
        """
        if self.train_data is None or self.train_labels is None:
            raise ValueError("Training data not loaded. Call load_data first.")
        
        start_time = time.time()
        
        # Train the model
        history = self.model.fit(
            self.train_data, 
            self.train_labels,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            verbose=(global_round % 5 == 0)  # Print progress every 5 rounds
        )
        
        train_time = time.time() - start_time
        
        # Evaluate on validation set if available
        val_loss = None
        if self.val_data is not None and self.val_labels is not None:
            predictions = self.model.predict(self.val_data)
            val_loss = self.model.compute_loss(predictions, self.val_labels)
        
        # Record metrics
        self.metrics['train_loss'].append(history['loss'][-1])
        if val_loss is not None:
            self.metrics['val_loss'].append(val_loss)
        self.metrics['train_time'].append(train_time)
        
        logger.info(f"Client {self.client_id} completed training for round {global_round}")
        logger.info(f"Training took {train_time:.2f} seconds, final loss: {history['loss'][-1]:.4f}")
        
        return {
            'train_loss': history['loss'][-1],
            'val_loss': val_loss,
            'train_time': train_time,
            'parameters_size': self._get_parameters_size()
        }
    
    def get_parameters(self) -> np.ndarray:
        """
        Get current model parameters.
        
        Returns:
            Model parameters
        """
        return self.model.get_parameters()
    
    def set_parameters(self, parameters: np.ndarray) -> None:
        """
        Update model with parameters from server.
        
        Args:
            parameters: New global parameters
        """
        self.model.set_parameters(parameters)
        logger.info(f"Client {self.client_id} updated parameters from server")
    
    def apply_differential_privacy(self, parameters: np.ndarray) -> np.ndarray:
        """
        Apply differential privacy to parameters before sharing.
        
        Args:
            parameters: Original parameters
            
        Returns:
            Parameters with noise added
        """
        if not self.differential_privacy:
            return parameters
        
        # Simple implementation of differential privacy
        # Add noise calibrated to sensitivity and privacy parameter epsilon
        sensitivity = 1.0  # Assuming parameters are normalized
        noise_scale = sensitivity / self.dp_epsilon
        noise = np.random.laplace(0, noise_scale, parameters.shape)
        
        return parameters + noise
    
    def _get_parameters_size(self) -> int:
        """
        Calculate size of model parameters in bytes.
        
        Returns:
            Size in bytes
        """
        params = self.get_parameters()
        return params.nbytes
    
    def serialize_parameters(self) -> Dict:
        """
        Serialize parameters for transmission.
        
        Returns:
            Dictionary with serialized parameters
        """
        params = self.get_parameters()
        
        # Apply privacy mechanism if enabled
        if self.differential_privacy:
            params = self.apply_differential_privacy(params)
        
        # Record communication overhead
        overhead = params.nbytes
        self.metrics['communication_overhead'].append(overhead)
        
        # Convert to serializable format
        if isinstance(params, np.ndarray):
            # Handle different numpy array shapes
            return {
                'format': 'numpy',
                'shape': params.shape,
                'dtype': str(params.dtype),
                'data': params.tolist()
            }
        else:
            # Handle other parameter types
            return {
                'format': 'list',
                'data': params
            }
    
    def deserialize_parameters(self, serialized_params: Dict) -> np.ndarray:
        """
        Deserialize parameters received from server.
        
        Args:
            serialized_params: Dictionary with serialized parameters
            
        Returns:
            Deserialized parameters
        """
        if serialized_params['format'] == 'numpy':
            # Reconstruct numpy array
            shape = tuple(serialized_params['shape'])
            dtype = np.dtype(serialized_params['dtype'])
            return np.array(serialized_params['data'], dtype=dtype).reshape(shape)
        else:
            # Handle other parameter types
            return serialized_params['data']
    
    def save_metrics(self, output_dir: str) -> None:
        """
        Save client metrics to file.
        
        Args:
            output_dir: Directory to save metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, f"client_{self.client_id}_metrics.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_file}")
    
    def evaluate(self, data: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Evaluate current model on data.
        
        Args:
            data: Dictionary with 'x' and 'y' for evaluation,
                 or None to use validation data
                 
        Returns:
            Evaluation metrics
        """
        if data is None:
            if self.val_data is None or self.val_labels is None:
                raise ValueError("No validation data available")
            x = self.val_data
            y = self.val_labels
        else:
            x = data['x']
            y = data['y']
        
        predictions = self.model.predict(x)
        loss = self.model.compute_loss(predictions, y)
        
        # For binary classification, calculate accuracy
        if y.ndim == 1 or y.shape[1] == 1:
            binary_preds = np.round(predictions).astype(int)
            accuracy = np.mean(binary_preds == y)
        else:
            # For multi-class, take the highest value
            class_preds = np.argmax(predictions, axis=1)
            true_classes = np.argmax(y, axis=1)
            accuracy = np.mean(class_preds == true_classes)
        
        return {
            'loss': float(loss),
            'accuracy': float(accuracy)
        }