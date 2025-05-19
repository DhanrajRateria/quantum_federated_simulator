"""
Federated Quantum Learning Manager and Server.

Provides utilities for managing quantum models in federated learning and
a specialized server class to handle quantum parameter types.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Type

# Use absolute imports
from src.federated.server import FederatedServer
from src.federated.aggregation import AggregationStrategy, FedAvg
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel
from src.core.quantum_client import QuantumFederatedClient # Use the new quantum client

logger = logging.getLogger(__name__)

class QuantumFederatedServer(FederatedServer):
    """
    Server specialized for federated learning with quantum models (all nn.Modules).
    """

    def __init__(
        self,
        model: Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module],
        aggregation_strategy: AggregationStrategy,
        evaluation_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        self.model_type = self._determine_model_type(model) # For logging
        logger.info(f"QuantumServer: Initializing with global model type '{self.model_type}'.")

        super().__init__(
            model=model,
            aggregation_strategy=aggregation_strategy,
            evaluation_dataset=evaluation_dataset,
            device=device
        )

    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of quantum model for informational purposes."""
        if isinstance(model, VariationalQuantumClassifier): return "vqc"
        elif isinstance(model, QuantumNeuralNetwork): return "qnn"
        elif isinstance(model, HybridQuantumModel): return "hybrid"
        elif isinstance(model, nn.Module): return "torch_generic"
        else:
            logger.warning(f"Unsupported model type for QuantumFederatedServer: {type(model)}")
            return "unknown"

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global model parameters (state_dict on CPU). VQC now uses state_dict."""
        logger.debug(f"Server: Getting global {self.model_type} parameters (state_dict).")
        return super().get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set global model parameters (state_dict). VQC now uses state_dict."""
        logger.debug(f"Server: Setting global {self.model_type} parameters (state_dict).")
        super().set_parameters(parameters)

    def evaluate(self, dataset=None):
        """Override to ensure correct device handling for quantum models"""
        logger.info(f"Server: Starting evaluation of global model (type: {self.model_type}).")
        
        if dataset is None:
            dataset = self.evaluation_dataset
        
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_samples = 0
            correct = 0
            total_loss = 0.0
            
            # Use CrossEntropyLoss as default loss function for evaluation
            loss_fn = nn.CrossEntropyLoss()
            
            # Move loss function to same device as model
            loss_fn = loss_fn.to(self.device)
            
            # Create a dataloader for evaluation
            eval_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
            
            for data, target in eval_loader:
                # Move data and target to the model's device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Ensure output is on same device as target before loss calculation
                output = output.to(self.device)
                
                # Calculate loss
                loss = loss_fn(output, target)
                
                # Update metrics
                current_batch_size = data.size(0)
                total_loss += loss.item() * current_batch_size
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total_samples += current_batch_size
        
        # Calculate overall metrics
        accuracy = correct / total_samples if total_samples > 0 else 0
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        logger.info(f"Evaluation results: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        return {"loss": avg_loss, "accuracy": accuracy}


class QuantumAggregationStrategy(AggregationStrategy):
    """
    Aggregation strategy specialized for quantum models.

    Handles VQC's specific parameter structure {"quantum_params": tensor},
    otherwise delegates to a base strategy (e.g., FedAvg) for state_dict structures.
    """

    def __init__(self, base_strategy: Optional[AggregationStrategy] = None):
        """
        Initialize the quantum aggregation strategy.

        Args:
            base_strategy: Underlying aggregation strategy (e.g., FedAvg, Median)
                           to use for non-VQC models or as basis for VQC. Defaults to FedAvg.
        """
        self.base_strategy = base_strategy or FedAvg()
        logger.debug(f"QuantumAggregationStrategy initialized with base: {type(self.base_strategy).__name__}")

    def aggregate(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        if not client_updates:
            logger.warning("QuantumAggregationStrategy: No client updates provided.")
            return {}

        # The old VQC check `list(first_params.keys()) == ["quantum_params"]`
        # is no longer relevant as VQC now has a full state_dict.
        # We can assume all updates are state_dicts and delegate.
        first_params = client_updates[0][0]
        # Example: Check if TorchLayer names parameters like 'q_layer.params'
        is_qnn_like_update = any(key.startswith("q_layer.") for key in first_params.keys())

        if is_qnn_like_update : # Or any other specific check for quantum layer params if needed
             logger.info(f"QuantumAggregationStrategy: Detected nn.Module (QNN/VQC/Hybrid) state_dict. "
                         f"Delegating to base strategy: {type(self.base_strategy).__name__}")
        else:
             logger.info(f"QuantumAggregationStrategy: Standard state_dict detected. "
                         f"Delegating to base strategy: {type(self.base_strategy).__name__}")

        return self.base_strategy.aggregate(client_updates)

    def _aggregate_vqc_params(self, client_updates: List[Tuple[Dict[str, torch.Tensor], float]]) -> Dict[str, torch.Tensor]:
        """Aggregate VQC model parameters based on the 'quantum_params' tensor."""
        num_clients = len(client_updates)
        logger.debug(f"Aggregating VQC 'quantum_params' from {num_clients} clients.")

        # Extract quantum parameters tensor and weights
        param_tensors = []
        weights_list = []
        for params_dict, weight in client_updates:
             if "quantum_params" in params_dict:
                 param_tensors.append(params_dict["quantum_params"].float()) # Ensure float for aggregation
                 weights_list.append(float(weight)) # Ensure float
             else:
                  logger.warning("VQC aggregation expected 'quantum_params' key, not found. Skipping update.")

        if not param_tensors:
             logger.error("VQC aggregation failed: No valid 'quantum_params' found in updates.")
             return {} # Or raise error

        weights = np.array(weights_list, dtype=np.float32)
        total_weight = weights.sum()
        if total_weight <= 0:
            logger.warning("VQC Aggregation: Total weight is zero, using equal weights.")
            normalized_weights = np.ones(len(param_tensors), dtype=np.float32) / len(param_tensors)
        else:
            normalized_weights = weights / total_weight

        # Initialize result tensor (use dtype of first tensor)
        result_tensor = torch.zeros_like(param_tensors[0])

        # Perform weighted averaging
        for i, param_tensor in enumerate(param_tensors):
            result_tensor += normalized_weights[i] * param_tensor

        logger.debug("VQC parameter aggregation complete.")
        # Return in the expected dictionary format
        return {"quantum_params": result_tensor.to(param_tensors[0].dtype)} # Match original dtype


class FederatedQuantumManager:
    @staticmethod
    def create_quantum_clients(
        client_ids: List[str],
        model_class: Type[Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module]],
        model_kwargs: Dict[str, Any],
        datasets: List[Dataset],
        client_kwargs: Optional[Dict[str, Any]] = None # Args for QuantumFederatedClient init
    ) -> List[QuantumFederatedClient]:
        if len(client_ids) != len(datasets):
            raise ValueError("Number of client IDs must match number of datasets")
        # All model_class options are now expected to be nn.Module or inherit from it
        if not issubclass(model_class, nn.Module):
             raise TypeError("model_class must be a PyTorch nn.Module or inherit from it.")

        clients = []
        client_setup_kwargs = client_kwargs or {}
        # Ensure defaults for optimizer and loss if not provided in client_setup_kwargs
        # These will be passed to QuantumFederatedClient, which now requires them.
        client_setup_kwargs.setdefault('optimizer_class', torch.optim.SGD)
        client_setup_kwargs.setdefault('loss_fn', nn.CrossEntropyLoss())
        # learning_rate should also be present in client_setup_kwargs or have a default in QFC init

        logger.info(f"Creating {len(client_ids)} quantum clients with model {model_class.__name__}.")

        for i, (client_id, dataset) in enumerate(zip(client_ids, datasets)):
            try:
                model_instance = model_class(**model_kwargs)
            except Exception as e:
                 logger.error(f"Failed to instantiate model {model_class.__name__} for client {client_id}: {e}", exc_info=True)
                 raise

            try:
                client = QuantumFederatedClient(
                    client_id=client_id,
                    model=model_instance,
                    dataset=dataset,
                    **client_setup_kwargs # This now includes optimizer_class, loss_fn, lr, batch_size etc.
                )
                clients.append(client)
                logger.debug(f"Successfully created quantum client '{client_id}' with {len(dataset)} samples.")
            except Exception as e:
                 logger.error(f"Failed to create QuantumFederatedClient '{client_id}': {e}", exc_info=True)
                 raise
        return clients

    @staticmethod
    def initialize_quantum_server(
        model_class: Type[Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module]],
        model_kwargs: Dict[str, Any],
        aggregation_strategy: AggregationStrategy,
        evaluation_dataset: Optional[Dataset] = None,
        server_kwargs: Optional[Dict[str, Any]] = None
    ) -> QuantumFederatedServer:
        if not issubclass(model_class, nn.Module):
             raise TypeError("model_class must be a PyTorch nn.Module or inherit from it.")

        logger.info(f"Initializing quantum server with global model {model_class.__name__}.")
        try:
            global_model = model_class(**model_kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate global model {model_class.__name__}: {e}", exc_info=True)
            raise

        server_setup_kwargs = server_kwargs or {}
        try:
            server = QuantumFederatedServer(
                model=global_model,
                aggregation_strategy=aggregation_strategy,
                evaluation_dataset=evaluation_dataset,
                **server_setup_kwargs
            )
            logger.info(f"Quantum federated server initialized successfully.")
            return server
        except Exception as e:
             logger.error(f"Failed to initialize QuantumFederatedServer: {e}", exc_info=True)
             raise