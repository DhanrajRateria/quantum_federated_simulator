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
    Server specialized for federated learning with potentially mixed quantum models.

    Extends FederatedServer to handle parameter conversion for VQC models.
    """

    def __init__(
        self,
        model: Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module],
        aggregation_strategy: AggregationStrategy,
        evaluation_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a quantum federated server.

        Args:
            model: Global model instance (VQC, QNN, Hybrid, or nn.Module).
            aggregation_strategy: Strategy for aggregating client updates.
            evaluation_dataset: Optional PyTorch Dataset for server-side evaluation.
            device: Device for server-side operations.
        """
        # Determine model type *before* calling super init
        self.model_type = self._determine_model_type(model)
        logger.info(f"QuantumServer: Initializing with model type '{self.model_type}'.")

        # Pass args to parent init
        super().__init__(
            model=model, # Parent moves model to device
            aggregation_strategy=aggregation_strategy,
            evaluation_dataset=evaluation_dataset,
            device=device # Pass device to parent
        )
        # self.model is now initialized and on self.device

    def _determine_model_type(self, model: Any) -> str:
        """Determine the type of quantum model."""
        if isinstance(model, VariationalQuantumClassifier): return "vqc"
        elif isinstance(model, QuantumNeuralNetwork): return "qnn"
        elif isinstance(model, HybridQuantumModel): return "hybrid"
        elif isinstance(model, nn.Module): return "torch"
        else: raise ValueError(f"Unsupported model type for QuantumFederatedServer: {type(model)}")

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get global parameters, converting VQC NumPy params to Torch tensor."""
        if self.model_type == "vqc":
            logger.debug("Server: Getting VQC parameters and converting to Tensor.")
            if self.model.params is None: raise ValueError("Global VQC model parameters not initialized.")
            # Ensure consistent dtype and place on CPU for distribution
            params_tensor = torch.tensor(self.model.params, dtype=torch.float64).cpu()
            return {"quantum_params": params_tensor}
        else:
            # Use parent method for PyTorch models (gets state_dict on CPU)
            logger.debug(f"Server: Getting {self.model_type} parameters (state_dict).")
            return super().get_parameters()

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        """Set global parameters, converting Torch tensor back to NumPy for VQC."""
        if self.model_type == "vqc":
            logger.debug("Server: Setting VQC parameters from Tensor.")
            if "quantum_params" in parameters:
                try:
                    # Convert incoming tensor (CPU) to NumPy array
                    vqc_params_np = parameters["quantum_params"].numpy()
                    if self.model.params is not None and self.model.params.shape != vqc_params_np.shape:
                         logger.error(f"Server VQC parameter shape mismatch. Model: {self.model.params.shape}, Received: {vqc_params_np.shape}")
                         raise ValueError("VQC parameter shape mismatch during set_parameters.")
                    self.model.params = vqc_params_np
                    logger.debug("Server: Global VQC parameters updated.")
                except Exception as e:
                     logger.error(f"Server: Error converting/setting VQC parameters: {e}", exc_info=True); raise
            else:
                logger.error("Server: 'quantum_params' key not found in parameters for VQC.")
                raise KeyError("'quantum_params' key missing for VQC model.")
        else:
            # Use parent method for PyTorch models (loads state_dict to server's device)
            logger.debug(f"Server: Setting {self.model_type} parameters (state_dict).")
            super().set_parameters(parameters)

    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the global model, handling VQC separately."""
        logger.info(f"Server: Starting evaluation (model type: {self.model_type}).")
        if self.model_type == "vqc":
            # Use specialized VQC evaluation
            return self._evaluate_vqc(dataset)
        else:
            # Use parent's evaluate method for PyTorch models
            return super().evaluate(dataset)

    def _evaluate_vqc(self, eval_dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the global VQC model."""
        dataset_to_eval = eval_dataset if eval_dataset is not None else self.evaluation_dataset
        if dataset_to_eval is None or len(dataset_to_eval) == 0:
             logger.warning("Server VQC evaluation skipped: No data.")
             return {'loss': None, 'accuracy': 0.0}

        logger.debug(f"Server VQC: Evaluating on {len(dataset_to_eval)} samples.")
        correct = 0; total = 0; all_labels = []; all_preds = []

        # VQC predict is sample-by-sample
        for i in range(len(dataset_to_eval)):
            try:
                data, target = dataset_to_eval[i]
                features_np = data.numpy()
                if features_np.ndim > 1: features_np = features_np.flatten()
                label = target.item() if isinstance(target, torch.Tensor) else int(target)

                # Use the server's global VQC model instance for prediction
                prediction = self.model.predict(features_np)
                if prediction == label: correct += 1
                total += 1
                all_labels.append(label)
                all_preds.append(prediction)
            except Exception as e:
                 logger.error(f"Server VQC: Error evaluating sample {i}: {e}", exc_info=False)

        accuracy = correct / total if total > 0 else 0.0
        # Optional: Add more metrics if needed (e.g., confusion matrix from all_labels/all_preds)
        metrics = {'accuracy': accuracy, 'loss': None} # Loss hard to calculate efficiently here
        logger.info(f"Server VQC: Evaluation finished. Metrics: {metrics}")
        return metrics


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
        """Aggregate model updates, detecting VQC structure."""
        if not client_updates:
            logger.warning("QuantumAggregationStrategy: No client updates provided.")
            return {}

        # Check the structure of the first update to determine model type
        first_params = client_updates[0][0]
        is_vqc_update = list(first_params.keys()) == ["quantum_params"] and isinstance(first_params["quantum_params"], torch.Tensor)

        if is_vqc_update:
            logger.info("QuantumAggregationStrategy: Detected VQC parameter structure. Aggregating 'quantum_params'.")
            return self._aggregate_vqc_params(client_updates)
        else:
            logger.info(f"QuantumAggregationStrategy: Detected non-VQC structure. Delegating to base strategy: {type(self.base_strategy).__name__}")
            # Use base strategy for standard state_dict updates
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
    """
    Static utility class for setting up quantum federated learning scenarios.
    """

    @staticmethod
    def create_quantum_clients(
        client_ids: List[str],
        model_class: Type[Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module]],
        model_kwargs: Dict[str, Any],
        datasets: List[Dataset],
        client_kwargs: Optional[Dict[str, Any]] = None # Args for QuantumFederatedClient init
    ) -> List[QuantumFederatedClient]:
        """Create quantum federated clients."""
        if len(client_ids) != len(datasets):
            raise ValueError("Number of client IDs must match number of datasets")
        if not issubclass(model_class, (VariationalQuantumClassifier, nn.Module)):
             raise TypeError("model_class must be VQC or a PyTorch nn.Module.")

        clients = []
        client_setup_kwargs = client_kwargs or {}
        logger.info(f"Creating {len(client_ids)} quantum clients with model {model_class.__name__}.")

        for i, (client_id, dataset) in enumerate(zip(client_ids, datasets)):
            # Create a distinct model instance for each client
            try:
                model_instance = model_class(**model_kwargs)
            except Exception as e:
                 logger.error(f"Failed to instantiate model {model_class.__name__} for client {client_id}: {e}", exc_info=True)
                 raise

            # Initialize QuantumFederatedClient
            try:
                client = QuantumFederatedClient(
                    client_id=client_id,
                    model=model_instance,
                    dataset=dataset,
                    **client_setup_kwargs # Pass batch_size, lr, optimizer, loss, device here
                )
                clients.append(client)
                logger.debug(f"Successfully created quantum client '{client_id}' with {len(dataset)} samples.")
            except Exception as e:
                 logger.error(f"Failed to create QuantumFederatedClient '{client_id}': {e}", exc_info=True)
                 # Decide whether to skip or raise
                 raise

        return clients

    @staticmethod
    def initialize_quantum_server(
        model_class: Type[Union[VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel, nn.Module]],
        model_kwargs: Dict[str, Any],
        aggregation_strategy: AggregationStrategy, # Can be QuantumAggregationStrategy or base
        evaluation_dataset: Optional[Dataset] = None,
        server_kwargs: Optional[Dict[str, Any]] = None # Args for QuantumFederatedServer init
    ) -> QuantumFederatedServer:
        """Initialize a quantum federated server."""
        if not issubclass(model_class, (VariationalQuantumClassifier, nn.Module)):
             raise TypeError("model_class must be VQC or a PyTorch nn.Module.")

        logger.info(f"Initializing quantum server with global model {model_class.__name__}.")
        # Initialize the global model instance
        try:
            global_model = model_class(**model_kwargs)
        except Exception as e:
            logger.error(f"Failed to instantiate global model {model_class.__name__}: {e}", exc_info=True)
            raise

        # Initialize the QuantumFederatedServer
        server_setup_kwargs = server_kwargs or {}
        try:
            server = QuantumFederatedServer(
                model=global_model,
                aggregation_strategy=aggregation_strategy,
                evaluation_dataset=evaluation_dataset,
                **server_setup_kwargs # Pass device here if needed
            )
            logger.info(f"Quantum federated server initialized successfully.")
            return server
        except Exception as e:
             logger.error(f"Failed to initialize QuantumFederatedServer: {e}", exc_info=True)
             raise

    # Removed adapt_evaluation_for_quantum_models as evaluation is handled internally now