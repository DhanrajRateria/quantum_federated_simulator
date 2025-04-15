# src/core/spark_server.py

import logging
import time
from typing import Dict, List, Optional, Tuple, Callable, Any, Union, Type
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Subset # Added Subset

# Use absolute imports
from src.federated.aggregation import AggregationStrategy, FedAvg
from src.quantum.models import VariationalQuantumClassifier, QuantumNeuralNetwork, HybridQuantumModel # For type hints
from src.models.classical_models import SimpleMLP # For type hints
from src.distributed.spark_manager import SparkManager
# Import the executor task function
from src.distributed.spark_executor_task import run_client_training_partition
# Import specialized aggregation strategy if needed for VQC
from src.core.quantum_manager import QuantumAggregationStrategy

logger = logging.getLogger(__name__)

class SparkFederatedServer:
    """
    Server orchestrating Federated Learning using Apache Spark.
    """
    def __init__(
        self,
        model: Union[VariationalQuantumClassifier, nn.Module], # Global model instance
        spark_manager: SparkManager,
        aggregation_strategy: AggregationStrategy,
        evaluation_dataset: Optional[Dataset] = None,
        device: Optional[torch.device] = None # Device for driver-side model/eval
    ):
        self.model = model
        self.spark_manager = spark_manager
        self.spark_context = spark_manager.get_context()
        self.aggregation_strategy = aggregation_strategy
        self.evaluation_dataset = evaluation_dataset
        self.round_history: List[Dict[str, Any]] = []
        self.model_type = self._determine_model_type(model)

        if device is None: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        if isinstance(self.model, nn.Module): self.model.to(self.device)

        logger.info(f"SparkFederatedServer initialized. Model: {self.model_type}, Agg: {type(aggregation_strategy).__name__}, Device: {self.device}")

    def _determine_model_type(self, model: Any) -> str:
        if isinstance(model, VariationalQuantumClassifier): return "vqc"
        elif isinstance(model, (QuantumNeuralNetwork, HybridQuantumModel, nn.Module)): return "torch"
        else: raise ValueError(f"Unsupported model type: {type(model)}")

    # --- Parameter handling on the DRIVER ---
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        logger.debug("Server: Getting global parameters for broadcast.")
        if self.model_type == "vqc":
            if self.model.params is None: raise ValueError("VQC params not initialized.")
            return {"quantum_params": torch.tensor(self.model.params, dtype=torch.float64).cpu()}
        elif isinstance(self.model, nn.Module):
            return {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        else: raise TypeError("Unknown model type for get_parameters.")

    def set_parameters(self, parameters: Dict[str, torch.Tensor]) -> None:
        logger.debug("Server: Setting global parameters from aggregation.")
        if self.model_type == "vqc":
            if "quantum_params" in parameters: self.model.params = parameters["quantum_params"].numpy()
            else: raise KeyError("'quantum_params' key missing for VQC.")
        elif isinstance(self.model, nn.Module):
            self.model.load_state_dict(parameters); self.model.to(self.device) # Ensure model stays on device
        else: raise TypeError("Unknown model type for set_parameters.")

    # --- Evaluation on DRIVER ---
    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        eval_dataset = dataset if dataset is not None else self.evaluation_dataset
        if eval_dataset is None or len(eval_dataset) == 0: logger.warning("Server eval skipped: No data."); return {}
        logger.info(f"Server: Starting evaluation (model type: {self.model_type}) on driver.")

        if self.model_type == "vqc": # VQC Evaluation (CPU)
             logger.debug(f"Server VQC: Evaluating on {len(eval_dataset)} samples.")
             correct = 0; total = 0;
             for i in range(len(eval_dataset)):
                 try: data, target = eval_dataset[i]; features_np = data.numpy(); label = target.item();
                 except AttributeError: data, target = eval_dataset[i]; features_np = np.array(data); label = int(target)
                 if features_np.ndim > 1: features_np = features_np.flatten()
                 try: prediction = self.model.predict(features_np)
                 except Exception as pred_e: logger.error(f"VQC predict failed sample {i}: {pred_e}"); continue
                 if prediction == label: correct += 1; total += 1
             accuracy = correct / total if total > 0 else 0.0; metrics = {'accuracy': accuracy, 'loss': None}
             logger.info(f"Server VQC: Evaluation finished. Metrics: {metrics}")
             return metrics

        elif isinstance(self.model, nn.Module): # PyTorch Evaluation (on self.device)
             eval_loader = DataLoader(eval_dataset, batch_size=128, shuffle=False)
             logger.debug(f"Starting server-side evaluation (PyTorch Model) on {len(eval_dataset)} using device {self.device}.")
             self.model.eval(); self.model.to(self.device)
             loss_fn = nn.CrossEntropyLoss(); total_loss = 0.0; correct = 0; total_samples = 0
             with torch.no_grad():
                 for data, target in eval_loader:
                     data, target = data.to(self.device), target.to(self.device); current_batch_size = data.shape[0]
                     output = self.model(data); loss = loss_fn(output, target); total_loss += loss.item() * current_batch_size
                     _, predicted = torch.max(output.data, 1); correct += (predicted == target).sum().item(); total_samples += current_batch_size
             avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
             accuracy = correct / total_samples if total_samples > 0 else 0.0
             metrics = {'loss': avg_loss, 'accuracy': accuracy}
             logger.info(f"Server evaluation (PyTorch Model) completed. Metrics: {metrics}")
             return metrics
        else: logger.warning("Cannot evaluate unknown model type."); return {}


    # --- Main Training Orchestration ---
    def train(
        self,
        num_rounds: int,
        train_dataset: Dataset, # Full training dataset on driver
        num_data_partitions: int, # How many partitions to create (== num clients)
        model_class_for_client: Type[Union[VariationalQuantumClassifier, nn.Module]],
        model_kwargs_for_client: Dict[str, Any],
        client_config: Dict[str, Any], # Client hyperparams (LR, batch size...)
        local_epochs: int = 1,
        proximal_term: float = 0.0, # FedProx mu
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """ Executes multiple rounds of Spark-based FL. """
        logger.info(f"Starting Spark FL process: {num_rounds} rounds, {num_data_partitions} partitions/clients.")
        if self.evaluation_dataset: logger.info("Performing initial evaluation..."); initial_metrics = self.evaluate(); logger.info(f"Initial Metrics: {initial_metrics}")

        # --- Prepare Data RDD ---
        try:
             logger.info(f"Preparing data RDD for {num_data_partitions} partitions...")
             # Option 1: Parallelize indices and load data within task (more complex)
             # Option 2: Parallelize data directly (simpler, assumes driver has memory)
             data_list = [(train_dataset[i][0], train_dataset[i][1]) for i in range(len(train_dataset))]
             # Create RDD with desired partitioning
             data_rdd = self.spark_context.parallelize(data_list, numSlices=num_data_partitions)
             # Optional: Repartition if needed, might cause shuffle
             # if data_rdd.getNumPartitions() != num_data_partitions:
             #     data_rdd = data_rdd.repartition(num_data_partitions)
             data_rdd = data_rdd.cache() # Cache for multiple rounds
             num_partitions = data_rdd.getNumPartitions()
             rdd_count = data_rdd.count() # Action to trigger caching and get count
             logger.info(f"Training data parallelized into {num_partitions} RDD partitions ({rdd_count} total samples) and cached.")
             if num_partitions != num_data_partitions:
                  logger.warning(f"Requested {num_data_partitions} partitions, but Spark created {num_partitions}.")
        except Exception as e:
             logger.error(f"Failed to parallelize training data: {e}", exc_info=True); return self.round_history

        # --- Run Rounds ---
        for round_idx in range(num_rounds):
            round_num = round_idx + 1
            round_start_time = time.time()  # Define round_start_time at the start of each round
            logger.info(f"===== Starting Spark Round {round_num}/{num_rounds} =====")

            # 1. Broadcast Global Parameters
            global_params_cpu = self.get_parameters()
            global_params_bc = self.spark_context.broadcast(global_params_cpu)
            logger.info(f"Round {round_num}: Broadcasted global parameters (ID: {global_params_bc.id})")

            # 2. Execute Training Task on RDD
            try:
                results_rdd = data_rdd.mapPartitionsWithIndex(
                    # Use lambda to pass args correctly to the executor function
                    lambda pid, iter: run_client_training_partition(
                        partition_iterator=iter, partition_id=pid,
                        model_info={'class_name': model_class_for_client.__name__, 'kwargs': model_kwargs_for_client},
                        client_config=client_config, global_params_bc=global_params_bc,
                        local_epochs=local_epochs, proximal_term=proximal_term, seed=seed + round_idx
                    )
                )
                # 3. Collect Results
                logger.info(f"Round {round_num}: Collecting results from executors...")
                start_collect = time.time()
                collected_results = results_rdd.collect() # List[Tuple[Dict, int]]
                logger.info(f"Round {round_num}: Collection finished in {time.time() - start_collect:.2f}s. "
                            f"Received results from {len(collected_results)} clients.")

            except Exception as e:
                 logger.error(f"Spark job failed during round {round_num}: {e}", exc_info=True)
                 global_params_bc.unpersist(); # Cleanup broadcast variable
                 # Log round failure but continue to next round? Or break?
                 round_info = {"round": round_num, "status": "failed", "error": f"Spark execution failed: {e}"}
                 self.round_history.append(round_info)
                 continue # Continue to next round

            global_params_bc.unpersist() # Cleanup broadcast variable

            # 4. Aggregate & Update
            client_updates = collected_results
            if not client_updates:
                logger.warning(f"Round {round_num}: No valid client updates received.")
                status="failed"; error_msg="No valid updates"; metrics={}
            else:
                agg_start = time.time()
                logger.info(f"Round {round_num}: Aggregating {len(client_updates)} updates using {type(self.aggregation_strategy).__name__}.")
                try:
                    aggregated_params_cpu = self.aggregation_strategy.aggregate(client_updates)
                    self.set_parameters(aggregated_params_cpu)
                    logger.info(f"Round {round_num}: Aggregation and model update complete in {time.time() - agg_start:.2f}s.")
                    status="success"; error_msg=None
                    # 5. Evaluate
                    metrics = {};
                    if self.evaluation_dataset is not None:
                        logger.info(f"Round {round_num}: Evaluating global model..."); eval_start = time.time()
                        try: metrics = self.evaluate()
                        except Exception as e: logger.error(f"Evaluation error: {e}", exc_info=True); metrics = {"eval_error": str(e)}
                        logger.info(f"Round {round_num}: Evaluation complete in {time.time() - eval_start:.2f}s.")
                except Exception as e:
                    logger.error(f"Aggregation/Update error: {e}", exc_info=True); status="failed"; error_msg="Aggregation/Update failed"; metrics={}

            # 6. Record History
            round_duration = time.time() - round_start_time # Redefine round_start_time before loop
            round_info = {
                "round": round_num, "status": status, "error": error_msg,
                "successful_clients": len(client_updates),
                "client_samples": [s for _, s in client_updates], "total_samples": sum(s for _, s in client_updates),
                "duration_seconds": round_duration, "evaluation_metrics": metrics
            }
            self.round_history.append(round_info)
            logger.info(f"Round {round_num} ({status}) completed in {round_duration:.2f}s. Eval Metrics: {metrics}")
            # Optional: Add early stopping logic here based on metrics

        # --- Cleanup ---
        data_rdd.unpersist()
        logger.info(f"Spark FL process completed after {num_rounds} rounds.")
        return self.round_history

    def save_model(self, path: str) -> None:
        # (Same logic as before - only handles nn.Module or VQC)
         if isinstance(self.model, nn.Module): logger.info(f"Saving global nn.Module state_dict to {path}"); torch.save(self.model.state_dict(), path)
         elif isinstance(self.model, VariationalQuantumClassifier): self.model.save_params(path.replace('.pt', '.npy')); logger.info(f"Saved VQC params to {path.replace('.pt', '.npy')}")
         else: logger.warning(f"Server save_model skipped: Model type {type(self.model)} not handled.")

    def load_model(self, path: str) -> None:
         # (Same logic as before)
         if isinstance(self.model, nn.Module): logger.info(f"Loading global nn.Module state_dict from {path}"); state_dict = torch.load(path, map_location=self.device); self.model.load_state_dict(state_dict); self.model.to(self.device)
         elif isinstance(self.model, VariationalQuantumClassifier): self.model.load_params(path.replace('.pt', '.npy')); logger.info(f"Loaded VQC params from {path.replace('.pt', '.npy')}")
         else: logger.warning(f"Server load_model skipped: Model type {type(self.model)} not handled.")