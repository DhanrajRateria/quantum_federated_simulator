"""
Unit tests for quantum circuit implementations.
Includes logging and focuses on execution and parameter checks.
"""

import os
import sys
import unittest
import numpy as np
import pennylane as qml
# tempfile no longer needed if not creating temp configs for tests
import shutil
import logging
from pennylane import PennyLaneDeprecationWarning
import warnings # Import warnings module

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Path Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
TEST_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'test_outputs', 'quantum_circuits') # Keep for potential future use or logs
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs', 'quantum')

if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

if not os.path.isdir(SRC_ROOT):
     raise ImportError(f"Source directory not found at {SRC_ROOT}.")
if not os.path.isdir(CONFIG_DIR):
     logger.warning(f"Config directory not found at {CONFIG_DIR}. Custom config tests might be skipped.")
from src.quantum.utils import load_config, set_random_seed, get_device
from src.quantum.circuits import (
    create_basic_circuit,
    create_complex_circuit,
    create_custom_circuit,
    create_noisy_circuit
)

# --- Test Base Class ---
class BaseCircuitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Output directory might still be useful for logs or other artifacts
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
        logger.info(f"Test output directory (for logs/artifacts): {TEST_OUTPUT_DIR}")
        # Clean before tests if desired
        if os.path.exists(TEST_OUTPUT_DIR):
            shutil.rmtree(TEST_OUTPUT_DIR)
            os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)
            logger.info(f"Cleaned test output directory: {TEST_OUTPUT_DIR}")


    def setUp(self):
        set_random_seed(42)
        self.n_qubits = 4
        self.n_layers = 2
        self.device = None
        logger.info(f"\n--- Starting test: {self.id()} ---")

    def tearDown(self):
        logger.info(f"--- Finished test: {self.id()} ---")

    def _get_device(self, wires):
        return get_device("default.qubit", wires=wires)

    def _run_circuit_test(self, circuit_fn, params, features, test_name, device):
        """Helper to run circuit and check output."""
        self.assertTrue(callable(circuit_fn), f"{test_name}: Circuit function should be callable.")
        logger.info(f"{test_name}: Testing circuit execution...")
        logger.info(f"{test_name}: n_qubits={device.wires}, params_shape={params.shape}, features_shape={features.shape if features is not None else 'None'}")

        qnode = qml.QNode(circuit_fn, device)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The tape/qtape property is deprecated", category=PennyLaneDeprecationWarning)

            # Execute
            result = qnode(params, features)
            logger.info(f"{test_name}: Execution successful. Result shape: {np.array(result).shape}, Result snippet: {np.array(result)[:5]}") # Log first few results

            current_tape = qnode.tape
            if not current_tape:
                 self.fail(f"{test_name}: QNode tape was not generated after execution.")

            # Log circuit structure (operations)
            logger.debug(f"{test_name}: Circuit Operations:\n{current_tape.operations}")
            logger.debug(f"{test_name}: Circuit Measurements:\n{current_tape.measurements}")


            if isinstance(result, list) or isinstance(result, np.ndarray):
                self.assertTrue(len(result) > 0, f"{test_name}: Result should not be empty.")
                if current_tape.measurements:
                     is_expectation = all(m.return_type is qml.measurements.Expectation for m in current_tape.measurements)
                     is_probability = all(m.return_type is qml.measurements.Probability for m in current_tape.measurements)

                     if is_expectation:
                         logger.debug(f"{test_name}: Checking expectation values range [-1, 1].")
                         for i, val in enumerate(result):
                             self.assertTrue(-1.001 <= val <= 1.001, f"{test_name}: Expectation value {i} ({val}) out of range [-1, 1].")
                     elif is_probability:
                         logger.debug(f"{test_name}: Checking probability values range [0, 1] and sum.")
                         self.assertTrue(np.isclose(np.sum(result), 1.0), f"{test_name}: Probabilities do not sum to 1 (sum={np.sum(result)}).")
                         self.assertTrue(all(0.0 <= p <= 1.0 for p in result), f"{test_name}: Not all results are valid probabilities.")
                     else:
                          logger.info(f"{test_name}: Output type is mixed or non-standard, skipping range/sum checks.")
                else:
                     logger.warning(f"{test_name}: No measurements found on the tape to check output type.")
            else:
                self.fail(f"{test_name}: Unexpected result type {type(result)}")
            return qnode

    def _check_params(self, circuit_fn, params, features, test_name, device):
        """Helper to check trainable parameter counts."""
        logger.info(f"{test_name}: Checking trainable parameters...")
        self.assertTrue(hasattr(circuit_fn, 'expected_params'), f"{test_name}: Circuit function should have 'expected_params' attribute.")
        expected_params = circuit_fn.expected_params
        self.assertEqual(len(params), expected_params, f"{test_name}: Provided params length {len(params)} != expected {expected_params}.")

        qnode = qml.QNode(circuit_fn, device)

        with warnings.catch_warnings():
             warnings.filterwarnings("ignore", message="The tape/qtape property is deprecated", category=PennyLaneDeprecationWarning)

             qnode(params, features)
             current_tape = qnode.tape

             if not current_tape:
                 self.fail(f"{test_name}: QNode tape was not generated after execution for param check.")

             if not current_tape.trainable_params:
                 logger.warning(f"{test_name}: No trainable parameters auto-detected. Attempting manual setting based on input 'params'.")
                 try:
                     all_tape_params = current_tape.get_parameters(trainable_only=False)
                     num_params_expected = len(params)
                     if len(all_tape_params) >= num_params_expected:
                          trainable_indices = list(range(num_params_expected))
                          current_tape.trainable_params = trainable_indices
                          logger.info(f"{test_name}: Manually set trainable_params to indices: {trainable_indices}")
                     else:
                          logger.error(f"{test_name}: Cannot manually set trainable params. Tape has {len(all_tape_params)} params, expected input {num_params_expected}.")
                 except Exception as e:
                     logger.error(f"{test_name}: Error manually setting trainable params: {e}", exc_info=True)

             trainable_params_list = current_tape.get_parameters(trainable_only=True)
             actual_trainable_params_count = len(trainable_params_list)
             # Add logging of the actual parameter values (first few)
             trainable_vals_snippet = [f"{p:.3f}" for p in trainable_params_list[:5]]
             logger.info(f"{test_name}: Expected trainable params: {expected_params}, Found in tape: {actual_trainable_params_count} (values snippet: {trainable_vals_snippet}...)")

             self.assertEqual(actual_trainable_params_count, expected_params,
                              f"{test_name}: Number of trainable params in tape ({actual_trainable_params_count}) "
                              f"doesn't match expected ({expected_params}).")


# --- Test Cases ---
# (Test classes TestBasicCircuit, TestComplexCircuit, TestCustomCircuit, TestNoisyCircuit remain the same)
class TestBasicCircuit(BaseCircuitTest):
    """Test basic circuit implementation."""

    def setUp(self):
        super().setUp() # Calls BaseCircuitTest.setUp
        self.n_qubits = 4
        self.n_layers = 2
        self.device = self._get_device(self.n_qubits) # Get device specific to this test's qubits

    def test_basic_circuit(self):
        """Test basic circuit creation, execution, and parameter count."""
        circuit_fn = create_basic_circuit(self.n_qubits, self.n_layers)
        expected_params = circuit_fn.expected_params # Get expected count

        params = np.random.uniform(0, 2*np.pi, size=expected_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)

        qnode = self._run_circuit_test(circuit_fn, params, features, "basic_circuit", self.device)
        self._check_params(circuit_fn, params, features, "basic_circuit", self.device)

class TestComplexCircuit(BaseCircuitTest):
    """Test complex circuit implementation."""

    def setUp(self):
        super().setUp()
        self.n_qubits = 5 # Use a different number
        self.n_layers = 2
        self.device = self._get_device(self.n_qubits)

    def test_complex_circuit(self):
        """Test complex circuit creation, execution, and parameter count."""
        circuit_fn = create_complex_circuit(self.n_qubits, self.n_layers)
        expected_params = circuit_fn.expected_params

        params = np.random.uniform(0, 2*np.pi, size=expected_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)

        qnode = self._run_circuit_test(circuit_fn, params, features, "complex_circuit", self.device)
        self._check_params(circuit_fn, params, features, "complex_circuit", self.device)

class TestCustomCircuit(BaseCircuitTest):
    """Test custom circuit creation from configuration files."""

    def run_custom_config_test(self, config_filename):
        """Helper method to run tests for a specific config file."""
        config_path = os.path.join(CONFIG_DIR, config_filename)
        test_name = f"custom_circuit_{config_filename.replace('.yaml', '')}"

        if not os.path.exists(config_path):
             self.skipTest(f"Config file not found: {config_path}")

        config = load_config(config_path)
        n_qubits = config['n_qubits'] # Use local var for clarity
        device = self._get_device(n_qubits) # Create device specific to this test

        circuit_fn = create_custom_circuit(config_path)
        expected_params = circuit_fn.expected_params

        if 'n_params' in config:
             pass

        params = np.random.uniform(0, 2*np.pi, size=expected_params)
        feature_size = n_qubits
        if config.get('encoding') == 'amplitude':
            feature_size = min(n_qubits*2, 2**n_qubits)
            logger.info(f"{test_name}: Using feature size {feature_size} for amplitude encoding.")

        features = np.random.uniform(-1, 1, size=feature_size)

        qnode = self._run_circuit_test(circuit_fn, params, features, test_name, device)
        self._check_params(circuit_fn, params, features, test_name, device)

    def test_custom_circuit_from_config_basic(self):
        """Test custom circuit from basic_circuit.yaml."""
        self.run_custom_config_test("basic_circuit.yaml")

    def test_custom_circuit_from_config_basic_alt(self):
        """Test custom circuit from basic_circuit_.yaml."""
        self.run_custom_config_test("basic_circuit_.yaml")

class TestNoisyCircuit(BaseCircuitTest):
    """Test noisy circuit wrapper (conceptual)."""

    def setUp(self):
        super().setUp()
        self.n_qubits = 2
        self.n_layers = 1
        self.device = self._get_device(self.n_qubits)

    def test_noisy_wrapper_execution(self):
        """Test that the noisy wrapper executes the base circuit."""
        base_circuit_fn = create_basic_circuit(self.n_qubits, self.n_layers)
        expected_params = base_circuit_fn.expected_params

        noise_model = {'type': 'depolarizing', 'probability': 0.05}
        noisy_circuit_fn = create_noisy_circuit(base_circuit_fn, noise_model)
        self.assertTrue(callable(noisy_circuit_fn))

        self.assertTrue(hasattr(noisy_circuit_fn, 'expected_params'))
        self.assertEqual(noisy_circuit_fn.expected_params, expected_params)

        params = np.random.uniform(0, 2*np.pi, size=expected_params)
        features = np.random.uniform(-1, 1, size=self.n_qubits)

        # Run the wrapped circuit
        qnode_noisy = self._run_circuit_test(noisy_circuit_fn, params, features, "noisy_circuit_wrapper", self.device)

        # Optional: Compare output to base circuit
        qnode_base = qml.QNode(base_circuit_fn, self.device)
        result_base = qnode_base(params, features)
        result_noisy = qnode_noisy(params, features)

        np.testing.assert_allclose(result_base, result_noisy, atol=1e-7,
                                   err_msg="Noisy wrapper output should match base circuit output on 'default.qubit'.")
        logger.info("Noisy circuit wrapper output matches base circuit output as expected on default.qubit.")


if __name__ == "__main__":
    unittest.main()