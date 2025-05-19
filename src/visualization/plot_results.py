import os
import sys
import logging
import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results_bc')
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations_bc')
ADVANCED_VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'advanced_vis_bc')

# Configure logging
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs_bc')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, f"visualization_{time.strftime('%Y%m%d-%H%M%S')}.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VisualizationRunner")

# Make sure the src directory is in the path
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Import our advanced visualization module
from visualization.advanced_plots import AdvancedVisualization

def load_results(results_dir: str, results_file: Optional[str] = None) -> List[Dict]:
    """
    Load experiment results from a file or the most recent file in the directory.
    
    Args:
        results_dir: Directory containing result files
        results_file: Optional specific file to load
        
    Returns:
        List of experiment result dictionaries
    """
    if not os.path.exists(results_dir):
        logger.error(f"Results directory does not exist: {results_dir}")
        return []
    
    # If no specific file is provided, find the most recent one
    if results_file is None:
        json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
        
        if json_files:
            json_files.sort(reverse=True)  # Most recent should be first
            results_file = json_files[0]
            file_format = 'json'
        elif csv_files:
            csv_files.sort(reverse=True)  # Most recent should be first
            results_file = csv_files[0]
            file_format = 'csv'
        else:
            logger.error(f"No result files found in directory: {results_dir}")
            return []
    else:
        file_format = results_file.split('.')[-1].lower()
    
    file_path = os.path.join(results_dir, results_file)
    logger.info(f"Loading results from: {file_path}")
    
    try:
        if file_format == 'json':
            with open(file_path, 'r') as f:
                results = json.load(f)
        elif file_format == 'csv':
            # For CSV, we need to reconstruct the data structure
            df = pd.read_csv(file_path)
            
            # Convert DataFrame to list of dictionaries
            results = []
            for _, row in df.iterrows():
                # Extract the experiment name and config
                exp_name = row.get('experiment_name')
                if not exp_name:
                    continue
                
                # Try to parse the round history if it exists
                round_history = []
                for col in df.columns:
                    if col.startswith('round_history'):
                        try:
                            # This is complex because the CSV might have flattened the JSON
                            # Let's try a simplified approach
                            round_history = json.loads(row.get(col, '[]'))
                            break
                        except:
                            pass
                
                # Create a result dictionary
                result = {
                    'experiment_name': exp_name,
                    'config': {},
                    'round_history': round_history,
                    'final_accuracy': row.get('final_accuracy'),
                    'final_loss': row.get('final_loss')
                }
                
                # Try to extract config information
                for col in df.columns:
                    if col.startswith('config_'):
                        # Remove the 'config_' prefix and set the value
                        config_key = col[7:]
                        try:
                            # Try to parse as JSON if it looks like a dictionary or list
                            value = row.get(col)
                            if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                value = json.loads(value)
                            result['config'][config_key] = value
                        except:
                            # If parsing fails, use the raw value
                            result['config'][config_key] = row.get(col)
                
                results.append(result)
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return []
        
        logger.info(f"Successfully loaded {len(results)} experiment results")
        return results
    except Exception as e:
        logger.error(f"Error loading results from {file_path}: {e}", exc_info=True)
        return []

def create_visualizations(results: List[Dict]) -> None:
    """
    Create all visualizations for the given experiment results.
    
    Args:
        results: List of experiment result dictionaries
    """
    if not results:
        logger.error("No results to visualize")
        return
    
    # Initialize our advanced visualization module
    vis = AdvancedVisualization(output_dir=ADVANCED_VIS_DIR, dpi=300)
    
    # Generate all visualizations
    try:
        vis.generate_all_visualizations(results)
        logger.info(f"All visualizations generated successfully in {ADVANCED_VIS_DIR}")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}", exc_info=True)

def main():
    """Main entry point for visualization generation."""
    logger.info("Starting visualization generation")
    
    # Load results from the latest file
    results = load_results(RESULTS_DIR)
    
    # Create visualizations
    create_visualizations(results)
    
    logger.info("Visualization process completed")

if __name__ == "__main__":
    main()