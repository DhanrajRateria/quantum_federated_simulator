import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse
import json
from typing import Dict, List, Any, Optional, Union

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULTS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'results_bc')
VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'visualizations_bc')
LOG_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'logs_bc')

if SRC_ROOT not in sys.path: 
    sys.path.insert(0, SRC_ROOT)

os.makedirs(VIS_DIR, exist_ok=True)

# Configure logging
log_file_path = os.path.join(LOG_DIR, f"bc_visualization_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BC_Visualization")

# Try to import the enhanced visualization function
try:
    from src.visualization.integrate_visualizations import enhanced_plot_bc_results
    has_enhanced_viz = True
    logger.info("Enhanced visualization module found")
except ImportError:
    has_enhanced_viz = False
    logger.warning("Enhanced visualization module not found, will use basic visualization")

def load_results_from_csv(csv_path: str) -> pd.DataFrame:
    """Load results from a CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded results from {csv_path} with {len(df)} entries")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {e}", exc_info=True)
        raise

def convert_df_to_results_list(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to a list of experiment result dictionaries.
    This mimics the structure expected by the original plot_bc_results function."""
    
    results_list = []
    for idx, row in df.iterrows():
        # Extract round history if it exists as a JSON string
        round_history = []
        if 'round_history' in df.columns:
            try:
                # Try to parse if it's a JSON string
                if isinstance(row['round_history'], str):
                    round_history = json.loads(row['round_history'])
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Could not parse round_history for experiment {row['experiment_name']}")
        
        # Create a result entry with available fields
        result = {
            "experiment_name": row['experiment_name'],
            "round_history": round_history,
        }
        
        # Add other fields if they exist
        for col in df.columns:
            if col not in ['experiment_name', 'round_history']:
                result[col] = row[col]
        
        results_list.append(result)
    
    return results_list

def plot_bc_results(results_list: List[Dict]):
    """Basic visualization function that works with the converted results list."""
    logger.info("Generating basic comparison plots for Breast Cancer experiments...")
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create visualization directory structure
    for subdir in ["accuracy_curves", "loss_curves", "resource_usage", "comparisons"]:
        os.makedirs(os.path.join(VIS_DIR, subdir), exist_ok=True)
    
    # Process data for plotting
    plot_data = []
    final_metrics_data = []
    
    for result in results_list:
        exp_name = result['experiment_name']
        if 'final_accuracy' in result:
            final_metrics_data.append({
                'Experiment': exp_name,
                'Final Accuracy': result['final_accuracy'],
                'Final Loss': result.get('final_loss', 0),
                # Try to extract model type, agg strategy and data partition type
                'Model Type': result.get('config_model_type', 'Unknown'),
                'Aggregation': result.get('config_aggregation_overrides_strategy', 'Unknown'),
                'Data': 'IID' if result.get('config_data_partition_iid', True) else 'Non-IID'
            })
    
    if not final_metrics_data:
        logger.warning("No valid data for plotting. Check your CSV structure.")
        return
    
    final_metrics_df = pd.DataFrame(final_metrics_data)
    
    # Create basic visualizations
    # 1. Final accuracy comparison
    fig, ax = plt.subplots(figsize=(10, max(6, len(final_metrics_df)*0.5)))
    sns.barplot(data=final_metrics_df, x='Final Accuracy', y='Experiment', ax=ax, orient='h')
    ax.set_title('BC: Final Accuracy Comparison')
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    output_path = os.path.join(VIS_DIR, "comparisons", "bc_final_accuracy_comparison.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"BC Plot saved to {output_path}")
    
    # 2. Grouped by model type
    if 'Model Type' in final_metrics_df.columns and not final_metrics_df['Model Type'].eq('Unknown').all():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=final_metrics_df, x='Model Type', y='Final Accuracy', ax=ax)
        ax.set_title('BC: Accuracy by Model Type')
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        output_path = os.path.join(VIS_DIR, "comparisons", "bc_model_type_comparison.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"BC Plot saved to {output_path}")
    
    # 3. Grouped by aggregation strategy
    if 'Aggregation' in final_metrics_df.columns and not final_metrics_df['Aggregation'].eq('Unknown').all():
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=final_metrics_df, x='Aggregation', y='Final Accuracy', ax=ax)
        ax.set_title('BC: Accuracy by Aggregation Strategy')
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        output_path = os.path.join(VIS_DIR, "comparisons", "bc_aggregation_comparison.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"BC Plot saved to {output_path}")
    
    # 4. IID vs Non-IID
    if 'Data' in final_metrics_df.columns and len(final_metrics_df['Data'].unique()) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=final_metrics_df, x='Data', y='Final Accuracy', ax=ax)
        ax.set_title('BC: IID vs Non-IID Comparison')
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        output_path = os.path.join(VIS_DIR, "comparisons", "bc_iid_vs_noniid_comparison.png")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"BC Plot saved to {output_path}")
    
    logger.info("Basic visualization complete. Check the output directory for plots.")

def main():
    parser = argparse.ArgumentParser(description="Visualize Breast Cancer Federated Learning Results")
    parser.add_argument('--results', type=str, required=True, 
                        help='Path to the results CSV file')
    args = parser.parse_args()
    
    results_path = args.results
    
    logger.info(f"Loading results from: {results_path}")
    try:
        # Load the results DataFrame
        results_df = load_results_from_csv(results_path)
        
        # Convert the DataFrame to a list of dictionaries (format expected by plot functions)
        results_list = convert_df_to_results_list(results_df)
        
        # Try using the enhanced visualization if available
        if has_enhanced_viz:
            try:
                logger.info("Attempting to use enhanced visualization with processed data...")
                enhanced_plot_bc_results(results_list)  # Pass the processed list instead of the file path
                logger.info("Enhanced visualization completed successfully")
            except Exception as e:
                logger.error(f"Enhanced visualization failed: {e}", exc_info=True)
                logger.info("Falling back to basic visualization...")
                plot_bc_results(results_list)
        else:
            # Use basic visualization
            plot_bc_results(results_list)
            
    except Exception as e:
        logger.error(f"Error processing results: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("Visualization complete")

if __name__ == "__main__":
    main()