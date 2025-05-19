"""
Utility module to integrate advanced visualizations with the existing experiment code.
This module provides a drop-in replacement for the plot_bc_results function in run_bc_experiment.py.
"""
import os
import sys
import logging
from typing import List, Dict, Any

# Setup project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(PROJECT_ROOT, 'src')
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Import the advanced visualization module
from visualization.advanced_plots import AdvancedVisualization

# Get a logger
logger = logging.getLogger("VisualizationIntegration")

def enhanced_plot_bc_results(all_results: List[Dict]) -> None:
    """
    Enhanced version of plot_bc_results that uses the advanced visualization module.
    This function can be used as a drop-in replacement for the existing plot_bc_results function.
    
    Args:
        all_results: List of experiment result dictionaries
    """
    # Ensure we have results to visualize
    if not all_results:
        logger.warning("No results to visualize")
        return
    
    # Setup paths
    EXPERIMENTS_BASE_DIR = os.path.join(PROJECT_ROOT, 'experiments')
    ADVANCED_VIS_DIR = os.path.join(EXPERIMENTS_BASE_DIR, 'advanced_vis_bc')
    
    # Create the advanced visualization directory
    os.makedirs(ADVANCED_VIS_DIR, exist_ok=True)
    
    # Initialize our advanced visualization module
    try:
        logger.info("Generating advanced visualizations for BC experiments...")
        vis = AdvancedVisualization(output_dir=ADVANCED_VIS_DIR, dpi=300)
        
        # Generate all visualizations
        vis.generate_all_visualizations(all_results)
        logger.info(f"All advanced visualizations generated successfully in {ADVANCED_VIS_DIR}")
    except Exception as e:
        logger.error(f"Error generating advanced visualizations: {e}")
        logger.info("Falling back to basic visualizations...")
        
        # Import the original plotting function from the global scope
        # Since this is a drop-in replacement, we need to import the original function
        # from the module that's calling this one
        current_frame = sys._getframe(1)
        if 'plot_bc_results' in current_frame.f_globals:
            # Call the original function
            original_plot_bc_results = current_frame.f_globals['plot_bc_results']
            original_plot_bc_results(all_results)
        else:
            logger.error("Could not find original plot_bc_results function for fallback")