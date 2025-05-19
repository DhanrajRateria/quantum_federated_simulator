import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy import stats
import matplotlib.patches as mpatches
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import math

# Configure logger
logger = logging.getLogger("AdvancedVisualization")

# Define custom color palettes
QML_PALETTE = {
    'classical': '#1f77b4',  # blue
    'vqc': '#ff7f0e',        # orange
    'qnn': '#2ca02c',        # green
    'hybrid': '#d62728',     # red
    'fedavg': '#7f7f7f',     # gray
    'fedprox': '#9467bd',    # purple
    'feddive': '#8c564b',    # brown
    'iid': '#e377c2',        # pink
    'noniid': '#bcbd22',     # olive
}

# Create custom colormaps
QUANTUM_CMAP = LinearSegmentedColormap.from_list('quantum_gradient', 
                                               ['#1f77b4', '#d1e5f0', '#fee0b6', '#ff7f0e'])
ACCURACY_CMAP = LinearSegmentedColormap.from_list('accuracy_gradient', 
                                                ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4'])

class AdvancedVisualization:
    """Class for creating advanced, publication-quality visualizations for federated learning experiments."""
    
    def __init__(self, output_dir: str, dpi: int = 300, style: str = 'whitegrid', 
                 context: str = 'paper', font_scale: float = 1.2):
        """
        Initialize the visualization class.
        
        Args:
            output_dir: Directory to save visualizations
            dpi: Resolution for saved figures
            style: Seaborn style ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
            context: Seaborn context ('paper', 'notebook', 'talk', 'poster')
            font_scale: Scale factor for font sizes
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.font_family = 'sans-serif'
        self.font_scale = font_scale
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set(style=style, context=context, font_scale=font_scale, 
                rc={'font.family': self.font_family, 'axes.linewidth': 1.5,
                    'axes.edgecolor': '0.2', 'xtick.major.width': 1.2,
                    'ytick.major.width': 1.2, 'xtick.minor.width': 1.0,
                    'ytick.minor.width': 1.0, 'grid.linestyle': '--',
                    'grid.linewidth': 0.8, 'grid.alpha': 0.8})
        
        # Configure matplotlib for better defaults
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['savefig.pad_inches'] = 0.1
        plt.rcParams['axes.titlepad'] = 14.0
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 0.9
        plt.rcParams['legend.edgecolor'] = '0.8'
        plt.rcParams['legend.fancybox'] = True
        
        logger.info(f"Advanced visualization initialized with output directory: {output_dir}")
    
    def _extract_model_type(self, exp_name: str) -> str:
        """Extract model type from experiment name."""
        if 'MLP' in exp_name:
            return 'classical'
        elif 'VQC' in exp_name:
            return 'vqc'
        elif 'QNN' in exp_name and 'Hybrid' not in exp_name:
            return 'qnn'
        elif 'Hybrid' in exp_name:
            return 'hybrid'
        return 'unknown'
    
    def _extract_agg_strategy(self, exp_name: str) -> str:
        """Extract aggregation strategy from experiment name."""
        if 'FedAvg' in exp_name:
            return 'FedAvg'
        elif 'FedProx' in exp_name:
            return 'FedProx'
        elif 'FedDive' in exp_name:
            return 'FedDive'
        return 'unknown'
    
    def _extract_data_partition(self, exp_name: str) -> str:
        """Extract data partitioning strategy from experiment name."""
        if 'IID' in exp_name and 'NonIID' not in exp_name:
            return 'IID'
        elif 'NonIID' in exp_name:
            return 'NonIID'
        return 'unknown'
    
    def _get_color_by_model(self, model_type: str) -> str:
        """Get color based on model type."""
        return QML_PALETTE.get(model_type.lower(), '#333333')
    
    def _get_marker_by_agg(self, agg_strategy: str) -> str:
        """Get marker style based on aggregation strategy."""
        markers = {'FedAvg': 'o', 'FedProx': 's', 'FedDive': '^', 'unknown': 'x'}
        return markers.get(agg_strategy, 'o')
    
    def _get_linestyle_by_data(self, data_partition: str) -> str:
        """Get line style based on data partitioning."""
        styles = {'IID': '-', 'NonIID': '--', 'unknown': '-.'}
        return styles.get(data_partition, '-')
    
    def _prepare_experiment_metadata(self, results: List[Dict]) -> pd.DataFrame:
        """
        Process experiment results and extract metadata for visualization.
        
        Args:
            results: List of experiment result dictionaries
            
        Returns:
            DataFrame with processed experiment data
        """
        logger.info("Preparing experiment metadata for visualization...")
        
        # Process experiment metadata
        experiment_meta = []
        for result in results:
            exp_name = result.get('experiment_name', 'Unknown')
            config = result.get('config', {})
            
            # Extract key information
            model_type = self._extract_model_type(exp_name)
            agg_strategy = self._extract_agg_strategy(exp_name)
            data_partition = self._extract_data_partition(exp_name)
            
            # Get more detailed config info if available
            model_config = config.get('model', {})
            data_config = config.get('data_config', {})
            
            # Extract final metrics
            final_metrics = {}
            round_history = result.get('round_history', [])
            if round_history and len(round_history) > 0:
                final_round = round_history[-1]
                final_metrics = final_round.get('evaluation_metrics', {})
                
                # Calculate total time and communication
                total_time = sum(round.get('duration_seconds', 0) for round in round_history)
                total_comm_up = sum(round.get('comm_total_upload_bytes', 0) for round in round_history)
                total_comm_down = sum(
                    round.get('comm_download_bytes_per_client', 0) * len(round.get('participating_clients', []))
                    for round in round_history
                )
            else:
                total_time = 0
                total_comm_up = 0
                total_comm_down = 0
            
            experiment_meta.append({
                'Experiment': exp_name,
                'Model Type': model_type,
                'Aggregation': agg_strategy,
                'Data Partition': data_partition,
                'PCA Features': data_config.get('pca_features', 'None'),
                'Hidden Dim': model_config.get('hidden_dim', None),
                'Quantum Layers': model_config.get('n_layers', model_config.get('q_layers', None)),
                'Circuit Type': model_config.get('circuit_type', model_config.get('q_circuit_type', None)),
                'Final Accuracy': final_metrics.get('accuracy', None),
                'Final Loss': final_metrics.get('loss', None),
                'Total Time (s)': total_time,
                'Total Communication (MB)': (total_comm_up + total_comm_down) / (1024 * 1024),
                'Color': self._get_color_by_model(model_type),
                'Marker': self._get_marker_by_agg(agg_strategy),
                'LineStyle': self._get_linestyle_by_data(data_partition)
            })
        
        return pd.DataFrame(experiment_meta)
    
    def _prepare_round_data(self, results: List[Dict]) -> pd.DataFrame:
        """
        Process round-by-round data for visualization.
        
        Args:
            results: List of experiment result dictionaries
            
        Returns:
            DataFrame with processed round data
        """
        logger.info("Preparing round-by-round data for visualization...")
        
        rounds_data = []
        for result in results:
            exp_name = result.get('experiment_name', 'Unknown')
            
            # Skip failed experiments
            if result.get("error") or not result.get('round_history'):
                logger.warning(f"Skipping failed experiment: {exp_name}")
                continue
            
            # Extract metadata
            model_type = self._extract_model_type(exp_name)
            agg_strategy = self._extract_agg_strategy(exp_name)
            data_partition = self._extract_data_partition(exp_name)
            
            # Process round history
            cumulative_time, cumulative_comm_up, cumulative_comm_down = 0, 0, 0
            
            for round_data in result.get('round_history', []):
                metrics = round_data.get('evaluation_metrics', {})
                round_num = round_data.get('round', 0)
                
                # Update cumulative metrics
                cumulative_time += round_data.get('duration_seconds', 0)
                cumulative_comm_up += round_data.get('comm_total_upload_bytes', 0)
                cumulative_comm_down += (
                    round_data.get('comm_download_bytes_per_client', 0) * 
                    len(round_data.get('participating_clients', []))
                )
                
                # Append data point
                rounds_data.append({
                    'Experiment': exp_name,
                    'Round': round_num,
                    'Accuracy': metrics.get('accuracy'),
                    'Loss': metrics.get('loss'),
                    'Cumulative Time (s)': cumulative_time,
                    'Cumulative Upload (MB)': cumulative_comm_up / (1024 * 1024),
                    'Cumulative Download (MB)': cumulative_comm_down / (1024 * 1024),
                    'Cumulative Communication (MB)': (cumulative_comm_up + cumulative_comm_down) / (1024 * 1024),
                    'Model Type': model_type,
                    'Aggregation': agg_strategy,
                    'Data Partition': data_partition,
                    'Color': self._get_color_by_model(model_type),
                    'Marker': self._get_marker_by_agg(agg_strategy),
                    'LineStyle': self._get_linestyle_by_data(data_partition)
                })
        
        return pd.DataFrame(rounds_data)
    
    def save_figure(self, fig: plt.Figure, filename: str, subdir: Optional[str] = None) -> str:
        """
        Save figure to the output directory.
        
        Args:
            fig: Matplotlib figure to save
            filename: Name of the file (without extension)
            subdir: Optional subdirectory within output_dir
            
        Returns:
            Path to the saved figure
        """
        if subdir:
            save_dir = os.path.join(self.output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
        else:
            save_dir = self.output_dir
            
        filepath = os.path.join(save_dir, f"{filename}.png")
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Figure saved to: {filepath}")
        
        # Also save as PDF for publication quality
        pdf_path = os.path.join(save_dir, f"{filename}.pdf")
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        
        plt.close(fig)
        return filepath
    
    def plot_accuracy_curves(self, results: List[Dict], highlight_best: bool = True) -> None:
        """
        Plot accuracy curves with advanced styling.
        
        Args:
            results: List of experiment result dictionaries
            highlight_best: Whether to highlight the best-performing model
        """
        logger.info("Generating advanced accuracy curves...")
        
        # Process data
        rounds_df = self._prepare_round_data(results)
        if rounds_df.empty:
            logger.warning("No round data available for plotting accuracy curves")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a plot with custom styling for each experiment
        for experiment, group in rounds_df.groupby('Experiment'):
            ax.plot(
                group['Round'], group['Accuracy'],
                marker=group['Marker'].iloc[0],
                linestyle=group['LineStyle'].iloc[0],
                color=group['Color'].iloc[0],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
                label=experiment
            )
        
        # Highlight best performing model if requested
        if highlight_best and not rounds_df.empty:
            # Get the experiment with the highest final accuracy
            final_rounds = rounds_df.loc[rounds_df.groupby('Experiment')['Round'].idxmax()]
            best_exp = final_rounds.loc[final_rounds['Accuracy'].idxmax()]
            
            # Add annotation for the best model
            best_acc = best_exp['Accuracy']
            best_round = best_exp['Round']
            ax.plot(best_round, best_acc, 'o', ms=12, mec='gold', mfc='none', mew=2)
            ax.annotate(
                f'Best: {best_acc:.4f}',
                xy=(best_round, best_acc),
                xytext=(best_round - 0.5, best_acc + 0.05),
                fontsize=12,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black')
            )
        
        # Add reference line for random classifier
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
        ax.text(0.02, 0.51, 'Random Classifier', fontsize=9, color='gray')
        
        # Enhance the plot appearance
        ax.set_title('Accuracy Progression Over Federated Rounds', fontsize=18, pad=20)
        ax.set_xlabel('Federated Round', fontsize=14, labelpad=10)
        ax.set_ylabel('Validation Accuracy', fontsize=14, labelpad=10)
        
        # Set appropriate limits with padding
        ax.set_xlim(-0.5, max(rounds_df['Round']) + 0.5)
        y_min = max(0, min(rounds_df['Accuracy'].dropna()) - 0.05)
        y_max = min(1.0, max(rounds_df['Accuracy'].dropna()) + 0.05)
        ax.set_ylim(y_min, y_max)
        
        # Ensure ticks are integers for rounds
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a custom legend with model type indicators
        handles, labels = ax.get_legend_handles_labels()
        
        # Create a figure legend with more information
        legend = ax.legend(
            handles, labels, 
            title='Experiments',
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor='lightgray',
            title_fontsize=14
        )
        
        # Create a secondary legend for model types
        model_patches = []
        for model_type, color in [('Classical', QML_PALETTE['classical']), 
                                 ('VQC', QML_PALETTE['vqc']), 
                                 ('QNN', QML_PALETTE['qnn']),
                                 ('Hybrid', QML_PALETTE['hybrid'])]:
            model_patches.append(mpatches.Patch(color=color, label=model_type))
        
        # Create a third legend for aggregation strategies
        agg_markers = []
        for agg, marker in [('FedAvg', 'o'), ('FedProx', 's'), ('FedDive', '^')]:
            agg_markers.append(plt.Line2D([0], [0], marker=marker, color='black', 
                                         label=agg, markersize=8, linestyle='None'))
        
        # Create a fourth legend for data partitioning
        data_lines = []
        for part, style in [('IID', '-'), ('NonIID', '--')]:
            data_lines.append(plt.Line2D([0], [0], color='black', 
                                       label=part, linestyle=style, linewidth=2))
        
        # Add the secondary legends below the main one
        ax.legend(handles=model_patches, loc='lower center', bbox_to_anchor=(0.5, -0.13), 
                 ncol=4, title="Model Types", frameon=True, title_fontsize=12)
        
        # Adjust layout to make room for the legends
        plt.subplots_adjust(bottom=0.18, right=0.8)
        
        # Save the figure
        self.save_figure(fig, "advanced_accuracy_vs_round", "learning_curves")
        
        # Create a version with the other legends too (more detailed)
        fig2 = plt.figure(figsize=(12, 10))
        ax2 = fig2.add_subplot(111)
        
        # Recreate the main plot
        for experiment, group in rounds_df.groupby('Experiment'):
            ax2.plot(
                group['Round'], group['Accuracy'],
                marker=group['Marker'].iloc[0],
                linestyle=group['LineStyle'].iloc[0],
                color=group['Color'].iloc[0],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
                label=experiment
            )
            
        # Add reference line for random classifier
        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
        ax2.text(0.02, 0.51, 'Random Classifier', fontsize=9, color='gray')
        
        # Highlight best performing model if requested
        if highlight_best and not rounds_df.empty:
            # Get the experiment with the highest final accuracy
            final_rounds = rounds_df.loc[rounds_df.groupby('Experiment')['Round'].idxmax()]
            best_exp = final_rounds.loc[final_rounds['Accuracy'].idxmax()]
            
            # Add annotation for the best model
            best_acc = best_exp['Accuracy']
            best_round = best_exp['Round']
            ax2.plot(best_round, best_acc, 'o', ms=12, mec='gold', mfc='none', mew=2)
            ax2.annotate(
                f'Best: {best_acc:.4f}',
                xy=(best_round, best_acc),
                xytext=(best_round - 0.5, best_acc + 0.05),
                fontsize=12,
                weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='black')
            )
        
        # Style the plot
        ax2.set_title('Accuracy Progression Over Federated Rounds', fontsize=18, pad=20)
        ax2.set_xlabel('Federated Round', fontsize=14, labelpad=10)
        ax2.set_ylabel('Validation Accuracy', fontsize=14, labelpad=10)
        
        # Set appropriate limits with padding
        ax2.set_xlim(-0.5, max(rounds_df['Round']) + 0.5)
        y_min = max(0, min(rounds_df['Accuracy'].dropna()) - 0.05)
        y_max = min(1.0, max(rounds_df['Accuracy'].dropna()) + 0.05)
        ax2.set_ylim(y_min, y_max)
        
        # Ensure ticks are integers for rounds
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add grid for readability
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add all the legends
        legend2 = ax2.legend(
            handles, labels, 
            title='Experiments',
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor='lightgray',
            title_fontsize=14
        )
        
        # Add the legend for model types at the bottom
        legend_model = fig2.legend(handles=model_patches, loc='lower center', 
                                  bbox_to_anchor=(0.5, 0.02), ncol=4, 
                                  title="Model Types", frameon=True)
        
        # Add legend for aggregation strategies and data partitioning side by side
        legend_agg = fig2.legend(handles=agg_markers, loc='lower left', 
                                bbox_to_anchor=(0.1, 0.02), ncol=3, 
                                title="Aggregation", frameon=True)
        
        legend_data = fig2.legend(handles=data_lines, loc='lower right', 
                                 bbox_to_anchor=(0.9, 0.02), ncol=2, 
                                 title="Data Distribution", frameon=True)
        
        # Make sure legend_model is on top of the others
        ax2.add_artist(legend2)
        fig2.add_artist(legend_model)
        fig2.add_artist(legend_agg)
        
        # Adjust layout
        plt.subplots_adjust(bottom=0.2, right=0.8)
        
        # Save the detailed version
        self.save_figure(fig2, "advanced_accuracy_vs_round_detailed", "learning_curves")
    
    def plot_loss_curves(self, results: List[Dict]) -> None:
        """
        Plot loss curves with advanced styling.
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating advanced loss curves...")
        
        # Process data
        rounds_df = self._prepare_round_data(results)
        if rounds_df.empty:
            logger.warning("No round data available for plotting loss curves")
            return
            
        # Filter out experiments with no loss data
        rounds_df_with_loss = rounds_df.dropna(subset=['Loss'])
        if rounds_df_with_loss.empty:
            logger.warning("No loss data available for any experiment")
            return
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a plot with custom styling for each experiment
        for experiment, group in rounds_df_with_loss.groupby('Experiment'):
            ax.plot(
                group['Round'], group['Loss'],
                marker=group['Marker'].iloc[0],
                linestyle=group['LineStyle'].iloc[0],
                color=group['Color'].iloc[0],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
                label=experiment
            )
        
        # Enhance the plot appearance
        ax.set_title('Loss Progression Over Federated Rounds', fontsize=18, pad=20)
        ax.set_xlabel('Federated Round', fontsize=14, labelpad=10)
        ax.set_ylabel('Validation Loss', fontsize=14, labelpad=10)
        
        # Set appropriate limits with padding
        ax.set_xlim(-0.5, max(rounds_df_with_loss['Round']) + 0.5)
        
        # Ensure ticks are integers for rounds
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Add grid for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add a custom legend
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(
            handles, labels, 
            title='Experiments',
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor='lightgray',
            title_fontsize=14
        )
        
        # Adjust layout
        plt.subplots_adjust(right=0.8)
        
        # Save the figure
        self.save_figure(fig, "advanced_loss_vs_round", "learning_curves")
    
    def plot_accuracy_vs_resources(self, results: List[Dict]) -> None:
        """
        Plot accuracy vs computational resources (time and communication).
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating accuracy vs resources plots...")
        
        # Process data
        rounds_df = self._prepare_round_data(results)
        if rounds_df.empty:
            logger.warning("No round data available for plotting accuracy vs resources")
            return
            
        # Create figures - one for time, one for communication
        fig_time, ax_time = plt.subplots(figsize=(12, 8))
        fig_comm, ax_comm = plt.subplots(figsize=(12, 8))
        
        # Plot accuracy vs time
        for experiment, group in rounds_df.groupby('Experiment'):
            ax_time.plot(
                group['Cumulative Time (s)'], group['Accuracy'],
                marker=group['Marker'].iloc[0],
                linestyle=group['LineStyle'].iloc[0],
                color=group['Color'].iloc[0],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
                label=experiment
            )
            
            # Plot accuracy vs communication
            ax_comm.plot(
                group['Cumulative Communication (MB)'], group['Accuracy'],
                marker=group['Marker'].iloc[0],
                linestyle=group['LineStyle'].iloc[0],
                color=group['Color'].iloc[0],
                linewidth=2.5,
                markersize=8,
                alpha=0.9,
                label=experiment
            )
        
        # Style time plot
        ax_time.set_title('Accuracy vs. Computational Time', fontsize=18, pad=20)
        ax_time.set_xlabel('Cumulative Training Time (seconds)', fontsize=14, labelpad=10)
        ax_time.set_ylabel('Validation Accuracy', fontsize=14, labelpad=10)
        
        # Add reference line for random classifier
        ax_time.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
        ax_time.text(min(rounds_df['Cumulative Time (s)']) + 1, 0.51, 'Random Classifier', fontsize=9, color='gray')
        
        # Set appropriate limits with padding for time plot
        y_min = max(0, min(rounds_df['Accuracy'].dropna()) - 0.05)
        y_max = min(1.0, max(rounds_df['Accuracy'].dropna()) + 0.05)
        ax_time.set_ylim(y_min, y_max)
        
        # Add grid for readability
        ax_time.grid(True, linestyle='--', alpha=0.7)
        
        # Add a custom legend
        handles_time, labels_time = ax_time.get_legend_handles_labels()
        legend_time = ax_time.legend(
            handles_time, labels_time, 
            title='Experiments',
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor='lightgray',
            title_fontsize=14
        )
        
        # Style communication plot
        ax_comm.set_title('Accuracy vs. Communication Cost', fontsize=18, pad=20)
        ax_comm.set_xlabel('Cumulative Communication (MB)', fontsize=14, labelpad=10)
        ax_comm.set_ylabel('Validation Accuracy', fontsize=14, labelpad=10)
        
        # Add reference line for random classifier
        ax_comm.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7, linewidth=1.5)
        ax_comm.text(
            min(rounds_df['Cumulative Communication (MB)']) + 0.1, 
            0.51, 
            'Random Classifier', 
            fontsize=9, 
            color='gray'
        )
        
        # Set appropriate limits with padding for communication plot
        ax_comm.set_ylim(y_min, y_max)
        
        # Add grid for readability
        ax_comm.grid(True, linestyle='--', alpha=0.7)
        
        # Add a custom legend
        handles_comm, labels_comm = ax_comm.get_legend_handles_labels()
        legend_comm = ax_comm.legend(
            handles_comm, labels_comm, 
            title='Experiments',
            loc='center left', 
            bbox_to_anchor=(1.02, 0.5),
            fontsize=12,
            frameon=True,
            fancybox=True,
            framealpha=0.9,
            edgecolor='lightgray',
            title_fontsize=14
        )
        
        # Adjust layout
        plt.figure(fig_time.number)
        plt.subplots_adjust(right=0.8)
        
        plt.figure(fig_comm.number)
        plt.subplots_adjust(right=0.8)
        
        # Save the figures
        self.save_figure(fig_time, "advanced_accuracy_vs_time", "resource_analysis")
        self.save_figure(fig_comm, "advanced_accuracy_vs_communication", "resource_analysis")
    
    def plot_comparative_barplots(self, results: List[Dict]) -> None:
        """
        Create comparative bar plots for final metrics across experiments.
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating comparative bar plots...")
        
        # Prepare experiment metadata
        exp_meta_df = self._prepare_experiment_metadata(results)
        if exp_meta_df.empty:
            logger.warning("No metadata available for comparative bar plots")
            return
        
        # Create accuracy comparison
        fig_acc, ax_acc = plt.subplots(figsize=(14, max(8, len(exp_meta_df) * 0.5)))
        
        # Sort by accuracy for better visualization
        exp_meta_df_sorted = exp_meta_df.sort_values('Final Accuracy', ascending=True)
        
        # Create horizontal bar plot with custom colors
        bars = ax_acc.barh(
            exp_meta_df_sorted['Experiment'], 
            exp_meta_df_sorted['Final Accuracy'],
            color=exp_meta_df_sorted['Color'],
            alpha=0.9,
            edgecolor='black',
            linewidth=1.2,
            height=0.6
        )
        
        # Add value labels to the bars
        for bar in bars:
            width = bar.get_width()
            ax_acc.text(
                width + 0.01,  # Position text at end of bar
                bar.get_y() + bar.get_height()/2,  # Vertical position (middle of bar)
                f'{width:.4f}',  # Text (formatted to 4 decimal places)
                va='center',  # Vertical alignment
                fontsize=10,
                fontweight='bold'
            )
        
        # Add reference line for random classifier
        ax_acc.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax_acc.text(0.51, -0.5, 'Random Classifier (0.5)', fontsize=10, color='red')
        
        # Style the plot
        ax_acc.set_title('Final Accuracy Comparison Across Experiments', fontsize=18, pad=20)
        ax_acc.set_xlabel('Validation Accuracy', fontsize=14, labelpad=10)
        ax_acc.set_ylabel('Experiment', fontsize=14, labelpad=10)
        
        # Set x-axis limits for accuracy (0 to 1)
        ax_acc.set_xlim(0, 1.05)
        
        # Add grid for readability (only horizontal lines)
        ax_acc.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Save accuracy comparison
        self.save_figure(fig_acc, "advanced_accuracy_comparison", "comparative_analysis")
        
        # Create time comparison
        fig_time, ax_time = plt.subplots(figsize=(14, max(8, len(exp_meta_df) * 0.5)))
        
        # Sort by computational time
        exp_meta_df_sorted_time = exp_meta_df.sort_values('Total Time (s)', ascending=True)
        
        # Create horizontal bar plot
        bars_time = ax_time.barh(
            exp_meta_df_sorted_time['Experiment'], 
            exp_meta_df_sorted_time['Total Time (s)'],
            color=exp_meta_df_sorted_time['Color'],
            alpha=0.9,
            edgecolor='black',
            linewidth=1.2,
            height=0.6
        )
        
        # Add value labels
        for bar in bars_time:
            width = bar.get_width()
            ax_time.text(
                width + 0.5,  # Position text at end of bar
                bar.get_y() + bar.get_height()/2,  # Vertical position (middle of bar)
                f'{width:.1f}s',  # Text (formatted to 1 decimal place)
                va='center',  # Vertical alignment
                fontsize=10,
                fontweight='bold'
            )
        
        # Style the plot
        ax_time.set_title('Total Computational Time Comparison', fontsize=18, pad=20)
        ax_time.set_xlabel('Total Time (seconds)', fontsize=14, labelpad=10)
        ax_time.set_ylabel('Experiment', fontsize=14, labelpad=10)
        
        # Add grid for readability (only horizontal lines)
        ax_time.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Save time comparison
        self.save_figure(fig_time, "advanced_time_comparison", "comparative_analysis")
        
        # Create communication comparison
        fig_comm, ax_comm = plt.subplots(figsize=(14, max(8, len(exp_meta_df) * 0.5)))
        
        # Sort by communication cost
        exp_meta_df_sorted_comm = exp_meta_df.sort_values('Total Communication (MB)', ascending=True)
        
        # Create horizontal bar plot
        bars_comm = ax_comm.barh(
            exp_meta_df_sorted_comm['Experiment'], 
            exp_meta_df_sorted_comm['Total Communication (MB)'],
            color=exp_meta_df_sorted_comm['Color'],
            alpha=0.9,
            edgecolor='black',
            linewidth=1.2,
            height=0.6
        )
        
        # Add value labels
        for bar in bars_comm:
            width = bar.get_width()
            ax_comm.text(
                width + 0.5,  # Position text at end of bar
                bar.get_y() + bar.get_height()/2,  # Vertical position (middle of bar)
                f'{width:.1f} MB',  # Text (formatted to 1 decimal place)
                va='center',  # Vertical alignment
                fontsize=10,
                fontweight='bold'
            )
        
        # Style the plot
        ax_comm.set_title('Total Communication Cost Comparison', fontsize=18, pad=20)
        ax_comm.set_xlabel('Total Communication (MB)', fontsize=14, labelpad=10)
        ax_comm.set_ylabel('Experiment', fontsize=14, labelpad=10)
        
        # Add grid for readability (only horizontal lines)
        ax_comm.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Save communication comparison
        self.save_figure(fig_comm, "advanced_communication_comparison", "comparative_analysis")
    
    def create_model_comparison_matrix(self, results: List[Dict]) -> None:
        """
        Create a matrix visualization comparing model types across metrics.
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating model comparison matrix...")
        
        # Prepare experiment metadata
        exp_meta_df = self._prepare_experiment_metadata(results)
        if exp_meta_df.empty:
            logger.warning("No metadata available for model comparison matrix")
            return
        
        # Group by model type and calculate average metrics
        model_type_summary = exp_meta_df.groupby('Model Type').agg({
            'Final Accuracy': ['mean', 'std', 'min', 'max', 'count'],
            'Total Time (s)': ['mean', 'std'],
            'Total Communication (MB)': ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-index columns
        model_type_summary.columns = [
            f'{col[0]}_{col[1]}' if col[1] else col[0] 
            for col in model_type_summary.columns
        ]
        
        # Calculate efficiency metrics
        model_type_summary['Accuracy_per_second'] = (
            model_type_summary['Final Accuracy_mean'] / 
            model_type_summary['Total Time (s)_mean']
        )
        
        model_type_summary['Accuracy_per_MB'] = (
            model_type_summary['Final Accuracy_mean'] / 
            model_type_summary['Total Communication (MB)_mean']
        )
        
        # Create the matrix plot
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        # Upper left: Accuracy with error bars
        ax1 = plt.subplot(gs[0, 0])
        bars1 = ax1.bar(
            model_type_summary['Model Type'],
            model_type_summary['Final Accuracy_mean'],
            yerr=model_type_summary['Final Accuracy_std'],
            capsize=7,
            color=[QML_PALETTE[m.lower()] for m in model_type_summary['Model Type']],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.01,
                f'{height:.4f}',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax1.set_title('Average Accuracy by Model Type', fontsize=16)
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('Validation Accuracy', fontsize=12)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add number of experiments as text under each bar
        for i, count in enumerate(model_type_summary['Final Accuracy_count']):
            ax1.text(
                i, -0.05,
                f'n={count}',
                ha='center',
                fontsize=9
            )
        
        # Add random classifier reference line
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(0, 0.51, 'Random', fontsize=9, color='red')
        
        # Upper right: Time with error bars
        ax2 = plt.subplot(gs[0, 1])
        bars2 = ax2.bar(
            model_type_summary['Model Type'],
            model_type_summary['Total Time (s)_mean'],
            yerr=model_type_summary['Total Time (s)_std'],
            capsize=7,
            color=[QML_PALETTE[m.lower()] for m in model_type_summary['Model Type']],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        
        # Add value labels
        for bar in bars2:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.2,
                f'{height:.1f}s',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax2.set_title('Average Computation Time by Model Type', fontsize=16)
        ax2.set_ylabel('Total Time (seconds)', fontsize=12)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Lower left: Accuracy per Time (efficiency)
        ax3 = plt.subplot(gs[1, 0])
        bars3 = ax3.bar(
            model_type_summary['Model Type'],
            model_type_summary['Accuracy_per_second'],
            color=[QML_PALETTE[m.lower()] for m in model_type_summary['Model Type']],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2,
                height + height*0.02,
                f'{height:.5f}',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax3.set_title('Accuracy per Second (Efficiency)', fontsize=16)
        ax3.set_ylabel('Accuracy / Second', fontsize=12)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Lower right: Communication with error bars
        ax4 = plt.subplot(gs[1, 1])
        bars4 = ax4.bar(
            model_type_summary['Model Type'],
            model_type_summary['Total Communication (MB)_mean'],
            yerr=model_type_summary['Total Communication (MB)_std'],
            capsize=7,
            color=[QML_PALETTE[m.lower()] for m in model_type_summary['Model Type']],
            edgecolor='black',
            linewidth=1.2,
            alpha=0.9
        )
        
        # Add value labels
        for bar in bars4:
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width()/2,
                height + 0.1,
                f'{height:.1f} MB',
                ha='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax4.set_title('Average Communication Cost by Model Type', fontsize=16)
        ax4.set_ylabel('Total Communication (MB)', fontsize=12)
        ax4.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add an overall title
        fig.suptitle('Quantum vs. Classical Model Performance Matrix', fontsize=20, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the matrix
        self.save_figure(fig, "model_comparison_matrix", "comparative_analysis")
    
    def create_heatmap_analysis(self, results: List[Dict]) -> None:
        """
        Create heatmap visualization for comparing different aspects of experiments.
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating heatmap analysis...")
        
        # Prepare experiment metadata
        exp_meta_df = self._prepare_experiment_metadata(results)
        if exp_meta_df.empty:
            logger.warning("No metadata available for heatmap analysis")
            return
        
        # Create a pivot table for model type vs aggregation strategy (accuracy)
        pivot_acc = pd.pivot_table(
            exp_meta_df, 
            values='Final Accuracy',
            index='Model Type',
            columns='Aggregation',
            aggfunc='mean'
        )
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with custom colormap
        sns.heatmap(
            pivot_acc,
            annot=True,
            cmap=ACCURACY_CMAP,
            vmin=0.5,  # Minimum value (random classifier)
            vmax=1.0,  # Maximum possible accuracy
            linewidths=0.5,
            cbar_kws={'label': 'Mean Accuracy'},
            fmt='.4f',
            ax=ax
        )
        
        # Style the plot
        ax.set_title('Mean Accuracy by Model Type and Aggregation Strategy', fontsize=18, pad=20)
        ax.set_ylabel('Model Type', fontsize=14, labelpad=10)
        ax.set_xlabel('Aggregation Strategy', fontsize=14, labelpad=10)
        
        # Save the accuracy heatmap
        self.save_figure(fig, "accuracy_heatmap_by_model_agg", "heatmap_analysis")
        
        # Create a pivot table for model type vs data partition (accuracy)
        pivot_data_acc = pd.pivot_table(
            exp_meta_df, 
            values='Final Accuracy',
            index='Model Type',
            columns='Data Partition',
            aggfunc='mean'
        )
        
        # Create figure
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with custom colormap
        sns.heatmap(
            pivot_data_acc,
            annot=True,
            cmap=ACCURACY_CMAP,
            vmin=0.5,  # Minimum value (random classifier)
            vmax=1.0,  # Maximum possible accuracy
            linewidths=0.5,
            cbar_kws={'label': 'Mean Accuracy'},
            fmt='.4f',
            ax=ax2
        )
        
        # Style the plot
        ax2.set_title('Mean Accuracy by Model Type and Data Partition', fontsize=18, pad=20)
        ax2.set_ylabel('Model Type', fontsize=14, labelpad=10)
        ax2.set_xlabel('Data Partition Strategy', fontsize=14, labelpad=10)
        
        # Save the data partition heatmap
        self.save_figure(fig2, "accuracy_heatmap_by_model_data", "heatmap_analysis")
        
        # Create a time heatmap
        # Create a pivot table for model type vs aggregation strategy (time)
        pivot_time = pd.pivot_table(
            exp_meta_df, 
            values='Total Time (s)',
            index='Model Type',
            columns='Aggregation',
            aggfunc='mean'
        )
        
        # Create figure
        fig3, ax3 = plt.subplots(figsize=(12, 10))
        
        # Create heatmap with reverse colormap (lower is better for time)
        sns.heatmap(
            pivot_time,
            annot=True,
            cmap='viridis_r',  # Reversed viridis (darker = higher values)
            linewidths=0.5,
            cbar_kws={'label': 'Mean Time (seconds)'},
            fmt='.1f',
            ax=ax3
        )
        
        # Style the plot
        ax3.set_title('Mean Computation Time by Model Type and Aggregation Strategy', fontsize=18, pad=20)
        ax3.set_ylabel('Model Type', fontsize=14, labelpad=10)
        ax3.set_xlabel('Aggregation Strategy', fontsize=14, labelpad=10)
        
        # Save the time heatmap
        self.save_figure(fig3, "time_heatmap_by_model_agg", "heatmap_analysis")
    
    def create_statistical_analysis(self, results: List[Dict]) -> None:
        """
        Create statistical analysis of experiment results.
        
        Args:
            results: List of experiment result dictionaries
        """
        logger.info("Generating statistical analysis...")
        
        # Prepare experiment metadata
        exp_meta_df = self._prepare_experiment_metadata(results)
        if exp_meta_df.empty:
            logger.warning("No metadata available for statistical analysis")
            return
        
        # Check if we have enough data for statistical tests
        if len(exp_meta_df) < 3:
            logger.warning("Not enough experiments for meaningful statistical analysis")
            return
        
        # Create figure for statistical comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Type Box Plots (Accuracy)
        sns.boxplot(
            x='Model Type', 
            y='Final Accuracy',
            data=exp_meta_df,
            ax=axes[0, 0],
            palette={model: QML_PALETTE[model.lower()] for model in exp_meta_df['Model Type'].unique()}
        )
        axes[0, 0].set_title('Accuracy Distribution by Model Type', fontsize=14)
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add random classifier reference line
        axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add sample size annotation
        for i, model in enumerate(exp_meta_df['Model Type'].unique()):
            count = len(exp_meta_df[exp_meta_df['Model Type'] == model])
            axes[0, 0].annotate(
                f'n={count}',
                xy=(i, 0),
                xytext=(i, 0.05),
                ha='center',
                fontsize=9
            )
        
        # 2. Aggregation Strategy Box Plots (Accuracy)
        sns.boxplot(
            x='Aggregation', 
            y='Final Accuracy',
            data=exp_meta_df,
            ax=axes[0, 1]
        )
        axes[0, 1].set_title('Accuracy Distribution by Aggregation Strategy', fontsize=14)
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add random classifier reference line
        axes[0, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add sample size annotation
        for i, agg in enumerate(exp_meta_df['Aggregation'].unique()):
            count = len(exp_meta_df[exp_meta_df['Aggregation'] == agg])
            axes[0, 1].annotate(
                f'n={count}',
                xy=(i, 0),
                xytext=(i, 0.05),
                ha='center',
                fontsize=9
            )
        
        # 3. Data Partition Box Plots (Accuracy)
        sns.boxplot(
            x='Data Partition', 
            y='Final Accuracy',
            data=exp_meta_df,
            ax=axes[1, 0]
        )
        axes[1, 0].set_title('Accuracy Distribution by Data Partition', fontsize=14)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add random classifier reference line
        axes[1, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add sample size annotation
        for i, part in enumerate(exp_meta_df['Data Partition'].unique()):
            count = len(exp_meta_df[exp_meta_df['Data Partition'] == part])
            axes[1, 0].annotate(
                f'n={count}',
                xy=(i, 0),
                xytext=(i, 0.05),
                ha='center',
                fontsize=9
            )
        
        # 4. Scatter plot of Accuracy vs. Time with model types
        scatter = sns.scatterplot(
            x='Total Time (s)', 
            y='Final Accuracy',
            hue='Model Type',
            style='Aggregation',
            size='PCA Features',
            sizes=(50, 200),
            data=exp_meta_df,
            palette={model: QML_PALETTE[model.lower()] for model in exp_meta_df['Model Type'].unique()},
            ax=axes[1, 1]
        )
        
        # Add annotations for each point
        for idx, row in exp_meta_df.iterrows():
            axes[1, 1].annotate(
                row['Experiment'].split('_')[0:2],  # Just show first two parts of experiment name
                xy=(row['Total Time (s)'], row['Final Accuracy']),
                xytext=(5, 0),
                textcoords='offset points',
                fontsize=8,
                alpha=0.7
            )
        
        axes[1, 1].set_title('Accuracy vs. Computation Time', fontsize=14)
        axes[1, 1].grid(True, linestyle='--', alpha=0.7)
        axes[1, 1].set_ylim(0, 1)
        
        # Add random classifier reference line
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # Add overall title
        fig.suptitle('Statistical Analysis of Experiment Results', fontsize=18, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the statistical analysis
        self.save_figure(fig, "statistical_analysis", "statistical")
        
        # Generate ANOVA tests if we have enough samples
        model_types = exp_meta_df['Model Type'].unique()
        agg_strategies = exp_meta_df['Aggregation'].unique()
        
        # Open a text file for statistical results
        stats_file_path = os.path.join(self.output_dir, "statistical", "statistical_tests.txt")
        os.makedirs(os.path.dirname(stats_file_path), exist_ok=True)
        
        with open(stats_file_path, 'w') as f:
            f.write("Statistical Analysis of Experiment Results\n")
            f.write("=========================================\n\n")
            
            # Summary statistics
            f.write("Summary Statistics for Final Accuracy:\n")
            summary_stats = exp_meta_df.groupby('Model Type')['Final Accuracy'].agg(
                ['count', 'mean', 'std', 'min', 'max']
            )
            f.write(f"{summary_stats}\n\n")
            
            # ANOVA by Model Type
            if len(model_types) >= 2 and all(exp_meta_df.groupby('Model Type')['Final Accuracy'].count() >= 2):
                try:
                    model_groups = [exp_meta_df[exp_meta_df['Model Type'] == model]['Final Accuracy'].values 
                                  for model in model_types]
                    f_val, p_val = stats.f_oneway(*model_groups)
                    f.write(f"One-way ANOVA for Model Types:\n")
                    f.write(f"F-value: {f_val:.4f}\n")
                    f.write(f"p-value: {p_val:.4f}\n")
                    f.write(f"Statistically significant difference: {p_val < 0.05}\n\n")
                except Exception as e:
                    f.write(f"Error running ANOVA for Model Types: {e}\n\n")
            else:
                f.write("Not enough data for ANOVA test on Model Types\n\n")
                
            # ANOVA by Aggregation Strategy
            if len(agg_strategies) >= 2 and all(exp_meta_df.groupby('Aggregation')['Final Accuracy'].count() >= 2):
                try:
                    agg_groups = [exp_meta_df[exp_meta_df['Aggregation'] == agg]['Final Accuracy'].values 
                                 for agg in agg_strategies]
                    f_val, p_val = stats.f_oneway(*agg_groups)
                    f.write(f"One-way ANOVA for Aggregation Strategies:\n")
                    f.write(f"F-value: {f_val:.4f}\n")
                    f.write(f"p-value: {p_val:.4f}\n")
                    f.write(f"Statistically significant difference: {p_val < 0.05}\n\n")
                except Exception as e:
                    f.write(f"Error running ANOVA for Aggregation Strategies: {e}\n\n")
            else:
                f.write("Not enough data for ANOVA test on Aggregation Strategies\n\n")
                
            # ANOVA by Data Partition
            partition_types = exp_meta_df['Data Partition'].unique()
            if len(partition_types) >= 2 and all(exp_meta_df.groupby('Data Partition')['Final Accuracy'].count() >= 2):
                try:
                    partition_groups = [exp_meta_df[exp_meta_df['Data Partition'] == p]['Final Accuracy'].values 
                                      for p in partition_types]
                    f_val, p_val = stats.f_oneway(*partition_groups)
                    f.write(f"One-way ANOVA for Data Partition Strategies:\n")
                    f.write(f"F-value: {f_val:.4f}\n")
                    f.write(f"p-value: {p_val:.4f}\n")
                    f.write(f"Statistically significant difference: {p_val < 0.05}\n\n")
                except Exception as e:
                    f.write(f"Error running ANOVA for Data Partition Strategies: {e}\n\n")
            else:
                f.write("Not enough data for ANOVA test on Data Partition Strategies\n\n")
            
            f.write("Note: A p-value < 0.05 indicates a statistically significant difference between groups.\n")
        
        logger.info(f"Statistical test results saved to {stats_file_path}")
    
        def create_combined_dashboard(self, results: List[Dict]) -> None:
            """
            Create a comprehensive dashboard combining key visualizations.
            
            Args:
                results: List of experiment result dictionaries
            """
            logger.info("Generating combined visualization dashboard...")
            
            # Prepare experiment metadata and round data
            exp_meta_df = self._prepare_experiment_metadata(results)
            rounds_df = self._prepare_round_data(results)
            
            if exp_meta_df.empty or rounds_df.empty:
                logger.warning("Not enough data for creating dashboard")
                return
            
            # Create a large figure for the dashboard
            fig = plt.figure(figsize=(20, 24))
            gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1.2])
            
            # 1. Accuracy over rounds (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            
            for experiment, group in rounds_df.groupby('Experiment'):
                ax1.plot(
                    group['Round'], group['Accuracy'],
                    marker=group['Marker'].iloc[0],
                    linestyle=group['LineStyle'].iloc[0],
                    color=group['Color'].iloc[0],
                    linewidth=2,
                    markersize=6,
                    alpha=0.9,
                    label=experiment
                )
            
            ax1.set_title('Accuracy Progression', fontsize=14)
            ax1.set_xlabel('Federated Round', fontsize=12)
            ax1.set_ylabel('Validation Accuracy', fontsize=12)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.set_ylim(0.4, 1.0)
            
            # Reference line for random classifier
            ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
            
            # 2. Time efficiency (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            
            for experiment, group in rounds_df.groupby('Experiment'):
                ax2.plot(
                    group['Cumulative Time (s)'], group['Accuracy'],
                    marker=group['Marker'].iloc[0],
                    linestyle=group['LineStyle'].iloc[0],
                    color=group['Color'].iloc[0],
                    linewidth=2,
                    markersize=6,
                    alpha=0.9,
                    label=experiment
                )
            
            ax2.set_title('Time Efficiency', fontsize=14)
            ax2.set_xlabel('Cumulative Time (s)', fontsize=12)
            ax2.set_ylabel('Validation Accuracy', fontsize=12)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.set_ylim(0.4, 1.0)
            
            # 3. Final accuracy comparison (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Sort by accuracy for better visualization
            sorted_exps = exp_meta_df.sort_values('Final Accuracy', ascending=True)
            
            # Use only first 8 chars of experiment name for readability
            sorted_exps['Short Name'] = sorted_exps['Experiment'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
            
            bars = ax3.barh(
                sorted_exps['Short Name'][-6:],  # Show only top 6 for readability
                sorted_exps['Final Accuracy'][-6:],
                color=sorted_exps['Color'][-6:],
                edgecolor='black',
                linewidth=0.8,
                alpha=0.9
            )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax3.text(
                    width + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f'{width:.4f}',
                    va='center',
                    fontsize=9,
                    fontweight='bold'
                )
            
            ax3.set_title('Top Performing Experiments', fontsize=14)
            ax3.set_xlabel('Final Accuracy', fontsize=12)
            ax3.set_xlim(0, 1.02)
            ax3.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7)
            ax3.grid(True, axis='x', linestyle='--', alpha=0.7)
            
            # 4. Communication efficiency (middle right)
            ax4 = fig.add_subplot(gs[1, 1])
            
            for experiment, group in rounds_df.groupby('Experiment'):
                ax4.plot(
                    group['Cumulative Communication (MB)'], group['Accuracy'],
                    marker=group['Marker'].iloc[0],
                    linestyle=group['LineStyle'].iloc[0],
                    color=group['Color'].iloc[0],
                    linewidth=2,
                    markersize=6,
                    alpha=0.9,
                    label=experiment
                )
            
            ax4.set_title('Communication Efficiency', fontsize=14)
            ax4.set_xlabel('Cumulative Communication (MB)', fontsize=12)
            ax4.set_ylabel('Validation Accuracy', fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_ylim(0.4, 1.0)
            
            # 5. Model type comparison (lower left)
            ax5 = fig.add_subplot(gs[2, 0])
            
            model_type_summary = exp_meta_df.groupby('Model Type').agg({
                'Final Accuracy': ['mean', 'std']
            }).reset_index()
            
            model_type_summary.columns = [
                f'{col[0]}_{col[1]}' if col[1] else col[0] 
                for col in model_type_summary.columns
            ]
            
            bars5 = ax5.bar(
                model_type_summary['Model Type'],
                model_type_summary['Final Accuracy_mean'],
                yerr=model_type_summary['Final Accuracy_std'],
                capsize=7,
                color=[QML_PALETTE[m.lower()] for m in model_type_summary['Model Type']],
                edgecolor='black',
                linewidth=1.2,
                alpha=0.9
            )
            
            # Add value labels
            for bar in bars5:
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 0.01,
                    f'{height:.4f}',
                    ha='center',
                    fontsize=9,
                    fontweight='bold'
                )
            
            ax5.set_title('Average Accuracy by Model Type', fontsize=14)
            ax5.set_ylabel('Mean Accuracy', fontsize=12)
            ax5.set_ylim(0, 1.05)
            ax5.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax5.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
            
            # 6. Aggregation strategy comparison (lower right)
            ax6 = fig.add_subplot(gs[2, 1])
            
            agg_summary = exp_meta_df.groupby(['Aggregation', 'Data Partition']).agg({
                'Final Accuracy': 'mean'
            }).reset_index()
            
            # Create grouped bar chart
            sns.barplot(
                x='Aggregation', 
                y='Final Accuracy', 
                hue='Data Partition',
                data=agg_summary, 
                ax=ax6
            )
            
            ax6.set_title('Accuracy by Aggregation & Data Distribution', fontsize=14)
            ax6.set_ylabel('Mean Accuracy', fontsize=12)
            ax6.set_ylim(0, 1.05)
            ax6.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax6.axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
            ax6.legend(title='Data Partition')
            
            # 7. Combined legend (bottom)
            ax7 = fig.add_subplot(gs[3, :])
            ax7.axis('off')  # Turn off axis
            
            # Create legend entries
            handles = []
            labels = []
            
            # Model types
            for model in exp_meta_df['Model Type'].unique():
                handles.append(mpatches.Patch(color=QML_PALETTE[model.lower()], label=f"Model: {model}"))
            
            # Add spacing
            labels.append('')
            handles.append(mpatches.Patch(color='none', label=''))
            
            # Aggregation strategies
            for agg in exp_meta_df['Aggregation'].unique():
                marker = {'FedAvg': 'o', 'FedProx': 's', 'FedDive': '^'}.get(agg, 'x')
                handles.append(plt.Line2D([0], [0], marker=marker, color='black', 
                                        label=f"Aggregation: {agg}", markersize=8, linestyle='None'))
            
            # Add spacing
            labels.append('')
            handles.append(mpatches.Patch(color='none', label=''))
            
            # Data distribution
            for partition in exp_meta_df['Data Partition'].unique():
                linestyle = '-' if partition == 'IID' else '--'
                handles.append(plt.Line2D([0], [0], color='black', 
                                        label=f"Data: {partition}", linestyle=linestyle, linewidth=2))
            
            # Create the legend in a grid layout
            legend_cols = min(4, max(len(exp_meta_df['Model Type'].unique()),
                                    len(exp_meta_df['Aggregation'].unique()),
                                    len(exp_meta_df['Data Partition'].unique())))
            
            legend = ax7.legend(
                handles=handles,
                loc='center',
                ncol=legend_cols,
                fontsize=10,
                frameon=True,
                fancybox=True,
                framealpha=0.9,
                title="Experiment Legend Guide"
            )
            legend.get_title().set_fontsize(12)
            
            # Add experiment metadata summary as text
            text_info = (
                f"Total Experiments: {len(exp_meta_df)}\n"
                f"Model Types: {', '.join(exp_meta_df['Model Type'].unique())}\n"
                f"Aggregation Strategies: {', '.join(exp_meta_df['Aggregation'].unique())}\n"
                f"Data Distributions: {', '.join(exp_meta_df['Data Partition'].unique())}\n"
                f"Best Accuracy: {exp_meta_df['Final Accuracy'].max():.4f} "
                f"({exp_meta_df.loc[exp_meta_df['Final Accuracy'].idxmax(), 'Experiment']})\n"
                f"Quantum vs Classical: {len(exp_meta_df[exp_meta_df['Model Type'] != 'classical'])} quantum models, "
                f"{len(exp_meta_df[exp_meta_df['Model Type'] == 'classical'])} classical models"
            )
            
            ax7.text(
                0.5, 0.3,
                text_info,
                ha='center',
                va='center',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=1', facecolor='white', alpha=0.8, edgecolor='gray')
            )
            
            # Add overall title
            fig.suptitle('Federated Learning Experiment Dashboard', fontsize=20, y=0.98)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.subplots_adjust(hspace=0.3, wspace=0.3)
            
            # Save the dashboard
            self.save_figure(fig, "experiment_dashboard", "dashboard")
            
        def generate_all_visualizations(self, results: List[Dict]) -> None:
            """
            Generate all visualizations in one go.
            
            Args:
                results: List of experiment result dictionaries
            """
            logger.info("Generating all visualizations...")
            
            # Learning curves
            self.plot_accuracy_curves(results)
            self.plot_loss_curves(results)
            
            # Resource analysis
            self.plot_accuracy_vs_resources(results)
            
            # Comparative analysis
            self.plot_comparative_barplots(results)
            self.create_model_comparison_matrix(results)
            
            # Heatmaps
            self.create_heatmap_analysis(results)
            
            # Statistical analysis
            self.create_statistical_analysis(results)
            
            # Dashboard
            self.create_combined_dashboard(results)
            
            logger.info("All visualizations generated successfully!")