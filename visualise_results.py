# visualize_results.py
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Professional Plotting Configuration ---
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.4)
plt.rcParams.update({
    'figure.figsize': (10, 6), 'axes.labelsize': 18, 'axes.titlesize': 22,
    'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
    'figure.dpi': 300, 'lines.linewidth': 3.0, 'lines.markersize': 8
})
OUTPUT_DIR = "paper_experiments/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Label and Color Mapping for Consistency ---
LABEL_MAP = {
    "FEDAVG": "FedAvg", "FEDPROX": "FedProx", "MEDIAN": "Median",
    "FEDDIVE": "FedDive (Ours)", "FEDDIVER": "FedDive-R (Ours)"
}
COLOR_MAP = {
    "FedAvg": "#1f77b4", "FedProx": "#ff7f0e", "Median": "#2ca02c",
    "FedDive (Ours)": "#d62728", "FedDive-R (Ours)": "#9467bd"
}
def get_label(alg_type_str):
    return LABEL_MAP.get(alg_type_str.upper(), alg_type_str)

# --- Data Loading and Processing ---
def load_and_process_data(results_file):
    """Loads results and processes them into pandas DataFrames for easy plotting."""
    with open(results_file, 'r') as f:
        raw_data = json.load(f)

    # Flatten the data for performance curves
    history_records = []
    for exp in raw_data:
        config = exp['experiment_config']
        for run in exp['runs']:
            if 'error' in run: continue
            for round_data in run['round_history']:
                history_records.append({
                    'group': config['group'],
                    'algorithm': get_label(config['aggregation']['type']),
                    'seed': run['seed'],
                    'round': round_data['round'],
                    'accuracy': round_data['evaluation_metrics'].get('accuracy'),
                    'loss': round_data['evaluation_metrics'].get('loss')
                })
    
    # Create final results summary
    summary_records = []
    for exp in raw_data:
        config = exp['experiment_config']
        for run in exp['runs']:
            if 'error' in run: continue
            client_accs = list(run.get('fairness_metrics', {}).values())
            summary_records.append({
                'group': config['group'],
                'algorithm': get_label(config['aggregation']['type']),
                'temperature': config['aggregation'].get('temperature'),
                'seed': run['seed'],
                'final_accuracy': run['final_accuracy'],
                'final_loss': run['final_loss'],
                'convergence_rounds': run['convergence_rounds'],
                'fairness_std_dev': np.std(client_accs) if client_accs else 0
            })

    return pd.DataFrame(history_records), pd.DataFrame(summary_records), raw_data

# --- Table Generation ---
def generate_summary_table(df_summary, group, title):
    """Generates a markdown table summarizing results for a specific group."""
    table_df = df_summary[df_summary['group'] == group].groupby('algorithm').agg(
        final_accuracy_mean=('final_accuracy', 'mean'),
        final_accuracy_std=('final_accuracy', 'std'),
        final_loss_mean=('final_loss', 'mean'),
        final_loss_std=('final_loss', 'std'),
        convergence_mean=('convergence_rounds', 'mean'),
        fairness_std=('fairness_std_dev', 'mean') # Mean of std devs
    ).reset_index()

    table_df['Final Accuracy'] = table_df.apply(lambda r: f"{r.final_accuracy_mean:.2%} ± {r.final_accuracy_std:.2%}", axis=1)
    table_df['Final Loss'] = table_df.apply(lambda r: f"{r.final_loss_mean:.4f} ± {r.final_loss_std:.4f}", axis=1)
    table_df['Convergence (Rounds)'] = table_df['convergence_mean'].round(1)
    table_df['Fairness (Std Dev)'] = table_df['fairness_std'].round(4)
    
    # Reorder columns for presentation
    alg_order = [get_label(alg) for alg in ["FEDAVG", "FEDPROX", "MEDIAN", "FEDDIVE", "FEDDIVER"]]
    table_df['algorithm'] = pd.Categorical(table_df['algorithm'], categories=alg_order, ordered=True)
    table_df = table_df.sort_values('algorithm')
    
    final_table = table_df[['algorithm', 'Final Accuracy', 'Final Loss', 'Convergence (Rounds)', 'Fairness (Std Dev)']]
    
    md_table = final_table.to_markdown(index=False)
    filename = os.path.join(OUTPUT_DIR, f"table_{group.lower()}.md")
    with open(filename, 'w') as f:
        f.write(f"### {title}\n\n")
        f.write(md_table)
    print(f"Saved table: {filename}")
    return md_table

# --- Plotting Functions ---

def plot_performance_curves(df_history, group, title, filename_base):
    """Plots performance curves with error bands for mean and std dev across runs."""
    df_group = df_history[df_history['group'] == group]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 7), sharex=True)
    fig.suptitle(title, fontsize=24)
    
    # Accuracy Plot
    sns.lineplot(data=df_group, x='round', y='accuracy', hue='algorithm',
                 palette=COLOR_MAP, style='algorithm', dashes=False, ax=axes[0])
    axes[0].set_title('Test Accuracy vs. Round')
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Global Model Test Accuracy")
    axes[0].legend(title='Algorithm')
    axes[0].grid(True, which='both', linestyle='--')
    
    # Loss Plot
    sns.lineplot(data=df_group, x='round', y='loss', hue='algorithm',
                 palette=COLOR_MAP, style='algorithm', dashes=False, ax=axes[1])
    axes[1].set_title('Test Loss vs. Round')
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Global Model Test Loss")
    axes[1].get_legend().remove() # Remove redundant legend
    axes[1].grid(True, which='both', linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename_base}.pdf"), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename_base}.pdf")

def plot_final_accuracy_boxplots(df_summary, group, title, filename):
    """Creates a box plot comparing the final accuracy distributions of algorithms."""
    df_group = df_summary[df_summary['group'] == group]
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(data=df_group, x='algorithm', y='final_accuracy', palette=COLOR_MAP)
    sns.stripplot(data=df_group, x='algorithm', y='final_accuracy', color=".25", size=5)
    
    plt.title(title)
    plt.ylabel("Final Test Accuracy")
    plt.xlabel("Aggregation Algorithm")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def plot_ablation_study(df_summary, title, filename):
    """Plots the ablation study for temperature, now with error bands."""
    df_ablation = df_summary[df_summary['group'] == 'Ablation']
    plt.figure(figsize=(10, 6))
    
    sns.lineplot(data=df_ablation, x='temperature', y='final_accuracy', marker='o',
                 color=COLOR_MAP['FedDive (Ours)'], errorbar='sd')

    plt.title(title)
    plt.xscale('log')
    plt.xlabel("Temperature (τ) [log scale]")
    plt.ylabel("Final Test Accuracy")
    plt.grid(True, which='both', linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

def plot_weight_heatmap(raw_data, group, title, filename):
    """Plots the REAL diversity weights assigned to clients over time."""
    # Find the first successful FedDive run in the specified group
    exp_data = next((exp for exp in raw_data if exp['experiment_config']['group'] == group and 'FEDDIVE' in exp['experiment_config']['name']), None)
    if not exp_data or not exp_data['runs']:
        print(f"Skipping weight heatmap: No FedDive data found for group '{group}'.")
        return

    # Use the first valid run
    run_data = next((run for run in exp_data['runs'] if 'error' not in run), None)
    if not run_data:
        print(f"Skipping weight heatmap: No successful FedDive run found for group '{group}'.")
        return
        
    weights_history = []
    num_clients = exp_data['experiment_config']['federated']['num_clients']
    for round_data in run_data['round_history']:
        # This assumes your server logs aggregator state like this
        agg_state = round_data.get('aggregator_state', {})
        client_weights = agg_state.get('client_weights', [1.0/num_clients] * num_clients)
        weights_history.append(client_weights)
        
    if not weights_history:
        print("Skipping weight heatmap: No weight data logged.")
        return

    weights_matrix = np.array(weights_history).T # Transpose to have clients on y-axis
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(weights_matrix, cmap="viridis", cbar_kws={'label': 'Assigned Aggregation Weight'})
    
    plt.title(title)
    plt.xlabel("Communication Round")
    plt.ylabel("Client ID")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {filename}")

# --- Main Visualization Orchestrator ---
def main(results_file):
    """Load results and generate all plots and tables."""
    df_history, df_summary, raw_data = load_and_process_data(results_file)

    print("\n--- Generating Summary Tables ---")
    generate_summary_table(df_summary, "IID", "Table I: Performance on IID Data")
    generate_summary_table(df_summary, "Non-IID", "Table II: Performance on Extreme Non-IID Data (α=0.1)")
    generate_summary_table(df_summary, "Robustness", "Table III: Performance Under Adversarial Attack")
    
    print("\n--- Generating Core Paper Figures ---")
    plot_performance_curves(df_history, "IID", "Performance on IID Data", "fig_iid_performance")
    plot_performance_curves(df_history, "Non-IID", "Performance on Non-IID Data (α=0.1)", "fig_noniid_performance")
    plot_performance_curves(df_history, "Robustness", "Performance Under Adversarial Attack", "fig_robustness_performance")
    plot_ablation_study(df_summary, "FedDive Performance vs. Temperature (τ)", "fig_ablation_temperature")

    print("\n--- Generating Additional Supporting Figures ---")
    plot_final_accuracy_boxplots(df_summary, "Non-IID", "Distribution of Final Accuracies (Non-IID)", "fig_noniid_final_accuracy_boxplot")
    # This plot is highly valuable as it shows both performance and stability.
    plot_final_accuracy_boxplots(df_summary, "Robustness", "Distribution of Final Accuracies (Robustness)", "fig_robustness_final_accuracy_boxplot")
    # The weight heatmap provides crucial insight into the mechanism of FedDive.
    plot_weight_heatmap(raw_data, "Non-IID", "Dynamic Client Weights in FedDive (Non-IID)", "fig_appendix_weights_heatmap")
    
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots and tables from experiment results.")
    parser.add_argument("results_file", type=str, help="Path to the detailed experiment results JSON file.")
    args = parser.parse_args()
    main(args.results_file)