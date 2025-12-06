import pandas as pd
import matplotlib.pyplot as plt
import os
import inventory_config as cfg

def plot_single_case(ax, df: pd.DataFrame, title: str):

    best_q_row = df.sort_values(
        by=['pct_best_found (%)', 'avg_dev_from_best_heuristic (%)'],
        ascending=[False, True]
    ).iloc[0]
    best_q = best_q_row['q_threshold']

    color1 = 'tab:blue'
    ax.set_xlabel('q_threshold')
    ax.set_ylabel('Best Solution Found (%)', color=color1)
    ax.bar(df['q_threshold'], df['pct_best_found (%)'], color=color1, width=0.08, alpha=0.6, label='Best Found (%)')
    ax.tick_params(axis='y', labelcolor=color1)
    ax.set_ylim(bottom=max(0, df['pct_best_found (%)'].min() - 10))

    ax2 = ax.twinx()
    color2_avg = 'tab:red'
    color2_max = 'tab:orange'
    ax2.set_ylabel('Deviation from Best Heuristic (%)', color=color2_avg)
    ax2.plot(df['q_threshold'], df['avg_dev_from_best_heuristic (%)'], color=color2_avg, marker='.', linestyle='-', label='Avg. Deviation (%)')
    ax2.plot(df['q_threshold'], df['max_dev_from_best_heuristic (%)'], color=color2_max, linestyle='--', label='Max. Deviation (%)')
    ax2.tick_params(axis='y', labelcolor=color2_avg)

    ax.axvline(x=best_q, color='green', linestyle=':', linewidth=2, label=f'Best q ≈ {best_q}')

    ax.set_title(title, fontsize=14, weight='bold')
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

def plot_q_threshold_analysis(low_cost_csv: str, high_cost_csv: str, output_image_path: str):
   
    try:
        df_low = pd.read_csv(low_cost_csv)
        df_high = pd.read_csv(high_cost_csv)
    except FileNotFoundError as e:
        print(f"Error: Could not find result file. {e}")
        print("Please run 'test_q_threshold.py' first to generate the summary files.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Q-Threshold Performance Analysis', fontsize=18, weight='bold')

    plot_single_case(axes[0], df_low, 'Low Cost Case')
    plot_single_case(axes[1], df_high, 'High Cost Case')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_image_path, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved to '{output_image_path}'")
    plt.show()


if __name__ == '__main__':
    low_cost_file = os.path.join(cfg.OUTPUT_DIR, 'q_threshold_summary_low_cost.csv')
    high_cost_file = os.path.join(cfg.OUTPUT_DIR, 'q_threshold_summary_high_cost.csv')
    
    output_plot_file = os.path.join(cfg.OUTPUT_DIR, 'q_threshold_analysis.png')

    plot_q_threshold_analysis(
        low_cost_csv=low_cost_file,
        high_cost_csv=high_cost_file,
        output_image_path=output_plot_file
    )