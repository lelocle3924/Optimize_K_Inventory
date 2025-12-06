import csv
import os
from datetime import datetime
import sys
import matplotlib.pyplot as plt

_stdout = sys.stdout
sys.stdout = sys.stderr

from inventory_env import InventorySystem, MajorItem, MinorItem
from optimize_k_batch import optimize_K
import inventory_config as cfg

def run_sensitivity_analysis(input_csv_path: str, output_dir: str):
    _stdout.write(f"\n--- Starting Sensitivity Analysis for N ---\n")
    _stdout.write(f"Base case file: '{os.path.basename(input_csv_path)}'\n")

    with open(input_csv_path, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        header_in = next(reader)
        try:
            base_row = next(reader)
        except StopIteration:
            _stdout.write("Error: The selected CSV file is empty or has no data rows.\n")
            return
            
    row_data = [float(val) for val in base_row[1:]]
    maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
    base_major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
    
    num_minors_base = (len(header_in) - 5) // 4
    base_minors = []
    for j in range(num_minors_base):
        start_idx = 4 + j * 4
        min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
        base_minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli, name=f"M{j+1}"))
    
    _stdout.write(f"Base case loaded with {num_minors_base} minor items.\n")

    if 'low_cost' in input_csv_path:
        q_thresh = cfg.q_threshold_low
    else:
        q_thresh = cfg.q_threshold_high

    results = []
    
    for n in range(1, num_minors_base + 1):
        _stdout.write(f"\r  Running for N = {n}/{num_minors_base}...")
        _stdout.flush()

        current_minors = base_minors[:n]
        system = InventorySystem(major=base_major, minors=current_minors)

        try:
            K_star, T_star, F_star, G_star = optimize_K(system, q_threshold=q_thresh)
            results.append({
                'N': n,
                'G_star': G_star,
                'T_star': T_star,
                'F_star': F_star,
                'K_star': str(K_star)
            })
        except Exception as e:
            _stdout.write(f"\nError at N={n}: {e}\n")
            continue
    _stdout.write("\nAnalysis complete. Generating report and plots...\n")

    output_csv_path = os.path.join(output_dir, f"sensitivity_N_{os.path.basename(input_csv_path)}")
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=['N', 'G_star', 'T_star', 'F_star', 'K_star'])
        writer.writeheader()
        writer.writerows(results)
    _stdout.write(f"-> Report saved to '{output_csv_path}'\n")

    n_values = [r['N'] for r in results]
    g_values = [r['G_star'] for r in results]
    f_values = [r['F_star'] for r in results]
    t_values = [r['T_star'] for r in results]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f'Sensitivity Analysis of N (Number of Minor Items)\nBase Case: {os.path.basename(input_csv_path)}', fontsize=16)

    ax1.plot(n_values, g_values, marker='o', linestyle='-', color='tab:red')
    ax1.set_ylabel("Total Cost (G*)")
    ax1.set_title("Effect on Total Cost")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(n_values, f_values, marker='o', linestyle='-', color='tab:blue')
    ax2.set_ylabel("Fill Rate (F*)")
    ax2.set_title("Effect on Optimal Fill Rate")
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax3.plot(n_values, t_values, marker='o', linestyle='-', color='tab:green')
    ax3.set_ylabel("Order Cycle (T*)")
    ax3.set_title("Effect on Optimal Order Cycle")
    ax3.grid(True, linestyle='--', alpha=0.6)

    plt.xlabel("Number of Minor Items (N)")
    plt.xticks(n_values) 
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    plot_path = os.path.join(output_dir, f"sensitivity_N_plot_{os.path.basename(input_csv_path).replace('.csv', '.png')}")
    plt.savefig(plot_path, dpi=150)
    _stdout.write(f"-> Plot saved to '{plot_path}'\n")
    
    plt.show()

if __name__ == '__main__':
    sys.stdout = _stdout

    base_file_name = 'test_cases_N25_low_cost.csv'
    
    input_file = os.path.join(cfg.INPUT_DIR, base_file_name)
    output_directory = "sensitivity_results"
    
    if not os.path.exists(input_file):
        _stdout.write(f"\nError: Base case file not found at '{input_file}'")
        _stdout.write(f"\nPlease ensure the file exists and cfg.INPUT_DIR is set correctly in inventory_config.py\n")
    else:
        os.makedirs(output_directory, exist_ok=True)
        run_sensitivity_analysis(input_file, output_directory)