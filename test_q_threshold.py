import csv
import os
from datetime import datetime
import sys
import numpy as np

_stdout = sys.stdout
sys.stdout = sys.stderr

from inventory_env import InventorySystem, MajorItem, MinorItem
from optimize_k_batch import optimize_K
import inventory_config as cfg

def load_exact_solutions(gekko_dir: str, test_case_filename: str) -> dict:
    exact_filename = f"results_gekko_{test_case_filename}"
    exact_filepath = os.path.join(gekko_dir, exact_filename)
    
    if not os.path.exists(exact_filepath):
        _stdout.write(f"\n  Warning: Exact solution file not found: {exact_filename}")
        return {}

    solutions = {}
    with open(exact_filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get('case_id')
            if not case_id: continue
            try:
                cost = float(row['G_star'])
                solutions[case_id] = cost
            except (ValueError, KeyError):
                solutions[case_id] = float('inf')
    return solutions

def run_aggregated_q_threshold_test(input_filenames: list, cost_type: str, output_csv_path: str):
    _stdout.write(f"\n--- Starting AGGREGATED Q-Threshold Test for '{cost_type}' ---\n")

    all_cases = []
    exact_solutions = {}
    
    for filename in input_filenames:
        _stdout.write(f"  Loading data from '{filename}'...\n")
        current_exact_solutions = load_exact_solutions(cfg.GEKKO_RESULTS_DIR, filename)
        exact_solutions.update(current_exact_solutions)

        input_filepath = os.path.join(INPUT_DIR, filename)
        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            reader = csv.reader(f_in)
            header_in = next(reader)
            for row in reader:
                all_cases.append({'case_id': row[0], 'data': row[1:], 'header': header_in})

    valid_cases = [
        case for case in all_cases 
        if case['case_id'] in exact_solutions and not np.isinf(exact_solutions[case['case_id']])
    ]
    
    _stdout.write(f"\n  Finished loading. Total valid cases to process for '{cost_type}': {len(valid_cases)}\n")
    if not valid_cases:
        _stdout.write("  No valid cases to process. Exiting.\n")
        return

    q_values = np.arange(0.1, 3.1, 0.1).round(1).tolist()
    
    summary_stats = {q: {'total_deviation': 0.0, 'max_deviation': 0.0, 'num_optimum': 0} for q in q_values}

    total_runs = len(valid_cases)
    for i, case in enumerate(valid_cases):
        progress = (i + 1) / total_runs
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        _stdout.write(f'\r  Processing cases: |{bar}| {progress:.1%} ({i+1}/{total_runs})')
        _stdout.flush()

        case_id = case['case_id']
        row_data = [float(val) for val in case['data']]
        header_in = case['header']
        exact_cost = exact_solutions[case_id]
        
        maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
        major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
        num_minors = (len(header_in) - 5) // 4
        minors = []
        for j in range(num_minors):
            start_idx = 4 + j * 4
            min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
            minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli))
        
        system = InventorySystem(major=major, minors=minors)

        for q in q_values:
            try:
                _, _, _, heuristic_cost = optimize_K(system, q_threshold=q)
                deviation = (heuristic_cost / exact_cost - 1) * 100
                
                summary_stats[q]['total_deviation'] += deviation
                if deviation > summary_stats[q]['max_deviation']:
                    summary_stats[q]['max_deviation'] = deviation
                
                if abs(heuristic_cost - exact_cost) <= 1e-3:
                    summary_stats[q]['num_optimum'] += 1
            except Exception:
                summary_stats[q]['total_deviation'] += float('inf')

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        header = ['q', 'No. of optimum', 'Total', 'Optimality (%)', 'Max. dev. (%)', 'Avg. dev. (%)']
        writer.writerow(header)
        
        total_cases = len(valid_cases)
        for q, stats in summary_stats.items():
            num_optimum = stats['num_optimum']
            optimality_pct = (num_optimum / total_cases) * 100
            avg_dev = stats['total_deviation'] / total_cases if total_cases > 0 else 0
            
            writer.writerow([
                f"{q:.1f}",
                num_optimum,
                total_cases,
                f"{optimality_pct:.2f}",
                f"{stats['max_deviation']:.4f}",
                f"{avg_dev:.6f}"
            ])
    _stdout.write(f"-> Aggregated test summary for '{cost_type}' saved to '{output_csv_path}'\n")


if __name__ == '__main__':
    sys.stdout = _stdout
    start_time = datetime.now()
    INPUT_DIR = "generate/generated_test_cases"
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    try:
        all_test_files = [f for f in os.listdir(INPUT_DIR) if f.startswith('N') and f.endswith('.csv')]
    except FileNotFoundError:
        _stdout.write(f"\nERROR: Input directory not found at 'generate/generated_test_cases'\n")
        all_test_files = []

    if not all_test_files:
        _stdout.write("No test case files found. Please generate them first.\n")
    else:
        test_files_by_type = {
            'low': sorted([f for f in all_test_files if 'low' in f]),
            'high': sorted([f for f in all_test_files if 'high' in f])
        }
        
        for cost_type, files in test_files_by_type.items():
            if not files:
                _stdout.write(f"No files found for cost type: {cost_type}\n")
                continue
            
            output_file_path = os.path.join("sensitivity_results", f"q_threshold_vs_exact_AGGREGATED_{cost_type}.csv")
            
            run_aggregated_q_threshold_test(files, cost_type, output_file_path)

    end_time = datetime.now()
    _stdout.write(f"\nTotal duration for all aggregated tests: {end_time - start_time}\n")