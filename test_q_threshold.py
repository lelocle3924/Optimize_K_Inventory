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

def run_q_threshold_test(input_csv: str, output_csv: str):
    """
    Chạy thử nghiệm để tìm giá trị q_threshold tối ưu.
    """
    _stdout.write(f"\nStarting q-threshold test for: '{input_csv}'\n")

    all_cases = []
    with open(input_csv, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in)
        header_in = next(reader)
        for row in reader:
            case_id = row[0]
            row_data = [float(val) for val in row[1:]]
            all_cases.append({'case_id': case_id, 'data': row_data, 'header': header_in})
    _stdout.write(f"  Loaded {len(all_cases)} cases into memory.\n")

    q_values = np.arange(0.1, 3.1, 0.1).round(1).tolist()
    _stdout.write(f"  Testing {len(q_values)} q-values from {q_values[0]} to {q_values[-1]}.\n")

    results = {case['case_id']: {} for case in all_cases}
    
    total_runs = len(q_values)
    for i, q in enumerate(q_values):
        progress = (i + 1) / total_runs
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        _stdout.write(f'\r  Running for q={q:.1f}: |{bar}| {progress:.1%}')
        _stdout.flush()

        for case in all_cases:
            case_id, row_data, header_in = case['case_id'], case['data'], case['header']
            
            maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
            major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
            
            num_minors = (len(header_in) - 5) // 4
            minors = []
            for j in range(num_minors):
                start_idx = 4 + j * 4
                min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
                minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli))
            
            system = InventorySystem(major=major, minors=minors)

            try:
                _, _, _, cost = optimize_K(system, q_threshold=q)
                results[case_id][q] = cost
            except Exception:
                results[case_id][q] = float('inf')
    _stdout.write("\n  All heuristic runs completed. Analyzing results...\n")

    summary_stats = {q: {'total_deviation': 0.0, 'max_deviation': 0.0, 'num_best': 0} for q in q_values}

    for case_id in results:
        case_results = results[case_id]
        if not all(np.isinf(v) for v in case_results.values()):
            best_cost = min(v for v in case_results.values() if not np.isinf(v))

            for q, cost in case_results.items():
                if np.isinf(cost): continue
                deviation = (cost / best_cost - 1) * 100
                
                summary_stats[q]['total_deviation'] += deviation
                if deviation > summary_stats[q]['max_deviation']:
                    summary_stats[q]['max_deviation'] = deviation
                
                if abs(cost - best_cost) < 1e-8:
                    summary_stats[q]['num_best'] += 1

    with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        header = [
            'q_threshold', 
            'num_best_found', 
            'pct_best_found (%)',
            'max_dev_from_best_heuristic (%)', 
            'avg_dev_from_best_heuristic (%)'
        ]
        writer.writerow(header)

        num_cases = len(all_cases)
        for q, stats in summary_stats.items():
            avg_dev = stats['total_deviation'] / num_cases
            pct_best = (stats['num_best'] / num_cases) * 100
            writer.writerow([
                f"{q:.1f}",
                stats['num_best'],
                f"{pct_best:.2f}",
                f"{stats['max_deviation']:.4f}",
                f"{avg_dev:.6f}"
            ])
    _stdout.write(f"-> Q-threshold test complete. Summary saved to '{output_csv}'\n")

if __name__ == '__main__':
    sys.stdout = _stdout
    start_time = datetime.now()
    
    test_files_to_run = {
        'low_cost': [f for f in os.listdir(cfg.INPUT_DIR) if 'low_cost' in f],
        'high_cost': [f for f in os.listdir(cfg.INPUT_DIR) if 'high_cost' in f]
    }
    
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    for cost_type, files in test_files_to_run.items():
        if not files:
            _stdout.write(f"No test files found for cost type: {cost_type}\n")
            continue

        input_file_path = os.path.join(cfg.INPUT_DIR, files[0])
        output_file_path = os.path.join(cfg.OUTPUT_DIR, f"q_threshold_summary_{cost_type}.csv")
        
        run_q_threshold_test(input_file_path, output_file_path)

    end_time = datetime.now()
    _stdout.write(f"\nTotal duration for all tests: {end_time - start_time}\n")