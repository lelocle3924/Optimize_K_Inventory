import csv
import os
from datetime import datetime
import sys
import math
from typing import List, Tuple
import numpy as np

_stdout = sys.stdout
sys.stdout = sys.stderr

from inventory_env import InventorySystem, MajorItem, MinorItem
import inventory_config as cfg
from logger import RunLogger

logger = RunLogger(log_dir=cfg.LOG_DIR)

def optimize_K(system: InventorySystem, q_threshold: float = 0.6):
    num_items = len(system.minors)
    K = [1] * num_items
    
    current_cost, Qi_list, Q0, _, _ = compute_quotients(system, K)
    
    # --- STEP 1: Bulk Increment based on Qi ---
    while True:
        candidates_idx = [i for i, Q in enumerate(Qi_list) if Q > q_threshold]
        
        if not candidates_idx: 
            break

        K_trial = list(K)
        for idx in candidates_idx: 
            K_trial[idx] += 1
            
        new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
        
        if new_cost < current_cost - 1e-8:
            K = K_trial
            current_cost = new_cost
            Qi_list = new_Qi
            Q0 = new_Q0
        else:
            break
            
    #STEP 2
    while True:
        ranking = sorted(
            [(Q if Q >= 1 else (1.0/Q if Q > 1e-9 else 99999.9), i) for i, Q in enumerate(Qi_list)], 
            reverse=True
        )
        
        found_better = False
        
        for _, idx in ranking:
            Q_val = Qi_list[idx]
            k_curr = K[idx]
            
            direction = 0
            if Q_val > 1.0:
                direction = 1
            elif Q_val < 1.0 and k_curr > 1:
                direction = -1
            if direction == 0:
                continue
            
            while True:
                K_trial = list(K)
                K_trial[idx] += direction
                
                if K_trial[idx] < 1:
                    break
                    
                new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
                
                if new_cost < current_cost - 1e-8:
                    K = K_trial
                    current_cost = new_cost
                    Qi_list = new_Qi
                    Q0 = new_Q0
                    found_better = True
                else:
                    break
        
        if not found_better:
            break
    
    # STEP 3: Adjusting based on Q0 (Major Setup)
    while True:
        found_better = True
        direction = 0
        if Q0 > 1.0:
            direction = -1
        elif Q0 < 1.0:
            direction = 1
        
        if direction == 0:
            break 
        
        while found_better:
            found_better = False
            best_gap = 0
            k_plus = np.zeros(num_items)
            for i in range(num_items):
                if direction == -1 and K[i] <= 1:
                    continue
                K_trial = list(K)
                K_trial[i] += direction
                
                c_trial, _, _, _, _ = compute_quotients(system, K_trial)

                if current_cost - c_trial > 0:
                    found_better = True
                    if current_cost - c_trial > best_gap:
                        best_gap = current_cost - c_trial
                        k_plus = np.zeros(num_items)
                        k_plus[i] = direction
                    K_trial = list(K)
                else:
                    continue
            
            K = [a+b for a, b in zip(K, k_plus)]
            current_cost, _, _, _, _ = compute_quotients(system, K)

            if not found_better:
                break
        
        if not found_better:
            break

    # STEP 4 Final calculation
    cost, _, _, T, F = compute_quotients(system, K)
    return K, T, F, cost

def heuristic_N(system: InventorySystem, q_threshold: float = 0.6):
    num_items = len(system.minors)
    K = [1] * num_items
    
    current_cost, Qi_list, Q0, _, _ = compute_quotients(system, K)
    
    while True:
        candidates_idx = [i for i, Q in enumerate(Qi_list) if Q > q_threshold]
        
        if not candidates_idx: 
            break

        K_trial = list(K)
        for idx in candidates_idx: 
            K_trial[idx] += 1
            
        new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
        
        if new_cost < current_cost - 1e-8:
            K = K_trial
            current_cost = new_cost
            Qi_list = new_Qi
            Q0 = new_Q0
        else:
            break
            
    #STEP 2
    while True:
        ranking = sorted(
            [(Q if Q >= 1 else (1.0/Q if Q > 1e-9 else 99999.9), i) for i, Q in enumerate(Qi_list)], 
            reverse=True
        )
        
        found_better = False
        
        for _, idx in ranking:
            Q_val = Qi_list[idx]
            k_curr = K[idx]
            
            direction = 0
            if Q_val > 1.0:
                direction = 1
            elif Q_val < 1.0 and k_curr > 1:
                direction = -1
            if direction == 0:
                continue
            
            while True:
                K_trial = list(K) 
                K_trial[idx] += direction
                
                if K_trial[idx] < 1:
                    break
                    
                new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
                
                if new_cost < current_cost - 1e-8:
                    K = K_trial
                    current_cost = new_cost
                    Qi_list = new_Qi
                    Q0 = new_Q0
                    found_better = True
                else:
                    break
        
        if not found_better:
            break
    
    # HEURISTIC N DOESN'T HAVE STEP 3 OF OPTIMIZING K
    # STEP 3 Final calculation
    cost, _, _, T, F = compute_quotients(system, K)
    return K, T, F, cost

def compute_quotients(system: InventorySystem, K: List[int]) -> Tuple[float, List[float], float, float, float]:
    try:
        T_star, F_star, cost = system.optimal_T_F_for_K(K)
    except Exception:
        return float('inf'), [], 0.0, 0.0, 0.0

    if T_star <= 1e-9:
        return float('inf'), [], 0.0, 0.0, 0.0

    major, minors, D = system.major, system.minors, system.major.D
    
    # Calculate Q0
    # Q0 = 2A / [D * T^2 * (Ch*F^2 + Cb*(1-F)^2)]
    denom0_term = (major.C_h * (F_star**2) + major.C_b * ((1.0 - F_star)**2))
    denom0 = D * (T_star**2) * denom0_term
    
    if denom0 > 1e-9:
        Q0 = 2.0 * major.A / denom0
    else:
        Q0 = 9999.9

    Qi_list = []
    for k_i, m, d_i in zip(K, minors, system.d_list):
        # Qi = 2Ai / [ki * D * T^2 * (ell * Chi * F^2 + (ki*di - ell)*Chi)]
        term = m.ell_i * m.C_hi * (F_star**2) + (k_i * d_i - m.ell_i) * m.C_hi
        denom_i = k_i * D * (T_star**2) * term
        
        if denom_i > 1e-9:
            Qi = 2.0 * m.A_i / denom_i
        else:
            Qi = 9999.9
            
        Qi_list.append(Qi)
        
    return cost, Qi_list, Q0, T_star, F_star

def process_file(input_csv: str, output_csv: str, verbose = False):
    logger.log(f"\nProcessing batch file: '{input_csv}'...\n")

    if 'low' in input_csv:
        selected_q_threshold = cfg.q_threshold_low
        _stdout.write(f"  Detected 'low_cost' case. Using q_threshold = {selected_q_threshold}\n")
    elif 'high' in input_csv:
        selected_q_threshold = cfg.q_threshold_high
        _stdout.write(f"  Detected 'high_cost' case. Using q_threshold = {selected_q_threshold}\n")
    else:
        selected_q_threshold = cfg.q_threshold_low
        _stdout.write(f"  Warning: Cost case not detected in filename. Defaulting to q_threshold = {selected_q_threshold}\n")
    
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        header_in = next(reader)
        num_minors = (len(header_in) - 5) // 4
        
        output_header = ['case_id', 'major_A', 'major_Ch', 'major_Cb', 'K_star', 'T_star', 'F_star', 'G_star', 'warning_negative_demand']
        writer.writerow(output_header)
        
        total_rows = sum(1 for row in open(input_csv)) - 1
        f_in.seek(0)
        next(reader)

        for i, row in enumerate(reader):
            if (i+1)%4 ==0 and verbose == True:
                progress = (i + 1) / total_rows
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                logger.log(f'  Running cases: |{bar}| {progress:.1%} ({i+1}/{total_rows})')

            case_id = row[0]
            try:
                row_data = [float(val) for val in row[1:]]
            except ValueError:
                print(f"Skipping malformed row {i}")
                continue
            
            maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
            major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
            
            minors = []
            for j in range(num_minors):
                start_idx = 4 + j * 4
                min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
                minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli))
            
            system = InventorySystem(major=major, minors=minors)
            
            # --- CALL OPTIMIZER ---
            try:
                if cfg.USE_HEURISTIC_N == True:
                    K_star, T_star, F_star, G_star = heuristic_N(system, q_threshold=selected_q_threshold)
                else:
                    K_star, T_star, F_star, G_star = optimize_K(system, q_threshold=selected_q_threshold)
            except Exception as e:
                print(f"Error optimizing Case {case_id}: {e}")
                K_star, T_star, F_star, G_star = "ERROR", -1.0, -1.0, -1.0

            if cfg.PLOT and cfg.BATCH==False:
                logger.log("\n==== DRAWING INVENTORY FIGURE ====")
                system.plot_figure(K=K_star, T=T_star, F=F_star, case_id=case_id)

            negative_demand_items = [f'M{j+1}' for j, m in enumerate(minors) if (m.D_i - m.ell_i * major.D) < 0]
            warning_msg = f"Negative demand: {','.join(negative_demand_items)}" if negative_demand_items else "None"
            
            if isinstance(G_star, str):
                row_T = "ERROR"
                row_F = "ERROR"
                row_G = "ERROR"
            else:
                row_T = f"{T_star:.6f}"
                row_F = f"{F_star:.6f}"
                row_G = f"{G_star:.4f}"

            output_row = [case_id, maj_A, maj_Ch, maj_Cb, str(K_star), row_T, row_F, row_G, warning_msg]
            writer.writerow(output_row)

    if cfg.BATCH:
        logger.log(f"\nBatch processing complete. Results saved to '{output_csv}'\n")
    else:
        logger.log(f"\nSingle file processing complete. Results saved to '{output_csv}'\n")

if __name__ == '__main__':
    sys.stdout = _stdout

    if cfg.BATCH:
        if not os.path.exists(cfg.INPUT_DIR):
             logger.log(f"Error: INPUT_DIR '{cfg.INPUT_DIR}' does not exist.")
             exit()
        input_files = [f for f in os.listdir(cfg.INPUT_DIR) if f.startswith('N') and f.endswith('.csv')]
    else:
        input_files = [cfg.single_file]
        
    if not input_files:
        logger.log("No 'test_cases_*.csv' files found. Please run 'generate_test_cases.py' first.")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        start_time = datetime.now()
        logger.log(f"Starting batch process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if cfg.USE_HEURISTIC_N == True:
            logger.log("Using heuristic N")
        else:
            logger.log("Using optimizer K")

        if cfg.USE_HEURISTIC_N == True:
            for input_file in sorted(input_files):
                output_file = os.path.join(cfg.OUTPUT_DIR, f"results_N_{os.path.basename(input_file)}")
                input_file = os.path.join(cfg.INPUT_DIR, input_file)
                process_file(input_file, output_file, verbose=cfg.OPTIMIZE_VERBOSE)
        else:
            for input_file in sorted(input_files):
                output_file = os.path.join(cfg.OUTPUT_DIR, f"results_K_{os.path.basename(input_file)}")
                input_file = os.path.join(cfg.INPUT_DIR, input_file)
                process_file(input_file, output_file, verbose=cfg.OPTIMIZE_VERBOSE)
            
        end_time = datetime.now()
        logger.log(f"\nAll batches finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Total duration: {end_time - start_time}")
        logger.save()