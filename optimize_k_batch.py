# optimize_k_batchprocess.py (Version 2)
import csv
import os
from datetime import datetime
import sys
from typing import List, Tuple


# Chuyển hướng stdout sang stderr để không ảnh hưởng progress bar
_stdout = sys.stdout
sys.stdout = sys.stderr

# Import các lớp và hàm
from inventory_env import InventorySystem, MajorItem, MinorItem
import inventory_config as cfg
from logger import RunLogger

logger = RunLogger(log_dir=cfg.LOG_DIR)

def optimize_K(system: InventorySystem, q_threshold: float = 0.6):
    num_items = len(system.minors)
    K = [1] * num_items 
    current_cost, Qi_list, Q0, _, _ = compute_quotients(system, K)
    
    # Step 1
    while True:
        candidates_idx = [i for i, Q in enumerate(Qi_list) if Q > q_threshold]
        if not candidates_idx: break
        K_trial = list(K)
        for idx in candidates_idx: K_trial[idx] += 1
        new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
        if new_cost < current_cost:
            K, current_cost, Qi_list, Q0 = K_trial, new_cost, new_Qi, new_Q0
        else:
            break
            
    # Step 2
    while True:
        ranking = sorted([(Q if Q >= 1 else (1.0/Q if Q > 0 else 9999.9), i) for i, Q in enumerate(Qi_list)], reverse=True)
        found_better = False
        for _, idx in ranking:
            Q_val, k_curr = Qi_list[idx], K[idx]
            direction = 1 if Q_val > 1 else -1
            if direction == -1 and k_curr <= 1: continue
            K_trial = list(K); K_trial[idx] += direction
            new_cost, new_Qi, new_Q0, _, _ = compute_quotients(system, K_trial)
            if new_cost < current_cost - 1e-8:
                K, current_cost, Qi_list, Q0 = K_trial, new_cost, new_Qi, new_Q0
                found_better = True
                break
        if not found_better: break

    # Step 3
    while True:
        best_K_step3, best_cost_step3 = None, current_cost
        if Q0 > 1.0:
            for i in range(num_items):
                if K[i] > 1:
                    K_trial = list(K); K_trial[i] -= 1
                    c_trial, _, _, _, _ = compute_quotients(system, K_trial)
                    if c_trial < best_cost_step3 - 1e-8:
                        best_cost_step3, best_K_step3 = c_trial, K_trial
        elif Q0 < 1.0:
            for i in range(num_items):
                K_trial = list(K); K_trial[i] += 1
                c_trial, _, _, _, _ = compute_quotients(system, K_trial)
                if c_trial < best_cost_step3 - 1e-8:
                     best_cost_step3, best_K_step3 = c_trial, K_trial
        if best_K_step3:
            K, current_cost = best_K_step3, best_cost_step3
            _, Qi_list, Q0, _, _ = compute_quotients(system, K)
        else:
            break

    cost, _, _, T, F = compute_quotients(system, K)
    return K, T, F, cost

def compute_quotients(system: InventorySystem, K: List[int]) -> Tuple[float, List[float], float, float, float]:
    T_star, F_star, cost = system.optimal_T_F_for_K(K)
    major, minors, D = system.major, system.minors, system.major.D
    denom0 = D * (T_star**2) * (major.C_h * (F_star**2) + major.C_b * ((1.0 - F_star)**2))
    Q0 = (2.0 * major.A / denom0) if denom0 > 0 else 9999.9
    Qi_list = []
    for k_i, m, d_i in zip(K, minors, system.d_list):
        term = m.ell_i * m.C_hi * (F_star**2) + (k_i * d_i - m.ell_i) * m.C_hi
        denom_i = k_i * D * (T_star**2) * term
        Qi = (2.0 * m.A_i / denom_i) if denom_i > 0 else 9999.9
        Qi_list.append(Qi)
    return cost, Qi_list, Q0, T_star, F_star
# ------------------------------------------------------------------

def process_file(input_csv: str, output_csv: str):
    """
    Đọc file CSV đầu vào, chạy tối ưu, và ghi kết quả tóm tắt ra file CSV đầu ra.
    """
    logger.log(f"\nProcessing batch file: '{input_csv}'...\n")

    if 'low_cost' in input_csv:
        selected_q_threshold = cfg.q_threshold_low
        _stdout.write(f"  Detected 'low_cost' case. Using q_threshold = {selected_q_threshold}\n")
    elif 'high_cost' in input_csv:
        selected_q_threshold = cfg.q_threshold_high
        _stdout.write(f"  Detected 'high_cost' case. Using q_threshold = {selected_q_threshold}\n")
    else:
        # Fallback nếu không xác định được, dùng giá trị low làm mặc định
        selected_q_threshold = cfg.q_threshold_low
        _stdout.write(f"  Warning: Cost case not detected in filename. Defaulting to q_threshold = {selected_q_threshold}\n")
    
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        # --- Đọc header và tạo header cho file output tóm tắt ---
        header_in = next(reader)
        num_minors = (len(header_in) - 5) // 4 # -5 vì có thêm case_id
        
        output_header = ['case_id', 'major_A', 'major_Ch', 'major_Cb', 'K_star', 'T_star', 'F_star', 'G_star', 'warning_negative_demand']
        writer.writerow(output_header)
        
        # Đếm số dòng để hiển thị progress bar
        # Lưu ý: việc này đọc lại file, có thể làm chậm nếu file rất lớn.
        # Với vài nghìn dòng thì không đáng kể.
        total_rows = sum(1 for row in open(input_csv)) - 1
        
        f_in.seek(0) # Quay lại đầu file
        next(reader) # Bỏ qua header một lần nữa

        for i, row in enumerate(reader):
            # --- Progress bar ---
            if (i+1)%4 ==0:
                progress = (i + 1) / total_rows
                bar_length = 40
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                logger.log(f'  Running cases: |{bar}| {progress:.1%} ({i+1}/{total_rows})')

            # --- Parse dữ liệu từ dòng CSV ---
            case_id = row[0]
            row_data = [float(val) for val in row[1:]]
            
            maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
            major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
            
            minors = []
            for j in range(num_minors):
                start_idx = 4 + j * 4
                min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
                minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli))
            
            system = InventorySystem(major=major, minors=minors)
            
            try:
                K_star, T_star, F_star, G_star = optimize_K(system, q_threshold=selected_q_threshold)
            except Exception as e:
                K_star, T_star, F_star, G_star = "ERROR", "ERROR", "ERROR", "ERROR"

            if cfg.PLOT and cfg.BATCH==False:
                logger.log("\n==== DRAWING INVENTORY FIGURE ====")
                system.plot_figure1_like(K=K_star, T=T_star, F=F_star, n_super_cycles=cfg.PLOT_OPTIONS["n_super_cycles"], case_id=case_id)
            else:
                pass

            negative_demand_items = [f'M{j+1}' for j, m in enumerate(minors) if (m.D_i - m.ell_i * major.D) < 0]
            warning_msg = f"Negative demand: {','.join(negative_demand_items)}" if negative_demand_items else "None"
            
            output_row = [case_id, maj_A, maj_Ch, maj_Cb, str(K_star), f"{T_star:.6f}", f"{F_star:.6f}", f"{G_star:.4f}", warning_msg]
            writer.writerow(output_row)

    if cfg.BATCH:
        logger.log(f"\n-> Batch processing complete. Results saved to '{output_csv}'\n")
    else:
        logger.log(f"\n-> Single file processing complete. Results saved to '{output_csv}'\n")

if __name__ == '__main__':
    sys.stdout = _stdout

    if cfg.BATCH:
        input_files = [f for f in os.listdir(cfg.INPUT_DIR) if f.startswith('test_cases_') and f.endswith('.csv')]
    else:
        input_files = [cfg.single_file]
    if not input_files:
        logger.log("No 'test_cases_*.csv' files found. Please run 'generate_test_cases.py' first.")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        start_time = datetime.now()
        logger.log(f"Starting batch process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        for input_file in sorted(input_files):
            output_file = os.path.join(cfg.OUTPUT_DIR, f"results_{os.path.basename(input_file)}")
            input_file = os.path.join(cfg.INPUT_DIR, input_file)
            process_file(input_file, output_file)
            
        end_time = datetime.now()
        logger.log(f"\nAll batches finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Total duration: {end_time - start_time}")
        logger.save()
    