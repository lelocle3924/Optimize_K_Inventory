import sys
import csv
import os
import math
from typing import List, Tuple
from datetime import datetime

_stdout = sys.stdout
sys.stdout = sys.stderr

try:
    from gekko import GEKKO
except ImportError:
    sys.stdout = _stdout
    print("Error: GEKKO not found. Please install via 'pip install gekko'")
    sys.exit(1)

from inventory_env import InventorySystem, MajorItem, MinorItem
import inventory_config as cfg
from logger import RunLogger

logger = RunLogger(log_dir=cfg.LOG_DIR)

def solve_with_gekko(system: InventorySystem) -> Tuple[List[int], float, float, float]:

    D = system.major.D
    C_h = system.major.C_h
    C_b = system.major.C_b
    sum_lambda_Chi = sum(m.ell_i * m.C_hi for m in system.minors)
    
    G1 = (C_h + C_b + sum_lambda_Chi) * D / 2.0
    G2 = C_b * D
    
    if G1 > 1e-9:
        F_star = G2 / (2.0 * G1)
    else:
        F_star = 1.0
    F_star = max(0.0, min(1.0, F_star))
    

    G3_base = (C_b * D / 2.0) - (sum_lambda_Chi * D / 2.0)
    Base_Beta = G1 * (F_star**2) - G2 * F_star + G3_base
    
    H_i_list = [(system.d_list[i] * m.C_hi * D / 2.0) for i, m in enumerate(system.minors)]
    
    m = GEKKO(remote=False) 
    m.options.SOLVER = 1 
    T = m.Var(value=0.1, lb=1e-5, ub=100.0, name='T') 
    
    K_vars = []
    for i in range(len(system.minors)):
        ki = m.Var(value=1, lb=1, ub=200, integer=True, name=f'k_{i}')
        K_vars.append(ki)
            
    sum_Ai_ki = m.Intermediate(sum([system.minors[i].A_i / K_vars[i] for i in range(len(system.minors))]))
    Alpha = m.Intermediate(system.major.A + sum_Ai_ki)
    
    sum_ki_Hi = m.Intermediate(sum([K_vars[i] * H_i_list[i] for i in range(len(system.minors))]))
    Beta = m.Intermediate(Base_Beta + sum_ki_Hi)
    
    m.Obj( (Alpha / T) + (Beta * T) )
    
    try:
        m.solve(disp=False)

        if m.options.APPSTATUS == 1:
            T_val = T.value[0]
            K_vals = [int(round(k.value[0])) for k in K_vars]
            Cost_val = m.options.OBJFCNVAL
            return K_vals, T_val, F_star, Cost_val
        else:
            return "INFEASIBLE", -1, -1, -1
            
    except Exception as e:
        return f"ERROR: {str(e)}", -1, -1, -1

def process_file_gekko(input_csv: str, output_csv: str, verbose = False):
    logger.log(f"\nProcessing batch file with GEKKO: '{input_csv}'...\n")
    
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        
        header_in = next(reader)
        num_minors = (len(header_in) - 5) // 4
        
        output_header = ['case_id', 'major_A', 'major_Ch', 'major_Cb', 'K_star', 'T_star', 'F_star', 'G_star']
        writer.writerow(output_header)
        
        total_rows = sum(1 for row in open(input_csv)) - 1
        f_in.seek(0)
        next(reader)

        for i, row in enumerate(reader):
            if verbose or (i+1) % 10 == 0:
                progress = (i + 1) / total_rows
                bar_length = 30
                filled_length = int(bar_length * progress)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                sys.stdout = _stdout
                print(f'\r  GEKKO Running: |{bar}| {progress:.1%} ({i+1}/{total_rows})', end='')
                sys.stdout = sys.stderr

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
            
            K_star, T_star, F_star, G_star = solve_with_gekko(system)
            
            if isinstance(K_star, str):
                output_row = [case_id, maj_A, maj_Ch, maj_Cb, K_star, str(T_star), str(F_star), str(G_star)]
            else:
                output_row = [
                    case_id, maj_A, maj_Ch, maj_Cb, 
                    str(K_star), 
                    f"{T_star:.6f}", 
                    f"{F_star:.6f}", 
                    f"{G_star:.6f}"
                ]
            writer.writerow(output_row)
            
    print("\n")

if __name__ == '__main__':
    sys.stdout = _stdout

    if cfg.BATCH:
        input_files = [f for f in os.listdir(cfg.INPUT_DIR) if f.startswith('N') and f.endswith('.csv')]
    else:
        input_files = [cfg.single_file]

    if not input_files:
        logger.log("No input files found.")
    else:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        target_file = input_files 
        start_time = datetime.now()
        
        logger.log(f"Starting GEKKO solver on: {target_file}")
        
        for input_file in sorted(input_files):
            output_file = os.path.join(cfg.OUTPUT_DIR, f"results_gekko_{os.path.basename(input_file)}")
            input_file = os.path.join(cfg.INPUT_DIR, input_file)
            process_file_gekko(input_file, output_file, verbose=cfg.OPTIMIZE_VERBOSE)
        
        
        end_time = datetime.now()
        
        logger.log(f"GEKKO finished. Duration: {end_time - start_time}")
        logger.save()