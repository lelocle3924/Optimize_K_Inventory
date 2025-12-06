import csv
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import pyomo
print(f"Đang chạy bằng Python tại: {sys.executable}")
print(f"Sử dụng Pyomo phiên bản: {pyomo.__version__}")
print("-" * 50)

# Import cấu hình
import inventory_config as cfg

# Chuyển hướng stdout sang stderr để không ảnh hưởng progress bar
_stdout = sys.stdout
sys.stdout = sys.stderr

from pyomo.environ import ConcreteModel, Var, Objective, NonNegativeReals, PositiveIntegers, SolverFactory, RangeSet, value, minimize
pyomo_available = True

# --- 2. CÁC LỚP DỮ LIỆU ---
@dataclass
class MajorItem:
    D: float; A: float; C_h: float; C_b: float

@dataclass
class MinorItem:
    D_i: float; A_i: float; C_hi: float; ell_i: float

@dataclass
class InventorySystem:
    major: MajorItem
    minors: List[MinorItem]
    d_list: List[float] = field(init=False)

    def __post_init__(self):
        D = self.major.D
        if D > 1e-9:
            self.d_list = [m.D_i / D for m in self.minors]
        else:
            self.d_list = [0.0] * len(self.minors)

# --- 3. BỘ GIẢI CHÍNH XÁC SỬ DỤNG PYOMO ---
def solve_with_pyomo(system: InventorySystem, k_max: int, solver_name: str) -> Tuple:
    """
    Giải bài toán JRP bằng Pyomo. Phiên bản này tương thích rộng rãi.
    """
    if not pyomo_available:
        return "PYOMO_NOT_INSTALLED", None, None, None, None

    model = ConcreteModel("JRP_MINLP")
    num_minors = len(system.minors)
    if num_minors == 0:
        return "NO_MINORS", None, None, None, None

    # --- Định nghĩa Model (biến, hàm mục tiêu) ---
    model.M = RangeSet(1, num_minors)
    model.T = Var(domain=NonNegativeReals, bounds=(1e-6, None), initialize=0.1)
    model.F = Var(domain=NonNegativeReals, bounds=(0.0, 1.0), initialize=0.8)
    model.K = Var(model.M, domain=PositiveIntegers, bounds=(1, k_max), initialize=1)
    major = system.major
    G0_expr = major.A + sum(system.minors[i-1].A_i / model.K[i] for i in model.M)
    sum_ellC = sum(m.ell_i * m.C_hi for m in system.minors)
    u_G1 = (major.D / 2.0) * (major.C_h + sum_ellC)
    u_G2 = (major.D / 2.0) * major.C_b
    sum_term_G3 = sum((model.K[i] * system.d_list[i-1] - system.minors[i-1].ell_i) * system.minors[i-1].C_hi for i in model.M)
    u_G3_expr = (major.D / 2.0) * sum_term_G3
    uF_expr = u_G1 * model.F**2 + u_G2 * (1 - model.F)**2 + u_G3_expr
    total_cost_expr = G0_expr / model.T + model.T * uF_expr
    model.total_cost = Objective(expr=total_cost_expr, sense=minimize)

    # --- Giải bài toán ---
    try:
        solver = SolverFactory(f"{solver_name}:asl")
        # Sử dụng 'solver_io' để đảm bảo giao tiếp qua định dạng file .nl,
        # cách này rất đáng tin cậy cho các bài toán phi tuyến.
        results = solver.solve(model, tee=False)
    except Exception as e:
        return f"SOLVER_ERROR: {e}", None, None, None, None

    status = str(results.solver.termination_condition)
    if status == "optimal" or status == "locallyOptimal":
        K_star = [int(round(value(model.K[i]))) for i in model.M]
        T_star = value(model.T)
        F_star = value(model.F)
        cost_star = value(model.total_cost)
        return "optimal", K_star, T_star, F_star, cost_star
    else:
        return status, None, None, None, None

# --- 4. HÀM XỬ LÝ BATCH (Giữ nguyên) ---
def process_batch_file_exact(input_csv: str, output_csv: str, k_max: int, solver_name: str):
    _stdout.write(f"\nProcessing batch file with Exact Solver: '{input_csv}'...\n")
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.reader(f_in); writer = csv.writer(f_out)
        header_in = next(reader)
        num_minors = (len(header_in) - 5) // 4
        output_header = ['case_id', 'K_star', 'T_star', 'F_star', 'G_star', 'solver_status', 'time_s']
        writer.writerow(output_header)
        
        all_rows = list(reader)
        total_rows = len(all_rows)

        for i, row in enumerate(all_rows):
            progress = (i + 1) / total_rows
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            _stdout.write(f'  Running cases: |{bar}| {progress:.1%} ({i+1}/{total_rows})')
            _stdout.flush()

            case_id = row[0]
            try:
                row_data = [float(val) for val in row[1:]]
                maj_D, maj_A, maj_Ch, maj_Cb = row_data[0:4]
                major = MajorItem(D=maj_D, A=maj_A, C_h=maj_Ch, C_b=maj_Cb)
                minors = []
                for j in range(num_minors):
                    start_idx = 4 + j * 4
                    min_Di, min_Ai, min_Chi, min_elli = row_data[start_idx : start_idx + 4]
                    minors.append(MinorItem(D_i=min_Di, A_i=min_Ai, C_hi=min_Chi, ell_i=min_elli))
                system = InventorySystem(major=major, minors=minors)
                
                start_time = time.perf_counter()
                status, K, T, F, cost = solve_with_pyomo(system, k_max, solver_name)
                exec_time = time.perf_counter() - start_time
                
                if K is not None:
                    output_row = [case_id, str(K), f"{T:.6f}", f"{F:.6f}", f"{cost:.4f}", status, f"{exec_time:.4f}"]
                else:
                    output_row = [case_id, "N/A", "N/A", "N/A", "N/A", status, f"{exec_time:.4f}"]
            except Exception as e:
                output_row = [case_id, "CODE_ERROR", "ERROR", "ERROR", "ERROR", str(e), 0.0]
            writer.writerow(output_row)
    _stdout.write(f"\n-> Exact solver batch processing complete. Results saved to '{output_csv}'\n")

if __name__ == '__main__':
    sys.stdout = _stdout
    
    if pyomo_available:
        if cfg.RUN_EXACT_SOLVER_BATCH:
            input_files = [f for f in os.listdir(cfg.INPUT_DIR) if f.startswith('test_cases_') and f.endswith('.csv')]
        else:
            input_files = [cfg.single_file]
        if not input_files:
            logger.log("No 'test_cases_*.csv' files found. Please run 'generate_test_cases.py' first.")
        else:
            os.makedirs(cfg.EXACT_OUTPUT_DIR, exist_ok=True)
                       
            start_time = datetime.now()
            print(f"Starting Exact Solver Batch Process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Solver: {cfg.EXACT_SOLVER_NAME}, K_max: {cfg.EXACT_SOLVER_K_MAX}")

            for input_file in sorted(input_files):
                output_file = os.path.join(cfg.EXACT_OUTPUT_DIR, f"exact_results_{os.path.basename(input_file)}")
                input_file = os.path.join(cfg.INPUT_DIR,input_file)
                process_batch_file_exact(
                    input_csv=input_file, 
                    output_csv=output_file,
                    k_max=cfg.EXACT_SOLVER_K_MAX,
                    solver_name=cfg.EXACT_SOLVER_NAME
                )
                
            end_time = datetime.now()
            print(f"\nAll batches finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total duration: {end_time - start_time}")
    
    elif not pyomo_available:
        print("Skipping exact solver batch because Pyomo is not installed.")
    else:
        print("To run the exact solver batch process, set 'RUN_EXACT_SOLVER_BATCH = True' in 'inventory_config.py'")