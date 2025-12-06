# inventory_env.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import itertools 

import inventory_config as cfg

# =========================
# Data structures
# =========================

@dataclass
class MajorItem:
    D: float
    A: float
    C_h: float
    C_b: float


@dataclass
class MinorItem:
    D_i: float
    A_i: float
    C_hi: float
    ell_i: float
    name: str = "Minor"


@dataclass
class InventorySystem:
    major: MajorItem
    minors: List[MinorItem]
    d_list: List[float] = field(init=False)

    def __post_init__(self):
        D = self.major.D
        self.d_list = [m.D_i / D for m in self.minors]

    def compute_G0_G1_G2_G3(self, K: List[int]) -> Tuple[float, float, float, float]:
        D = self.major.D
        A = self.major.A
        C_h = self.major.C_h
        C_b = self.major.C_b

        if len(K) != len(self.minors):
            raise ValueError("Length of K must equal number of minors")

        G0 = A + sum(m.A_i / k_i for m, k_i in zip(self.minors, K))
        sum_ellC = sum(m.ell_i * m.C_hi for m in self.minors)
        G1 = (C_h + C_b + sum_ellC) * D / 2.0
        G2 = C_b * D

        d_list = self.d_list
        sum_term = 0.0
        for k_i, m, d_i in zip(K, self.minors, d_list):
            sum_term += (k_i * d_i - m.ell_i) * m.C_hi
        G3 = (C_b + sum_term) * D / 2.0

        return G0, G1, G2, G3

    def u_of_F(self, F: float, G1: float, G2: float, G3: float) -> float:
        return G1 * F**2 - G2 * F + G3

    # ---------- Cost Function (Eq 2 / Eq 5) ----------
    def cost_given_K_T_F(self, K: List[int], T: float, F: float) -> float:
        if T <= 0: return float('inf')
        G0, G1, G2, G3 = self.compute_G0_G1_G2_G3(K)
        uF = self.u_of_F(F, G1, G2, G3)
        return G0 / T + T * uF

    # ---------- Optimal T* and F* (Eq 8 & Eq 6) ----------
    def optimal_T_F_for_K(self, K: List[int]) -> Tuple[float, float, float]:
        G0, G1, G2, G3 = self.compute_G0_G1_G2_G3(K)
        
        # Eq (8): F*
        if G1 <= 0:
            F_star = 1.0
        else:
            F_star = max(0.0, min(1.0, (G2 / (2.0 * G1))))
            
        # Eq (9) or Eq (6): T*
        # u(F) = G1*F^2 - G2*F + G3
        uF = self.u_of_F(F_star, G1, G2, G3)
        
        if uF <= 0:
            # Fallback for numerical stability, though theoretically u(F) > 0
            T_star = 1.0 
        else:
            T_star = math.sqrt(G0 / uF)
            
        cost_star = self.cost_given_K_T_F(K, T_star, F_star)
        return T_star, F_star, cost_star

    # =================================================
    #  PLOT FUNCTIONS (Updated for Multiple Items)
    # =================================================
    def _build_major_paths(self, T: float, F: float, n_cycles: int = 2):
        D = self.major.D
        times, onhand, backlog = [], [], []
        H_max = D * F * T
        B_max = D * (1.0 - F) * T
        current_time = 0.0
        
        for c in range(n_cycles):
            t0 = current_time
            times += [t0, t0 + F*T]; onhand += [H_max, 0.0]; backlog += [0.0, 0.0]
            times += [t0 + F*T, t0 + T]; onhand += [0.0, 0.0]; backlog += [0.0, B_max]
            current_time += T
        return times, onhand, backlog

    def _build_minor_path(self, idx: int, k_i: int, T: float, F: float, n_super_cycles: int = 1):
        m = self.minors[idx]
        D, D_i, ell_i = self.major.D, m.D_i, m.ell_i
        D_iu = D_i - ell_i * D
        B = D * (1.0 - F) * T
        I_iuu = ell_i * B
        Qi = k_i * D_i * T
        times, level = [], []
        
        for s in range(n_super_cycles):
            base = s * k_i * T
            inv = Qi
            for j in range(k_i):
                cycle_start = base + j*T
                times.append(cycle_start); level.append(inv)
                inv -= I_iuu
                times.append(cycle_start); level.append(inv)
                t1 = cycle_start + F*T
                times.append(t1); inv -= D_i * F * T; level.append(inv)
                t2 = cycle_start + T
                times.append(t2); inv -= D_iu * (1.0 - F) * T; level.append(inv)
            times.append(base + k_i*T); level.append(inv)
            times.append(base + k_i*T); inv = Qi; level.append(inv)
        return times, level

    def plot_figure1_like(self, K: List[int], T: float, F: float, n_super_cycles: int = 1, minor_index=None, case_id=None):
        max_k = max(K) if K else 1
        major_cycles_needed = max_k * n_super_cycles
        
        times_M, onhand_M, backlog_M = self._build_major_paths(T, F, n_cycles=major_cycles_needed)

        plt.figure(figsize=(12, 6))
        plt.plot(times_M, onhand_M, label="Major (On-hand)", linewidth=2, color='tab:blue')
        plt.plot(times_M, backlog_M, label="Major (Backlog)", linestyle="--", linewidth=2, color='tab:blue')

        colors = itertools.cycle(['tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink'])
        
        for i, (k_i, minor) in enumerate(zip(K, self.minors)):
            cycles_for_this_item = math.ceil((max_k * n_super_cycles) / k_i)
            times_m, level_m = self._build_minor_path(i, k_i, T, F, cycles_for_this_item)
            
            # Trim to match plot length
            max_time = times_M[-1]
            times_m_trimmed = [t for t in times_m if t <= max_time]
            level_m_trimmed = level_m[:len(times_m_trimmed)]
            
            c = next(colors)
            plt.plot(times_m_trimmed, level_m_trimmed, label=f"{minor.name} (k={k_i})", linewidth=1.5, color=c)

        plt.axhline(0, color="black", linewidth=0.8)
        plt.xlabel("Time (Years)")
        plt.ylabel("Inventory Level")
        plt.title(f"Joint Replenishment Inventory Levels case {case_id}\nT*={T:.4f}, F*={F:.2%}, K={K}")
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{cfg.LOG_DIR}/inventory_plot_case_{case_id}.png", dpi=cfg.DPI)
        plt.show()

if __name__ == '__main__':
    import os
    import csv
    from datetime import datetime

    def process_file_with_fixed_k(input_csv: str, output_csv: str):
        """
        Đọc một file test case và tính toán chi phí cho mỗi dòng với K cố định là [1,1,...].
        """
        print(f"  Processing baseline for: '{os.path.basename(input_csv)}'")
        with open(input_csv, 'r', encoding='utf-8') as f_in, \
             open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
            
            reader = csv.reader(f_in)
            writer = csv.writer(f_out)
            
            # Tạo header cho file output, tương tự như file kết quả của optimize_k_batch
            header_in = next(reader)
            num_minors = (len(header_in) - 5) // 4
            output_header = ['case_id', 'major_A', 'major_Ch', 'major_Cb', 'K_star', 'T_star', 'F_star', 'G_star', 'warning_negative_demand']
            writer.writerow(output_header)

            for row in reader:
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
                
                # --- Tính toán với K cố định ---
                # Đây là điểm khác biệt chính: K luôn là một vector gồm các số 1
                K_fixed = [1] * len(minors)
                
                try:
                    # Sử dụng phương thức có sẵn để tính T*, F*, và Cost* cho K đã cho
                    T_star, F_star, G_star = system.optimal_T_F_for_K(K_fixed)
                except Exception as e:
                    T_star, F_star, G_star = "ERROR", "ERROR", f"ERROR: {e}"

                # --- Kiểm tra "nhu cầu âm" ---
                negative_demand_items = [f'M{j+1}' for j, m in enumerate(minors) if (m.D_i - m.ell_i * major.D) < 0]
                warning_msg = f"Negative demand: {','.join(negative_demand_items)}" if negative_demand_items else "None"
                
                # --- Ghi kết quả ---
                # Chuyển đổi T_star, F_star, G_star sang string để ghi cho an toàn
                T_str = f"{T_star:.6f}" if isinstance(T_star, float) else T_star
                F_str = f"{F_star:.6f}" if isinstance(F_star, float) else F_star
                G_str = f"{G_star:.4f}" if isinstance(G_star, float) else G_star

                output_row = [case_id, maj_A, maj_Ch, maj_Cb, str(K_fixed), T_str, F_str, G_str, warning_msg]
                writer.writerow(output_row)

    # --- Logic chính để chạy hàng loạt ---
    print("="*50)
    print("RUNNING BASELINE CALCULATION (K=[1,1,...])")
    print("="*50)

    # Đảm bảo các thư mục tồn tại
    os.makedirs(cfg.INPUT_DIR, exist_ok=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    try:
        input_files = [f for f in os.listdir(cfg.INPUT_DIR) if f.startswith('test_cases_') and f.endswith('.csv')]
    except FileNotFoundError:
        print(f"\nError: Input directory '{cfg.INPUT_DIR}' not found.")
        input_files = []

    if not input_files:
        print("No 'test_cases_*.csv' files found in input directory.")
        print("Please run 'generate_test_cases.py' first and ensure files are in the correct directory.")
    else:
        start_time = datetime.now()
        print(f"Starting baseline calculation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Found {len(input_files)} test case file(s) in '{cfg.INPUT_DIR}/'")

        for input_file in sorted(input_files):
            # Tạo tên file output riêng biệt
            output_filename = f"results_K1_{os.path.basename(input_file)}"
            
            input_path = os.path.join(cfg.INPUT_DIR, input_file)
            output_path = os.path.join(cfg.OUTPUT_DIR, output_filename)
            
            process_file_with_fixed_k(input_path, output_path)
            
        end_time = datetime.now()
        print(f"\nAll baseline calculations finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {end_time - start_time}")
        print(f"Baseline results saved in '{cfg.OUTPUT_DIR}/' with prefix 'results_K1_'")