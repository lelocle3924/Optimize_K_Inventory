import csv
import random
import itertools
import os
import config as cfg

def generate_file_for_scenario(num_minors: int, cost_case: str, major_A: float, start_id: int):

    filename = f"{cfg.OUTPUT_DIR}/N{num_minors}_{major_A}_{cost_case}.csv"
    print(f"Generating file for N={num_minors}, case='{cost_case}', A={major_A} -> {filename}")

    major_demands = [
        round(random.uniform(cfg.MAJOR_DEMAND_MIN, cfg.MAJOR_DEMAND_MAX), 2)
        for _ in range(cfg.NUM_DEMANDS_PER_FILE)
    ]

    header = ['case_id', 'maj_D', 'maj_A', 'maj_Ch', 'maj_Cb']
    for i in range(1, num_minors + 1):
        header.extend([f'min_Di_{i}', f'min_Ai_{i}', f'min_Chi_{i}', f'min_lambdai_{i}'])

    multiplier_combinations = list(itertools.product(cfg.MAJOR_CH_MULTIPLIERS, cfg.MAJOR_CB_MULTIPLIERS))

    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for i in range(cfg.NUM_DEMANDS_PER_FILE):
            case_id = start_id + i
            current_major_D = major_demands[i]

            multiplier_pair_index = i // 100
            ch_multiplier, cb_multiplier = multiplier_combinations[multiplier_pair_index]

            minors_data = []
            min_Chi_list = []
            for _ in range(num_minors):
                lambda_i = round(random.uniform(*cfg.LAMBDA_I_RANGE), 2)
                d_i_ratio = lambda_i + random.uniform(*cfg.D_I_EXTRA_RANGE)

                if cost_case == 'low':
                    A_i = round(random.uniform(*cfg.LOW_COST_MIN_A_RANGE), 2)
                    C_hi = round(random.uniform(*cfg.LOW_COST_MIN_CHI_RANGE), 2)
                else: # high
                    A_i = round(random.uniform(*cfg.HIGH_COST_MIN_A_RANGE), 2)
                    C_hi = round(random.uniform(*cfg.HIGH_COST_MIN_CHI_RANGE), 2)
                
                minors_data.append({'d_i_ratio': d_i_ratio, 'A_i': A_i, 'C_hi': C_hi, 'lambda_i': lambda_i})
                min_Chi_list.append(C_hi)

            # calculate holding and backorder for major
            avg_min_Chi = sum(min_Chi_list) / len(min_Chi_list) if min_Chi_list else 1.0
            maj_Ch = round(ch_multiplier * avg_min_Chi, 2)
            maj_Cb = round(cb_multiplier * maj_Ch, 2)

            row = [case_id, current_major_D, major_A, maj_Ch, maj_Cb]
            for item_data in minors_data:
                D_i = round(item_data['d_i_ratio'] * current_major_D, 2)
                row.extend([D_i, item_data['A_i'], item_data['C_hi'], item_data['lambda_i']])
            
            writer.writerow(row)

    print(f"-> Done. Generated {cfg.NUM_DEMANDS_PER_FILE} cases.")


if __name__ == '__main__':
    # Đảm bảo thư mục output tồn tại
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    global_case_id = 1
    
    # Tạo tất cả các tổ hợp kịch bản
    scenarios = itertools.product(
        cfg.NUM_MINORS_LIST,
        cfg.MAJOR_A_LEVELS, 
        ['high', 'low']
    )

    for num_minors, major_A, cost_case in scenarios:
        generate_file_for_scenario(
            num_minors=num_minors, 
            cost_case=cost_case,
            major_A=major_A,
            start_id=global_case_id
        )
        global_case_id += cfg.NUM_DEMANDS_PER_FILE
    
    print("\nAll test case files generated successfully.")