import csv
import random
import itertools
import os
import inventory_config as cfg
def generate_cases(num_minors: int, cost_case: str, start_id: int = 1):
    major_D_fixed = cfg.MAJOR_DEMAND
    major_A_levels = cfg.MAJOR_A_LEVELS
    major_Ch_multipliers = cfg.MAJOR_CH_MULTIPLIERS
    major_Cb_multipliers = cfg.MAJOR_CB_MULTIPLIERS
    num_samples = len(major_A_levels)*len(major_Cb_multipliers)*len(major_Ch_multipliers)

    filename = f"{cfg.INPUT_DIR}/test_cases_N{num_minors}_{cost_case}_cost.csv"
    print(f"Generating file: {filename} with {num_samples} factorial cases...")

    header = ['case_id', 'maj_D', 'maj_A', 'maj_Ch', 'maj_Cb']
    for i in range(1, num_minors + 1):
        header.extend([f'min_Di_{i}', f'min_Ai_{i}', f'min_Chi_{i}', f'min_elli_{i}'])
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        minors_data = []
        min_Chi_list = []
        for _ in range(num_minors):
            ell_i = round(0.1 + 1.5 * random.random(), 2)
            d_i = round(ell_i + random.random(), 2)
            if cost_case == 'low':
                A_i = round(5 + 13 * random.random(), 2)
                C_hi = round(0.2 + 1 * random.random(), 2)
            else:
                A_i = round(0.5 + 5 * random.random(), 2)
                C_hi = round(0.2 + 3 * random.random(), 2)
            
            minors_data.append({'ell_i': ell_i, 'd_i': d_i, 'A_i': A_i, 'C_hi': C_hi})
            min_Chi_list.append(C_hi)
        
        case_counter = 0
        for maj_A in major_A_levels:
            for ch_multiplier in major_Ch_multipliers:
                for cb_multiplier in major_Cb_multipliers:
                    case_id = start_id + case_counter                   
                    avg_min_Chi = sum(min_Chi_list) / len(min_Chi_list) if min_Chi_list else 1.0
                    
                    maj_Ch = round(ch_multiplier * avg_min_Chi, 2)
                    maj_Cb = round(cb_multiplier * maj_Ch, 2)
                    
                    row = [case_id, major_D_fixed, maj_A, maj_Ch, maj_Cb]
                    for item_data in minors_data:
                        D_i = round(item_data['d_i'] * major_D_fixed, 2)
                        row.extend([D_i, item_data['A_i'], item_data['C_hi'], item_data['ell_i']])
                    
                    writer.writerow(row)
                    case_counter += 1
    os.makedirs(cfg.INPUT_DIR, exist_ok=True)
    filename = os.path.join(cfg.INPUT_DIR, filename)
    print(f"-> Done. Saved to '{filename}'")

if __name__ == '__main__':
    NUM_MINORS_LIST = [5, 10, 15, 20, 25]
    COST_CASES = ['low', 'high']
    SAMPLES_PER_FILE = len(cfg.MAJOR_A_LEVELS)*len(cfg.MAJOR_CB_MULTIPLIERS)*len(cfg.MAJOR_CH_MULTIPLIERS)
    
    global_case_id = 1
    for num_minors, cost_case in itertools.product(NUM_MINORS_LIST, COST_CASES):
        generate_cases(
            num_minors=num_minors, 
            cost_case=cost_case, 
            start_id=global_case_id
        )
        global_case_id += SAMPLES_PER_FILE
    
    print("\nAll test case files generated successfully.")