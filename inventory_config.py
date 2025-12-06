# ------------------------------------------------------------------------------
PLOT_OPTIONS = {
    "n_super_cycles": 2,
    "minor_index": 0        
}

LOG_DIR = "logs"
OUTPUT_DIR = "batch_results"
INPUT_DIR = "data_test"
# ------------------------------------------------------------------------------
# Q_THRESHOLD
# ------------------------------------------------------------------------------
q_threshold = 0.6
q_threshold_high = 0.6
q_threshold_low = 0.6
DPI = 300

# ------------------------------------------------------------------------------
# BATCH  or  SINGLE PROCESSING
# ------------------------------------------------------------------------------
BATCH = True

PLOT = True
single_file = "test_cases_N5_high_cost.csv"

# ------------------------------------------------------------------------------
# 8. EXACT SOLVER (MINLP) BATCH CONFIGURATION
# ------------------------------------------------------------------------------
# Chạy batch processing bằng bộ giải chính xác (Pyomo).
# Điều này sẽ chạy rất chậm so với heuristic.
RUN_EXACT_SOLVER_BATCH = False

# Giới hạn trên cho biến K trong solver. Giá trị lớn hơn sẽ chính xác hơn
# nhưng cũng làm solver chạy chậm hơn đáng kể.
EXACT_SOLVER_K_MAX = 20

# Tên của solver MINLP đã cài đặt (ví dụ: 'couenne', 'bonmin', 'scip')
EXACT_SOLVER_NAME = 'scip'
EXACT_OUTPUT_DIR = "exact_solver_results"
EXACT_SOLVER_PATH = r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2212\cplex\bin\x64_win64\cplex.exe"