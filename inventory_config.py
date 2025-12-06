
# ------------------------------------------------------------------------------
# GENERATOR PARAMS
# ------------------------------------------------------------------------------
MAJOR_DEMAND = 10000
MAJOR_A_LEVELS = [5.0, 10.0, 15.0, 20.0]
MAJOR_CH_MULTIPLIERS = [1.0, 5.0, 10.0, 20.0]
MAJOR_CB_MULTIPLIERS = [0.1, 0.5, 1.0, 5.0]

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
q_threshold_high = 0.4
q_threshold_low = 0.6
DPI = 300

# ------------------------------------------------------------------------------
# BATCH  or  SINGLE PROCESSING
# ------------------------------------------------------------------------------
BATCH = False

PLOT = True
single_file = "test_cases_N5_high_cost - Copy.csv"

# ------------------------------------------------------------------------------
# EXACT SOLVER PARAMS
# ------------------------------------------------------------------------------

RUN_EXACT_SOLVER_BATCH = True

EXACT_SOLVER_K_MAX = 20
EXACT_SOLVER_TIME_LIMIT = 10

EXACT_SOLVER_NAME = 'scip'
EXACT_OUTPUT_DIR = "exact_solver_results"
EXACT_SOLVER_PATH = r"C:\Program Files\IBM\ILOG\CPLEX_Studio_Community2212\cplex\bin\x64_win64\cplex.exe"