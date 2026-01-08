
# ------------------------------------------------------------------------------
# GENERATOR PARAMS
# ------------------------------------------------------------------------------
MAJOR_DEMAND = 10000
MAJOR_A_LEVELS = [5.0, 10.0, 15.0, 20.0]
MAJOR_CH_MULTIPLIERS = [1.0, 5.0, 10.0, 20.0]
MAJOR_CB_MULTIPLIERS = [0.1, 0.5, 1.0, 5.0]

# ------------------------------------------------------------------------------
# OPTIMIZER PARAMS
# ------------------------------------------------------------------------------
OPTIMIZE_VERBOSE = False
USE_HEURISTIC_N = False
LOG_DIR = "logs"
#OUTPUT_DIR = "batch_results/exact_64000_results"
#OUTPUT_DIR = "batch_results/gekko_64000_results"
#OUTPUT_DIR = "batch_results/optimize_k_results"
OUTPUT_DIR = "sensitivity_results"
INPUT_DIR = "generate/generated_test_cases_full"
GEKKO_RESULTS_DIR = "batch_results/gekko_64000_results"
#INPUT_DIR = "data_test"
PLOT_OPTIONS = {
    "n_super_cycles": 2,
    "minor_index": 0        
}


# ------------------------------------------------------------------------------
# Q_THRESHOLD
# ------------------------------------------------------------------------------
q_threshold_high = 0.4
q_threshold_low = 0.6
DPI = 300

# ------------------------------------------------------------------------------
# BATCH  or  SINGLE PROCESSING
# ------------------------------------------------------------------------------
BATCH = True

PLOT = True
single_file = "Ntest_low.csv"