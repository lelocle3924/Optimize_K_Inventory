

OUTPUT_DIR = "generated_test_cases"

MAJOR_DEMAND_MIN = 100
MAJOR_DEMAND_MAX = 100000
MAJOR_A_LEVELS = [5, 10, 15, 20] 

NUM_MINORS_LIST = [5, 10, 15, 20, 25]

# Ch_major = multiplier * avg(Ch_minor)
MAJOR_CH_MULTIPLIERS = [1, 5, 10, 20]
# Cb_major = multiplier * Ch_major
MAJOR_CB_MULTIPLIERS = [0.1, 0.5, 1, 5]

# low cost_case
LOW_COST_MIN_A_RANGE = (5, 18)      # A_i = 5 + 13 * rand()
LOW_COST_MIN_CHI_RANGE = (0.2, 1.2) # C_hi = 0.2 + 1 * rand()

# high cost_case
HIGH_COST_MIN_A_RANGE = (0.5, 5.5)   # A_i = 0.5 + 5 * rand()
HIGH_COST_MIN_CHI_RANGE = (0.2, 3.2)  # C_hi = 0.2 + 3 * rand()

LAMBDA_I_RANGE = (0.1, 1.6) # λ_i = 0.1 + 1.5 * rand()
D_I_EXTRA_RANGE = (0, 1)

NUM_DEMANDS_PER_FILE = 1600