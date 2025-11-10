import numpy as np
from docplex.cp.model import CpoModel
import sys
import os

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/battery_data.txt")
with open(filename, "r") as file:
    lines = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]

params = lines[0].split()
NUM_PARALLEL_MODULES = int(params[0])
NUM_SERIES_COLUMNS = int(params[1])
NUM_PERIODS = int(params[2])
CELL_NOMINAL_VOLTAGE = float(params[3])
I_MAX_MODULE = int(params[4])
all_numbers = [float(num) for line in lines[1:] for num in line.split()]
data_points_per_matrix = NUM_PERIODS * NUM_PARALLEL_MODULES * NUM_SERIES_COLUMNS
if len(all_numbers) != 3 * data_points_per_matrix:
    print(f"Error: Data size mismatch in '{filename}'.")
    print(f"Expected {3 * data_points_per_matrix} numbers, but found {len(all_numbers)}.")
    sys.exit(1)

C = np.array(all_numbers[0 : data_points_per_matrix]).reshape((NUM_PERIODS, NUM_PARALLEL_MODULES, NUM_SERIES_COLUMNS))
R = np.array(all_numbers[data_points_per_matrix : 2 * data_points_per_matrix]).reshape((NUM_PERIODS, NUM_PARALLEL_MODULES, NUM_SERIES_COLUMNS))
V = np.array(all_numbers[2 * data_points_per_matrix :]).reshape((NUM_PERIODS, NUM_PARALLEL_MODULES, NUM_SERIES_COLUMNS))

NOMINAL_VOLTAGE = NUM_SERIES_COLUMNS * CELL_NOMINAL_VOLTAGE
I_load = [35] * NUM_PERIODS
V_lower = [NOMINAL_VOLTAGE * 0.95] * NUM_PERIODS
V_upper = [NOMINAL_VOLTAGE * 1.05] * NUM_PERIODS

model = CpoModel()

s_keys = [(k, i, j) for k in range(NUM_PERIODS)
          for i in range(NUM_PARALLEL_MODULES)
          for j in range(NUM_SERIES_COLUMNS)]
s = model.binary_var_dict(s_keys, name='s')
p = model.integer_var_list(NUM_PERIODS, min=1, max=NUM_PARALLEL_MODULES, name='p')

for k in range(NUM_PERIODS):
    for j in range(NUM_SERIES_COLUMNS):
        model.add(model.sum(s[k, i, j] for i in range(NUM_PARALLEL_MODULES)) == p[k])

    vars_for_period_k = [s[k, i, j] for i in range(NUM_PARALLEL_MODULES) for j in range(NUM_SERIES_COLUMNS)]
    voltages_for_period_k = [V[k, i, j] for i in range(NUM_PARALLEL_MODULES) for j in range(NUM_SERIES_COLUMNS)]
    total_voltage_contribution = model.scal_prod(vars_for_period_k, voltages_for_period_k)

    model.add(total_voltage_contribution >= V_lower[k] * p[k])
    model.add(total_voltage_contribution <= V_upper[k] * p[k])
    model.add(I_load[k] <= p[k] * I_MAX_MODULE)

ALLOWED_USAGE_DEVIATION = 1
module_usage = [model.sum(s[k, i, j] for k in range(NUM_PERIODS))
                for i in range(NUM_PARALLEL_MODULES)
                for j in range(NUM_SERIES_COLUMNS)]
model.add(model.max(module_usage) - model.min(module_usage) <= ALLOWED_USAGE_DEVIATION)

cost = C + R
s_vars_flat = [s[key] for key in s_keys]
cost_flat = [cost[k, i, j] for k, i, j in s_keys]
total_cost_objective = model.scal_prod(s_vars_flat, cost_flat)
model.add(model.minimize(total_cost_objective))

print("Solving model...")
solution = model.solve(TimeLimit=100)

if solution:
    print("Solution found!")
    print(f"Minimized objective function value: {solution.get_objective_value():.4f}\n")

    for k in range(NUM_PERIODS):
        active_modules_count = solution[p[k]]
        print(
            f"--- Period {k + 1} (Load: {I_load[k]} A) Optimal Configuration (Active per column: {active_modules_count}) ---")
        config_matrix = np.zeros((NUM_PARALLEL_MODULES, NUM_SERIES_COLUMNS), dtype=int)
        total_v_calc = 0
        for j in range(NUM_SERIES_COLUMNS):
            col_v = 0
            for i in range(NUM_PARALLEL_MODULES):
                is_active = solution[s[k, i, j]]
                config_matrix[i, j] = is_active
                if is_active:
                    col_v += V[k, i, j]
            if active_modules_count > 0:
                total_v_calc += col_v / active_modules_count
        print("Switch Matrix (1=ON, 0=OFF):")
        print(config_matrix)
        print(f"  - Calculated Output Voltage: {total_v_calc:.2f} V (Constraint: [{V_lower[k]:.2f}, {V_upper[k]:.2f}])")
        if active_modules_count > 0:
            module_current = I_load[k] / active_modules_count
            print(f"  - Current per active module: {module_current:.2f} A (Max allowed: {I_MAX_MODULE} A)\n")
        else:
            print("  - No active modules.\n")
else:
    print("No feasible solution found within the time limit.")
