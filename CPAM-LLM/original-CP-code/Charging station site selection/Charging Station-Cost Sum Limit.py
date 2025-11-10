import os
from docplex.cp.model import CpoModel

filename = os.path.join("data", "plant_location.data")
with open(filename, "r") as file:
    data = [int(v) for v in file.read().split()]

ptr = 0
full_nbCustomer = data[ptr];
ptr += 1
full_nbLocation = data[ptr];
ptr += 1
full_cost = [[data[ptr + c * full_nbLocation + l] for l in range(full_nbLocation)] for c in range(full_nbCustomer)]
ptr += full_nbCustomer * full_nbLocation
full_demand = [data[ptr + c] for c in range(full_nbCustomer)];
ptr += full_nbCustomer
full_fixedCost = [data[ptr + p] for p in range(full_nbLocation)];
ptr += full_nbLocation
full_capacity = [data[ptr + p] for p in range(full_nbLocation)];
ptr += full_nbLocation

nbDemandAreas = 30
nbLocations = 20

cost = [row[:nbLocations] for row in full_cost[:nbDemandAreas]]
demand = full_demand[:nbDemandAreas]
fixedCost = full_fixedCost[:nbLocations]
capacity = full_capacity[:nbLocations]

mdl = CpoModel()

assignment = mdl.integer_var_list(nbDemandAreas, 0, nbLocations - 1, "DemandAreaAssignment")
open_station = mdl.integer_var_list(nbLocations, 0, 1, "OpenStation")
load = [mdl.integer_var(0, cap, f"StationLoad_{p}") for p, cap in enumerate(capacity)]

for p in range(nbLocations):
    mdl.add(open_station[p] == (load[p] > 0))
mdl.add(mdl.pack(load, assignment, demand))

obj = mdl.scal_prod(fixedCost, open_station)
mdl.add(obj >= 20000)
mdl.add(25000 >= obj)

for c in range(nbDemandAreas):
    obj += mdl.element(assignment[c], cost[c])
mdl.add(mdl.minimize(obj))

print("Solving the model...")
msol = mdl.solve(TimeLimit=100)

if msol:
    print("\n------------------ SOLUTION FOUND ------------------")
    print(f"Objective Value (Total Cost): {msol.get_objective_value():.2f}")
    open_stations_indices = [i for i, var in enumerate(open_station) if msol.get_value(var) == 1]
    print(f"Stations to open: {open_stations_indices}")

    print("\n--- Station Loads and Capacities ---")
    loads_solution = [msol.get_value(l) for l in load]
    for i in open_stations_indices:
        utilization = (loads_solution[i] / capacity[i]) * 100 if capacity[i] > 0 else 0
        print(f"Station {i}: Load = {loads_solution[i]}, Capacity = {capacity[i]} ({utilization:.1f}% utilized)")

    print("\n--- Demand Area Assignments ---")
    assignment_solution = [msol.get_value(v) for v in assignment]
    for i, assigned_station in enumerate(assignment_solution):
        print(f"Demand Area {i} is assigned to Station {assigned_station}")
else:
    print("\nNo solution found within the time limit.")