import math
import numpy as np
from docplex.cp.model import CpoModel

data_file_path = "data/25c101.txt"
customer_data = []
with open(data_file_path, 'r') as f:
    for line in f:
        if line.strip().startswith("CUST NO."):
            break
    for line in f:
        parts = line.strip().split()
        if parts:
            customer_data.append([int(p) for p in parts])

num_vehicles = 3
vehicle_capacity = 200
depot = 0

num_locations = len(customer_data)
coords = np.array([[c[1], c[2]] for c in customer_data])
demands = [c[3] for c in customer_data]
ready_times = [c[4] for c in customer_data]
due_times = [c[5] for c in customer_data]
service_times = [c[6] for c in customer_data]

distance_matrix = [[0.0] * num_locations for _ in range(num_locations)]
for i in range(num_locations):
    for j in range(num_locations):
        dist = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
        distance_matrix[i][j] = dist

model = CpoModel()

arc = [[[model.binary_var(name=f"arc_{v}_{i}_{j}") for j in range(num_locations)] for i in range(num_locations)] for v
       in range(num_vehicles)]
t = [[model.integer_var(min=ready_times[i], max=due_times[i], name=f"t_{v}_{i}") for i in range(num_locations)] for v in
     range(num_vehicles)]

locations = [i for i in range(num_locations) if i != depot]

for i in locations:
    model.add(model.sum(arc[v][j][i] for v in range(num_vehicles) for j in range(num_locations) if j != i) == 1)

for v in range(num_vehicles):
    model.add(model.sum(arc[v][depot][j] for j in locations) <= 1)
    model.add(model.sum(arc[v][i][depot] for i in locations) == model.sum(arc[v][depot][j] for j in locations))

for v in range(num_vehicles):
    for i in locations:
        model.add(model.sum(arc[v][j][i] for j in range(num_locations) if j != i) == model.sum(
            arc[v][i][k] for k in range(num_locations) if k != i))

for v in range(num_vehicles):
    total_demand = model.sum(
        demands[i] * model.sum(arc[v][j][i] for j in range(num_locations) if j != i) for i in locations)
    model.add(total_demand <= vehicle_capacity)

for v in range(num_vehicles):
    model.add(t[v][depot] == 0)
    for i in range(num_locations):
        for j in locations:
            if i != j:
                travel_time = distance_matrix[i][j]
                service_time = service_times[i]
                model.add(model.if_then(
                    arc[v][i][j] == 1,
                    t[v][j] >= t[v][i] + service_time + int(travel_time)
                ))

total_distance = model.sum(
    arc[v][i][j] * distance_matrix[i][j]
    for v in range(num_vehicles)
    for i in range(num_locations)
    for j in range(num_locations)
)
model.minimize(total_distance)

print("Solving the model...")
solution = model.solve(TimeLimit=60, Workers=4)

if solution:
    print(f"\nSolution found! Objective value (total distance): {solution.get_objective_value():.2f}")
    print("\nThe planned routes are as follows:")
    for v in range(num_vehicles):
        if sum(solution.get_value(f"arc_{v}_{depot}_{j}") for j in locations) == 0:
            continue
        route = [depot]
        current_loc = depot
        while True:
            next_loc = -1
            for j in range(num_locations):
                if current_loc != j and solution.get_value(f"arc_{v}_{current_loc}_{j}") > 0.5:
                    next_loc = j
                    break
            if next_loc == -1 or next_loc == depot:
                route.append(depot)
                break
            route.append(next_loc)
            current_loc = next_loc
        route_str = " -> ".join(map(str, route))
        print(route_str)
else:
    print("\nNo solution found within the time limit!")
