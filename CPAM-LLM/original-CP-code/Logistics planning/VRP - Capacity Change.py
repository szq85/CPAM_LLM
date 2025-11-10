import math
import numpy as np
from docplex.cp.model import CpoModel

data_file_path = 'data/25c101.txt'
customer_data = []
with open(data_file_path, 'r') as f:
    next(f)
    for line in f:
        parts = line.split()
        if parts:
            customer_data.append([int(p) for p in parts[:4]])

num_vehicles = 3
vehicle_capacity = 180
depot = 0

num_locations = len(customer_data)
locations = [i for i in range(num_locations) if i != depot]
demands = [data[3] for data in customer_data]
coords = np.array([[data[1], data[2]] for data in customer_data])

distance_matrix = [[0.0] * num_locations for _ in range(num_locations)]
for i in range(num_locations):
    for j in range(num_locations):
        if i != j:
            x1, y1 = customer_data[i][1], customer_data[i][2]
            x2, y2 = customer_data[j][1], customer_data[j][2]
            distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            distance_matrix[i][j] = distance

model = CpoModel()

arc = [[[model.binary_var(name=f"arc_{v}_{i}_{j}")
         for j in range(num_locations)]
        for i in range(num_locations)]
       for v in range(num_vehicles)]

for i in locations:
    model.add(model.sum(arc[v][j][i] for v in range(num_vehicles) for j in range(num_locations) if i != j) == 1)
    model.add(model.sum(arc[v][i][j] for v in range(num_vehicles) for j in range(num_locations) if i != j) == 1)

for v in range(num_vehicles):
    model.add(model.sum(arc[v][depot][j] for j in locations) <= 1)
    model.add(model.sum(arc[v][j][depot] for j in locations) <= 1)

for v in range(num_vehicles):
    for i in locations:
        model.add(model.sum(arc[v][j][i] for j in range(num_locations) if j != i) ==
                  model.sum(arc[v][i][j] for j in range(num_locations) if j != i))

for v in range(num_vehicles):
    vehicle_demand = model.sum(
        demands[j] * model.sum(arc[v][i][j] for i in range(num_locations) if i != j) for j in locations)
    model.add(vehicle_demand <= vehicle_capacity)

max_route_length = num_locations
u = [[model.integer_var(min=0, max=max_route_length, name=f"u_{v}_{i}")
      for i in range(num_locations)]
     for v in range(num_vehicles)]

for v in range(num_vehicles):
    model.add(u[v][depot] == 0)
    for i in locations:
        model.add(model.if_then(model.sum(arc[v][i][j] for j in range(num_locations) if i != j) == 0, u[v][i] == 0))
        for j in locations:
            if i != j:
                model.add(model.if_then(arc[v][i][j] == 1, u[v][j] >= u[v][i] + 1))

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
