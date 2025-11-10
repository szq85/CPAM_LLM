from docplex.cp.model import CpoModel

with open("data/word.txt", "r") as file:
    word_list = [list(map(int, line.strip()[1:-1].split(","))) for line in file]

complement_map = [1, 0, 3, 2]  
reverse_map = list(range(8))[::-1] 

model = CpoModel()

n_words = 6

words = [[model.integer_var(0, 3, f"w_{i}_{j}") for j in range(8)] for i in range(n_words)]

for i in range(n_words):
    model.add(model.allowed_assignments([words[i][j] for j in range(8)], word_list))

for i in range(n_words):
    model.add(model.count([words[i][j] for j in range(8)], 2) + model.count([words[i][j] for j in range(8)], 3) == 4)

for i in range(n_words):
    for j in range(i + 1, n_words):
        model.add(model.sum(words[i][k] != words[j][k] for k in range(8)) >= 4)

for i in range(n_words):
    for j in range(n_words):
        x_comp = [model.element(complement_map, words[i][k]) for k in range(8)]  
        y_rev = [words[j][reverse_map[k]] for k in range(8)]  
        model.add(model.sum(x_comp[k] != y_rev[k] for k in range(8)) >= 4)

for i in range(n_words):
    model.add((words[i][0] == 2) | (words[i][0] == 3))
    model.add((words[i][7] == 2) | (words[i][7] == 3))

solution = model.solve(TimeLimit = 100, Workers = 4)

if solution:
    print("Found solutions:")
    for i in range(n_words):
        dna_seq = "".join("ATCG"[solution[words[i][j]]] for j in range(8))  
        print(f"{i+1}: {dna_seq}")
else:
    print("No solution found.")
