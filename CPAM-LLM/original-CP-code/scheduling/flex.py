from docplex.cp.model import CpoModel
import os

filename = os.path.dirname(os.path.abspath(__file__)) + "/data/fjsp_location_newdataset_1.fjs"
with open(filename, "r") as file:
    NB_JOBS, NB_MACHINES = [int(v) for v in file.readline().split()]
    list_jobs = [[v if v in ['tight', 'loose', 'none'] else int(v) for v in file.readline().split()] for i in
                 range(NB_JOBS)]

JOBS = []
CONSTRAINTS = []  

for jline in list_jobs:
    nbstps = jline.pop(0)
    job = []
    constraints = []
    for stp in range(nbstps):
        nbc = jline.pop(0)
        choices = []
        for c in range(nbc):
            m = jline.pop(0)
            d = jline.pop(0)
            choices.append((m - 1, d))
        constraint = jline.pop(0) 
        job.append(choices)
        constraints.append(constraint)
    JOBS.append(job)
    CONSTRAINTS.append(constraints)

mdl = CpoModel()

job_number = {}
all_operations = [] 
machine_operations = [[] for m in range(NB_MACHINES)] 
job_operations = []  

for jx, job in enumerate(JOBS):    
    op_vars = []
    for ox, op in enumerate(job):       
        choice_vars = []
        for cx, (m, d) in enumerate(op):    
            cv = mdl.interval_var(name="J{}_O{}_C{}_M{}".format(jx, ox, cx, m), optional=True, size=d)
            job_number[cv.get_name()] = jx
            choice_vars.append(cv)
            machine_operations[m].append(cv)

        jv = mdl.interval_var(name="J{}_O{}".format(jx, ox))
        mdl.add(mdl.alternative(jv, choice_vars))
        op_vars.append(jv)

        if ox > 0:
            mdl.add(mdl.end_before_start(op_vars[ox - 1], op_vars[ox]))
        if ox > 0 and CONSTRAINTS[jx][ox - 1] == 'tight':
            mdl.add(mdl.start_of(op_vars[ox]) - mdl.end_of(op_vars[ox - 1]) <= 15)
        if ox > 0 and CONSTRAINTS[jx][ox - 1] == 'loose':
            mdl.add(mdl.start_of(op_vars[ox]) - mdl.end_of(op_vars[ox - 1]) >= 30)
    all_operations.extend(op_vars)
    job_operations.append(op_vars)  

for lops in machine_operations:
    mdl.add(mdl.no_overlap(lops))

mdl.add(mdl.minimize(mdl.max([mdl.end_of(op) for op in all_operations])))

print("Solving model....")
msol = mdl.solve(FailLimit=100000)

if msol:
    print("Solution found.")
    print("Objective value (Makespan):", msol.get_objective_value())

    print("=== Machine Schedules ===")
    for m in range(NB_MACHINES):
        print(f"\nMachine M{m}:")
        machine_schedule = []
        for v in machine_operations[m]:
            itv = msol.get_var_solution(v)
            if itv.is_present():
                jn = job_number[v.get_name()]
                machine_schedule.append((itv.start, itv.end, f"J{jn}"))
        machine_schedule.sort(key=lambda x: x[0])
        for s, e, jn in machine_schedule:
            print(f"  {jn:<5} Start: {s:<5}  End: {e:<5}  Duration: {e - s}")
else:
    print("No solution found.")