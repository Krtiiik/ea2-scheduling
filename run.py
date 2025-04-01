import argparse
import os
import pickle

from matplotlib import pyplot as plt
import tabulate

import cpsolver
from evolution import EvolutionSolver
from instances import load_instances
from plotting import plot_gantt_chart
import solvers


# DATA_DIR = os.path.join("data", "j30.sm")
DATA_DIR = os.path.join("data")
RESULTS_DIR = "results"
RESULTS = {
    "exact": os.path.join(RESULTS_DIR, "exact.pkl"),
    "evolution": os.path.join(RESULTS_DIR, "evolution.pkl"),
    "table": os.path.join(RESULTS_DIR, "results.txt"),
}
CONFIGURATION : solvers.Configuration = {
    "time_limit": 60,
}

parser = argparse.ArgumentParser()


def main(args):
    solver_exact = cpsolver.CPSolver()
    solver_evolution = EvolutionSolver()
    solver_exact.configure(CONFIGURATION, args)
    solver_evolution.configure(CONFIGURATION, args)

    # with open(RESULTS["evolution"], "rb") as f:
    #     solutions_evolution = pickle.load(f)
    
    # print(*solutions_evolution, sep="\n")

    # raise Exception("Stop here")

    instances = load_instances(DATA_DIR)

    # solutions_exact = solver_exact.solve_all(instances)
    solutions_evolution = solver_evolution.solve_all(instances)

    # with open(RESULTS["exact"], "wb") as f:
    #     pickle.dump(solutions_exact, f)
    with open(RESULTS["evolution"], "wb") as f:
        pickle.dump(solutions_evolution, f)

    # TODO analyze solutions
    # save_results_table(instances, solutions_exact)

    inst_ = "j3013_1.sm"
    [(i, inst)] = [(i, inst) for i, inst in enumerate(instances) if inst.name == inst_]
    inst = instances[i]
    # solution = solutions_exact[i]

    # plot_gantt_chart(solution, inst)



def save_results_table(instances, solutions_exact):
    with open(RESULTS["table"], 'wt') as f:
        f.write(tabulate.tabulate(
            sorted(
                [{
                    "instance": instance.name,
                    "exact_kind": solution["kind"].value,
                    "exact_makespan": solution["makespan"],
                    # "evolution": solution_evolution["makespan"],
                } for solution, instance in zip(solutions_exact, instances)],
                key=lambda x: (
                    x["instance"].startswith("j120"), 
                    x["instance"].startswith("j90"),
                    x["instance"].startswith("j60"), 
                    x["instance"].startswith("j30"))
            ),
            headers="keys",
        ))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
