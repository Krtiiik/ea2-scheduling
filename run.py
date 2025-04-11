import argparse
import os
import pickle

from matplotlib import pyplot as plt
import tabulate

from annealing import AnnealingSolver
from annealing import AnnealingSolver
import cpsolver
from evolution import EvolutionSolver
from instances import ProblemInstance, load_instances
from plotting import plot_fitnesses, plot_gantt_chart, plot_fitness_graph, plot_comparison, plot_makespans
import solvers


# DATA_DIR = os.path.join("data", "j30.sm")
DATA_DIR = os.path.join("data")
RESULTS_DIR = "results"
RESULTS = {
    "exact": os.path.join(RESULTS_DIR, "exact.pkl"),
    "evolution": os.path.join(RESULTS_DIR, "evolution.pkl"),
    "annealing": os.path.join(RESULTS_DIR, "annealing.pkl"),
    "table": os.path.join(RESULTS_DIR, "results.txt"),
}
RESULTS_EVAL_DIR = os.path.join(RESULTS_DIR, "evals ")
RESULTS_EVAL = {
    "exact": os.path.join(RESULTS_EVAL_DIR, "exact"),
    "evolution": os.path.join(RESULTS_EVAL_DIR, "evolution"),
    "annealing": os.path.join(RESULTS_EVAL_DIR, "annealing"),
}
CONFIGURATION = solvers.Configuration(
    time_limit = 30,
)

parser = argparse.ArgumentParser()


def main(args):
    solver_exact = cpsolver.CPSolver()
    solver_evolution = EvolutionSolver()
    solver_annealing = AnnealingSolver()
    solver_exact.configure(CONFIGURATION, args)
    solver_evolution.configure(CONFIGURATION, args)
    solver_annealing.configure(CONFIGURATION, args)

    # instances = load_instances(DATA_DIR)
    with open("instances.pkl", "rb") as f:
        instances = pickle.load(f)

    # print("Solving exact...")
    # solutions_exact = solver_exact.solve_all(instances)
    # print("Solving evolution...")
    # solutions_evolution = solver_evolution.solve_all(instances)
    # print("Solving annealing...")
    # solutions_annealing = solver_annealing.solve_all(instances)

    # with open(RESULTS["exact"], "wb") as f:
    #     pickle.dump(solutions_exact, f)
    # with open(RESULTS["evolution"], "wb") as f:
    #     pickle.dump(solutions_evolution, f)
    # with open(RESULTS["annealing"], "wb") as f:
    #     pickle.dump(solutions_annealing, f)

    with open(RESULTS["exact"], "rb") as f:
        solutions_exact = pickle.load(f)
    with open(RESULTS["evolution"], "rb") as f:
        solutions_evolution = pickle.load(f)
    with open(RESULTS["annealing"], "rb") as f:
        solutions_annealing = pickle.load(f)

    save_results_table(instances, solutions_exact, solutions_evolution, solutions_annealing)
    plot_comparison(instances, solutions_exact, solutions_evolution, solutions_annealing)
    plot_makespans(instances, solutions_exact, solutions_evolution, solutions_annealing)
    plot_fitnesses(
        [sol.eval_log for sol in solutions_evolution],
        [sol.eval_log for sol in solutions_annealing],
    )


def instance_size_param(instance):
    size_param1, param2 = instance[1:].split('_')
    split = size_param1.index('0')
    size, param1 = size_param1[:split+1], size_param1[split+1:]
    return tuple(map(int, (size, param1, param2)))


def save_results_table(
    instances: list[ProblemInstance],
    solutions_exact: list[solvers.Solution],
    solutions_evolution: list[solvers.Solution],
    solutions_annealing: list[solvers.Solution],
    ):
    with open(RESULTS["table"], 'wt') as f:
        f.write(tabulate.tabulate(
            sorted(
                [{
                    "instance": instance.name[:-3],
                    "makespan_exact": sol_exact.makespan,
                    "makespan_evolution": sol_evolution.makespan,
                    "makespan_annealing": sol_annealing.makespan,
                    "runtime_exact": sol_exact.runtime,
                    "runtime_evolution": sol_evolution.runtime,
                    "runtime_annealing": sol_annealing.runtime,
                } for instance, sol_exact, sol_evolution, sol_annealing
                  in zip(instances, solutions_exact, solutions_evolution, solutions_annealing)],
                key=lambda x: instance_size_param(x["instance"]),
            ),
            headers="keys",
        ))


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
