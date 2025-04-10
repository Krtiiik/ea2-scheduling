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
from plotting import plot_gantt_chart, plot_fitness_graph
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
    print("Current absolute directory:", os.path.abspath(os.getcwd()))

    solver_exact = cpsolver.CPSolver()
    solver_evolution = EvolutionSolver()
    solver_annealing = AnnealingSolver()
    solver_exact.configure(CONFIGURATION, args)
    solver_evolution.configure(CONFIGURATION, args)
    solver_annealing.configure(CONFIGURATION, args)

    instances = load_instances(DATA_DIR)

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

    # def compute_consumptions(schedule: dict[int, int], instance: ProblemInstance):
    #     for t in range(instance.horizon):
    #         scheduled_jobs = [job for job in instance.jobs if schedule[job.id_job] <= t < (schedule[job.id_job] + job.duration)]
    #         print(f"{t:3} | ", end='')
    #         for resource in instance.resources:
    #             consumption = sum(job.resource_consumption[resource] for job in scheduled_jobs)
    #             print(f'{consumption} / {resource.capacity} | ', end='')
    #         # print(scheduled_jobs)
    #         print()

    # compute_consumptions(solutions_exact[0]["schedule"], instances[0])
    # compute_consumptions(solutions_evolution[0][0], instances[0])
    # compute_consumptions(solutions_annealing[0][0], instances[0])

    print("Makespans")
    print(" exact:", solutions_exact[0].makespan)
    print(" evolution:", solutions_evolution[0].makespan)
    print(" annealing:", solutions_annealing[0].makespan)
    print("Runtime")
    print(" exact", solutions_exact[0].runtime)
    print(" evolution", solutions_evolution[0].runtime)
    print(" annealing", solutions_annealing[0].runtime)
    print("Fitness evaluations")
    print(" evolution", solutions_evolution[0].eval_log[-1][0])
    print(" annealing", solutions_annealing[0].eval_log[-1][0])

    plot_gantt_chart(solutions_exact[0].schedule, instances[0])
    plot_gantt_chart(solutions_evolution[0].schedule, instances[0])
    plot_gantt_chart(solutions_annealing[0].schedule, instances[0])

    f = plt.figure()
    ax = f.gca()
    plot_fitness_graph(solutions_evolution[0].eval_log, "Evolution", ax)
    plot_fitness_graph(solutions_annealing[0].eval_log, "Annealing", ax)
    plt.show(block=True)


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
