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
from plotting import plot_fitnesses, plot_fitnesses_graph, plot_gantt_chart, plot_fitness_graph, plot_comparison, plot_makespans, plot_runtimes
import solvers
import csv


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
    with open(os.path.join(RESULTS_DIR, "instances.pkl"), "rb") as f:
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

    from instances_drawing import plot_instance_graph
    plot_instance_graph(instances[0])
    plot_instance_graph(instances[27])

    save_results_table(instances, solutions_exact, solutions_evolution, solutions_annealing)
    plot_comparison(instances[:20], solutions_exact[:20], solutions_evolution[:20], solutions_annealing[:20], save_as=os.path.join(RESULTS_DIR, "comparisson1.pdf"))
    plot_comparison(instances[20:], solutions_exact[20:], solutions_evolution[20:], solutions_annealing[20:], save_as=os.path.join(RESULTS_DIR, "comparisson2.pdf"))
    plot_makespans(instances, solutions_exact, solutions_evolution, solutions_annealing, save_as=os.path.join(RESULTS_DIR, "makespans.pdf"))
    plot_runtimes(instances, solutions_exact, solutions_evolution, solutions_annealing, save_as=os.path.join(RESULTS_DIR, "runtimes.pdf"))
    # plot_fitnesses(
    #     instances,
    #     [sol.eval_log for sol in solutions_evolution],
    #     [sol.eval_log for sol in solutions_annealing],
    # )

    instances_names: list = [inst.name for inst in instances]

    fun_instances = [instances_names.index(name) for name in ["j3046_6.sm", "j1202_6.sm", "j6025_9.sm", "j601_4.sm"]]
    fun_logs = [(solutions_evolution[i].eval_log, solutions_annealing[i].eval_log) for i in fun_instances]
    fun_instances = [instances[i] for i in fun_instances]
    for (inst, (log_evo, log_ann)) in zip(fun_instances, fun_logs):
        plot_fitnesses_graph(log_evo, log_ann, inst.name[:-3], save_as=os.path.join(RESULTS_DIR, inst.name[:-3]+'.pdf'))


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
    with open(RESULTS["table"], 'wt', newline='') as f:
        # f.write(tabulate.tabulate(
        #     sorted(
        #         [{
        #             "instance": instance.name[:-3],
        #             "makespan_exact": sol_exact.makespan,
        #             "makespan_evolution": sol_evolution.makespan,
        #             "makespan_annealing": sol_annealing.makespan,
        #             "runtime_exact": sol_exact.runtime,
        #             "runtime_evolution": sol_evolution.runtime,
        #             "runtime_annealing": sol_annealing.runtime,
        #         } for instance, sol_exact, sol_evolution, sol_annealing
        #           in zip(instances, solutions_exact, solutions_evolution, solutions_annealing)],
        #         key=lambda x: instance_size_param(x["instance"]),
        #     ),
        #     headers="keys",
        # ))
        writer = csv.DictWriter(
            f,
            fieldnames=[
            "instance",
            "makespan_exact",
            "makespan_evolution",
            "makespan_annealing",
            "runtime_exact",
            "runtime_evolution",
            "runtime_annealing",
            ],
        )
        writer.writeheader()
        writer.writerows(
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
            )
        )


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
