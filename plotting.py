from collections import namedtuple
import itertools

from matplotlib import pyplot as plt
import numpy as np

from instances import ProblemInstance
from solvers import Schedule, Solution


Interval = namedtuple("Interval", ("key", "start", "end"))


def plot_comparison(instances: list[ProblemInstance], solutions_exact: list[Solution], solution_evolution: list[Solution], solutions_annealing: list[Solution]):
    from scipy.stats import rankdata
    from matplotlib.colors import ListedColormap

    makespans = np.array([
        (sol_ex.makespan, sol_evo.makespan, sol_ann.makespan)
        for sol_ex, sol_evo, sol_ann in zip(solutions_exact, solution_evolution, solutions_annealing)
        ])
    order = rankdata(makespans, method='min', axis=1)

    cmap = ListedColormap(["green", "orange", "red"])

    f = plt.figure(figsize=(10, 3))
    ax = f.gca()
    ax.scatter(
        x=np.tile(np.arange(len(instances)), 3).flatten(),
        y=np.tile(np.array([2,1,0]), (len(instances), 1)).flatten(),
        c=order.flatten(), cmap=cmap,
        )

    ax.set_yticks([2, 1, 0], ["Exact", "Evolution", "Annealing"])
    ax.set_xticks(np.arange(len(instances)), [inst.name for inst in instances], rotation=90)
    for pos in ['top', 'right', 'left', 'bottom']:
        ax.spines[pos].set_visible(False)
    ax.set_ylim(-0.5, 2.5)
    # ax.axis("tight")
    f.subplots_adjust(top=0.8, bottom=0.5)
    # f.set_size_inches()
    # f.tight_layout()

    plt.show(block=True)


def plot_makespans(instances: list[ProblemInstance], solutions_exact: list[Solution], solution_evolution: list[Solution], solutions_annealing: list[Solution]):
    makespans = np.array([
        (sol_ex.makespan, sol_evo.makespan, sol_ann.makespan)
        for sol_ex, sol_evo, sol_ann in zip(solutions_exact, solution_evolution, solutions_annealing)
        ])

    f = plt.figure(figsize=(10, 5))
    ax = f.gca()

    ax.vlines(np.arange(len(instances)), 0, 200, colors="gray", zorder=-1, linewidth=0.5)
    ax.scatter(np.arange(len(instances)), makespans[:, 0], marker='o', label="Exact")
    ax.scatter(np.arange(len(instances)), makespans[:, 1], marker='D', label="Evolution")
    ax.scatter(np.arange(len(instances)), makespans[:, 2], marker='x', label="Annealing")

    ax.set_ylim(makespans.min() - 10, makespans.max() + 10)
    ax.set_xticks(np.arange(len(instances)), [inst.name for inst in instances], rotation=90)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    f.subplots_adjust(top=0.9, bottom=0.3)

    plt.show(block=True)


def plot_fitnesses(eval_logs_evo, eval_logs_ann):
    for i in range(0, len(eval_logs_evo), 4):
        f, ax = plt.subplots(2, 2, figsize=(10, 10))
        for i, (log_evo, log_ann) in enumerate(zip(eval_logs_evo[i:i+4], eval_logs_ann[i:i+4])):
            ax = plt.subplot(2, 2, 1+i)
            plot_fitnesses_graph(log_evo, log_ann, ax)
        plt.show(block=True)


def plot_fitnesses_graph(eval_logs_evo, eval_logs_ann, ax=None):
    show = False
    if ax is None:
        show = True
        f = plt.figure()
        ax = f.gca()
    plot_fitness_graph(eval_logs_evo, "Evolution", ax)
    plot_fitness_graph(eval_logs_ann, "Annealing", ax)

    if show:
        plt.show(block=True)


def plot_fitness_graph(eval_logs, label="Fitness", ax=None):
    show = False
    if ax is None:
        show = True
        _, ax = plt.subplots(figsize=(10, 6))

    evals, fitnesses = zip(*eval_logs)
    ax.plot(evals, fitnesses, marker='o', linestyle='-', label=label)
    ax.set_title("Fitness over Evaluations")
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Fitness")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    if show:
        plt.show(block=True)


def plot_gantt_chart(schedule: Schedule, instance: ProblemInstance) -> None:
    f = plt.figure(figsize=(10, 6))
    f.suptitle(f"Gantt Chart")
    ax = f.add_subplot(111)
    ax.set_xlabel("Time")
    ax.set_ylabel("Jobs")
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    job_starts = [schedule[job.id_job] for job in instance.jobs]
    intervals = [Interval(key=job.id_job, start=start, end=start + job.duration)
                 for job, start in zip(instance.jobs, job_starts)]
    interval_levels, max_level = compute_interval_levels(intervals)

    interval_width = 16
    scale = 1.
    cmap = plt.get_cmap("tab20")
    for key, start, end in intervals:
        level = interval_levels[key]

        ax.plot([start, end], [scale*level, scale*level], linestyle='', marker='|', markeredgecolor='gray', markersize=interval_width)
        color = cmap(2*key % 20)
        ax.hlines(scale*level, start, end, colors=color, lw=interval_width)
        ax.text(float(start + end) / 2, scale*level, str(key), horizontalalignment='center', verticalalignment='center')

    plt.show()

def compute_interval_levels(intervals: list[Interval], max_level: int = None) -> tuple[dict[Interval, int], int]:
    import heapq

    intervals = sorted(intervals, key=lambda i: (i.start, i.end))
    if max_level is None:
        max_level = compute_max_interval_overlap(intervals)

    h = [(intervals[0].start, lvl) for lvl in range(0, max_level)]
    heapq.heapify(h)

    levels = dict()
    for itv in intervals:
        lvl = heapq.heappop(h)[1]
        levels[itv.key] = lvl
        heapq.heappush(h, (itv.end, lvl))

    return {k: max_level - l for k, l in levels.items()}, max_level

def compute_max_interval_overlap(intervals):
    events = sorted([(itv.start, +1) for itv in intervals] + [(itv.end, -1) for itv in intervals])
    return max(itertools.accumulate(events, lambda cur, event: cur + event[1], initial=0))
