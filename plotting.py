from collections import namedtuple
import itertools

from matplotlib import pyplot as plt

from instances import ProblemInstance
from solvers import Schedule


Interval = namedtuple("Interval", ("key", "start", "end"))


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
