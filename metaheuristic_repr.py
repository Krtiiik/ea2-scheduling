import random
from instances import ProblemInstance, Resource
from solvers import Schedule


import numpy as np


from collections import defaultdict
ActivityList = list[int]  # List of activities


class SerialScheduleGenerationSchemeDecoder:
    def __init__(self):
        self._debug = False
        self.instance: ProblemInstance = None
        pass

    def init(self, instance: ProblemInstance) -> None:
        self.instance = instance

        self.predecessors = defaultdict(list)
        for precedence in instance.precedences:
            self.predecessors[precedence.id_parent].append(precedence.id_child)

        self.durations = {job.id_job: job.duration for job in instance.jobs}
        self.consumptions = {
            job.id_job: {resource.id_resource: job.resource_consumption[resource]
                         for resource in instance.resources}
            for job in instance.jobs
        }

    def __call__(self, individual: ActivityList) -> tuple[Schedule, int]:
        """
        Decode the individual into a schedule.
        Using the Serial Schedule Generation Scheme.
        """
        instance = self.instance
        self.capacities = {resource: np.full(instance.horizon, resource.capacity)
                           for resource in instance.resources}
        predecessors = self.predecessors

        schedule: Schedule = {}
        for j in individual:
            # self._debug_print(f"Processing job {j}")
            earliest_start = max((schedule[pred] + self.durations[pred] for pred in predecessors[j]),
                                 default=0)
            # self._debug_print(f"  Earliest start time: {earliest_start}")
            assert earliest_start < instance.horizon, f"Job {j} cannot start at {earliest_start}"
            possible_starts = self._possible_starts(j, earliest_start)
            # self._debug_print(f"  Possible start times: {possible_starts}")
            assert possible_starts, f"No possible start times for job {j}"
            start = min(t for t in possible_starts)
            # self._debug_print(f"  Selected start time: {start}")
            schedule[j] = start
            self._decrease_capacities(j, start)
            # self._debug_print(f'  Updated capacities: {self.capacities}')

        makespan = max(start + self.durations[j] for j, start in schedule.items())
        return schedule, makespan

    def _possible_starts(self, j: int, start: int) -> list[int]:
        """Compute the possible start times for job j."""
        overheads = self._capacities_overhead(self._sufficient_capacities(j), start)
        overheads = [min(overheads[r][t] for r in self.instance.resources) for t in range(self.instance.horizon)]
        whatever = [t for t in range(start, self.instance.horizon) if overheads[t] >= self.durations[j]]
        return whatever

    def _sufficient_capacities(self, j: int) -> dict[Resource, np.ndarray[bool]]:
        """Compute times when the resources are sufficient for consumption of job j."""
        r_capacities = {}
        for resource in self.instance.resources:
            consumption = self.consumptions[j][resource.id_resource]
            if consumption == 0:
                r_capacities[resource] = np.full(self.instance.horizon, True, dtype=bool)
                continue
            capacities = self.capacities[resource] >= consumption
            r_capacities[resource] = capacities
        return r_capacities


    def _capacities_overhead(self, r_capacities: dict[Resource, np.ndarray[bool]], _start: int) -> int:
        """Compute the overhead of the capacities."""
        r_overheads = {}
        overhead = 0
        for _resource in self.instance.resources:
            overheads = np.zeros(self.instance.horizon, dtype=int)
            for t in range(self.instance.horizon-1, _start-1, -1):
                if r_capacities[_resource][t]:
                    overhead += 1
                else:
                    overhead = 0
                overheads[t] = overhead
            r_overheads[_resource] = overheads
        # self._debug_print(f"  Resource overheads\n\t{r_overheads}")
        return r_overheads

    def _decrease_capacities(self, j: int, start: int) -> None:
        """Decrease the capacities of the resources."""
        duration = self.durations[j]
        for resource in self.instance.resources:
            self.capacities[resource][start:start+duration] = self.capacities[resource][start:start+duration] - self.consumptions[j][resource.id_resource]

    def _debug_print(self, message: str) -> None:
        """Print debug messages."""
        if self._debug:
            print(message)


class Move:
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        # self.swappable_pairs = set()
        # V = set(job.id_job for job in instance.jobs)
        # for job in instance.jobs:
        #     X = V - ({job.id_job} | instance.predecessors_closure[job.id_job] | instance.successors_closure[job.id_job])
        #     for x in X:
        #         self.swappable_pairs.add(tuple(sorted((job.id_job, x))))
        # self.swappable_pairs = list(self.swappable_pairs)

    def __call__(self, state: ActivityList) -> ActivityList:
        """
        Generate a neighboring state by swapping two random jobs in the schedule.
        """
        new_state = state.copy()
        success = False
        while not success:
            success = True
            idx1, idx2 = random.sample(range(len(new_state)), 2)
            i, j = min(idx1, idx2), max(idx1, idx2)
            ni, nj = new_state[i], new_state[j]
            sni, pnj = self.instance.successors_closure[ni], self.instance.predecessors_closure[nj]
            for k in range(i, j+1):
                nk =  new_state[k]
                if nk in sni or nk in pnj:
                    success = False
                    break
        # print(f"Swapping pos {i} and {j}: {new_state[i]} and {new_state[j]}")
        new_state[i], new_state[j] = new_state[j], new_state[i]
        return new_state