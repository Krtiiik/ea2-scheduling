from collections import defaultdict
import random
from multiprocessing import Pool

import numpy as np

from instances import ProblemInstance, Resource
from plotting import plot_gantt_chart
from solvers import Schedule, Solver
import deap_utils as du


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
            job.id_job: {resource.id_resource: job.resource_consumption.consumption_by_resource.get(resource.id_resource, 0)
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
        overheads = [max(overheads[r][t] for r in self.instance.resources) for t in range(self.instance.horizon)]
        return [t for t in range(start, self.instance.horizon) if overheads[t] >= self.durations[j]]

    def _sufficient_capacities(self, j: int) -> dict[Resource, np.ndarray[bool]]:
        """Compute times when the resources are sufficient for consumption of job j."""
        r_capacities = {}
        for resource in self.instance.resources:
            consumption = self.consumptions[j][resource.id_resource]
            if consumption == 0:
                r_capacities[resource] = np.full(self.instance.horizon, True, dtype=bool)
                continue
            capacities = self.capacities[resource] < consumption
            r_capacities[resource] = capacities
        return r_capacities

        # r_capacities = {}
        # for resource in self.instance.resources:
        #     capacities = [True] * self.instance.horizon
        #     consumption = self.instance.jobs_by_id[j].resource_consumption.consumption_by_resource.get(resource.id_resource, 0)
        #     if consumption == 0:
        #         r_capacities[resource] = capacities
        #         continue
        #     for t in range(start, self.instance.horizon):
        #         if capacities[t] and self.capacities[resource][t] < consumption:
        #             capacities[t] = False
        #     r_capacities[resource] = capacities
        # # self._debug_print(f"  Resource capacities\n\t{r_capacities}")
        # return r_capacities

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
        for resource in self.instance.resources:
            self.capacities[resource][start:] = self.capacities[resource][start:] - self.consumptions[j][resource.id_resource]

    def _debug_print(self, message: str) -> None:
        """Print debug messages."""
        if self._debug:
            print(message)


EVO_SETTINGS = {
    "population_size": 10,
    "max_gen": 5,
    "tournament_size": 3,
    "candidates_size": 5
}


class EvolutionSolver(Solver):
    def __init__(self):
        super().__init__()
        self._decoder = SerialScheduleGenerationSchemeDecoder()

    def solve(self, instance):
        """
        Solve the given instance using the evolutionary method.
        """
        self._decoder.init(instance)
        pop = du.generate_population(instance, EVO_SETTINGS["population_size"])
        # _cx = du.CrossOverPaper(lambda ind: self._fitness(ind, instance))
        _cx = du.CrossoverMultipleCandidates(max_candidates=EVO_SETTINGS["candidates_size"])
        def cx(mating_pool):
            random.shuffle(mating_pool)
            return mating_pool + [ind
                                  for ind1, ind2 in zip(mating_pool[::2], mating_pool[1::2])
                                  for ind in _cx(ind1, ind2)
                                  ]
        # mut = du.Mutation(self._fitness)
        def select(pop, fits): return [self._select_tournament(pop, fits) for _ in range(EVO_SETTINGS["population_size"])]

        log = []
        print("EVO")
        for gen in range(EVO_SETTINGS["max_gen"]):
            print(f'{gen+1}/{EVO_SETTINGS["max_gen"]}', end='\r')
            fits = [self._fitness(ind, instance) for ind in pop]
            with Pool() as pool:
                fits = pool.starmap(self._fitness, [(ind, instance) for ind in pop])
            log.append(min(fits))
            mating_pool = select(pop, fits)
            off = cx(mating_pool)
            # off = mutation(off)
            off[0] = min(pop, key=lambda ind: self._fitness(ind, instance))
            pop = off[:]

        return pop, log

    def _select_tournament(self, population: list[ActivityList], fitnesses: list[int]) -> ActivityList:
        candidates_ids = random.sample(range(EVO_SETTINGS["population_size"]), EVO_SETTINGS["tournament_size"])
        candidates = [population[i] for i in candidates_ids]
        candidates_fitnesses = [fitnesses[i] for i in candidates_ids]
        best_i = min(((fitn, i) for i, fitn in enumerate(candidates_fitnesses)), key=lambda x: x[0])[1]
        return candidates[best_i]

    def _fitness(self, individual: ActivityList, instance) -> int:
        """
        Compute the fitness of the individual.
        """
        schedule, makespan = self._decoder(individual)
        return makespan


if __name__ == "__main__":
    from instances import Job, Resource, ResourceType
    
    activity_list = [0, 1, 2, 3]
    instance = ProblemInstance(
        jobs=[
            Job(id_job=0, duration=3, resource_consumption={0: 2}),
            Job(id_job=1, duration=2, resource_consumption={0: 1}),
            Job(id_job=2, duration=4, resource_consumption={0: 3}),
            Job(id_job=3, duration=1, resource_consumption={0: 1})
        ],
        resources=[
            Resource(id_resource=0, resource_type=ResourceType.RENEWABLE, capacity=3)
        ],
        precedences=[],
        horizon=10
    )

    decoder = SerialScheduleGenerationSchemeDecoder()
    decoder.init(instance)
    schedule = decoder(activity_list)
    print(schedule)
    schedule = {instance.jobs_by_id[j]: start for j, start in schedule.items()}
    plot_gantt_chart(schedule, instance)
