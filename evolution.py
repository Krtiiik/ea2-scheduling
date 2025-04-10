import random
from multiprocessing import Pool
import time


from metaheuristic_repr import ActivityList, SerialScheduleGenerationSchemeDecoder
from instances import ProblemInstance, Resource, ResourceConsumption
from plotting import plot_gantt_chart
from solvers import Solution, Solver
import deap_utils as du


EVO_SETTINGS = {
    "population_size": 20,
    "max_gen": 50,
    "tournament_size": 3,
    "candidates_size": 5
}


class EvolutionSolver(Solver):
    def __init__(self):
        super().__init__()
        self._decoder = SerialScheduleGenerationSchemeDecoder()

    def _solve(self, instance):
        """
        Solve the given instance using the evolutionary method.
        """
        self._decoder.init(instance)
        eval_log = []
        pop = du.generate_population(instance, EVO_SETTINGS["population_size"])
        # _cx = du.CrossOverPaper(lambda ind: self._fitness(ind, instance))
        _cx = du.CrossoverMultipleCandidates(max_candidates=EVO_SETTINGS["candidates_size"])
        def cx(mating_pool):
            random.shuffle(mating_pool)
            with Pool() as pool:
                offspring = pool.starmap(_cx, zip(mating_pool[::2], mating_pool[1::2]))
                return mating_pool + [ind for pair in offspring for ind in pair]

        # mut = du.Mutation(self._fitness)
        def select(pop, fits): return [self._select_tournament(pop, fits) for _ in range(EVO_SETTINGS["population_size"])]

        log = []; fitness_eval_count = 0
        end_time = time.time() + self._config.time_limit

        with Pool() as pool:
            fits = pool.starmap(self._fitness, [(ind, instance) for ind in pop]); fitness_eval_count += len(fits)
        eval_log.append((fitness_eval_count, fits[argmin(fits)]))

        for gen in range(EVO_SETTINGS["max_gen"]):
            if time.time() > end_time:
                best_ind = min(log, key=lambda ind_fit: ind_fit[1])[0]
                best_schedule, makespan = self._decoder(best_ind)
                return Solution(schedule=best_schedule, makespan=makespan, eval_log=eval_log)

            mating_pool = select(pop, fits)
            off = cx(mating_pool)
            # off = mutation(off)

            with Pool() as pool:
                fits_off = pool.starmap(self._fitness, [(ind, instance) for ind in off]); fitness_eval_count += len(fits)

            # elitism
            best_in_pop = argmin(fits)
            eval_log.append((fitness_eval_count, fits[best_in_pop]))
            off[0] = pop[best_in_pop][:]
            fits_off[0] = fits[best_in_pop]
            log.append((off[0], fits_off[0]))

            pop = off[:]
            fits = fits_off

        best_i = argmin(fits)
        best_ind = pop[best_i]
        best_schedule, makespan = self._decoder(best_ind)
        return Solution(schedule=best_schedule, makespan=makespan, eval_log=eval_log)

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


def argmin(iter, return_x=False):
    best_i = None
    best = None
    for i, x in enumerate(iter):
        if best is None or x < best:
            best = x
            best_i = i
    return best_i if not return_x else (best_i, best)


if __name__ == "__main__":
    from instances import Job, Resource, ResourceType

    activity_list = [0, 1, 2, 3]
    instance = ProblemInstance(
        jobs=[
            Job(id_job=0, duration=3, resource_consumption=ResourceConsumption({0: 2})),
            Job(id_job=1, duration=2, resource_consumption=ResourceConsumption({0: 1})),
            Job(id_job=2, duration=4, resource_consumption=ResourceConsumption({0: 3})),
            Job(id_job=3, duration=1, resource_consumption=ResourceConsumption({0: 1}))
        ],
        resources=[
            Resource(id_resource=0, resource_type=ResourceType.RENEWABLE, capacity=3)
        ],
        precedences=[],
        horizon=10
    )

    decoder = SerialScheduleGenerationSchemeDecoder()
    decoder.init(instance)
    schedule, makespane = decoder(activity_list)
    print(schedule)
    schedule = {instance.jobs_by_id[j]: start for j, start in schedule.items()}
    plot_gantt_chart(schedule, instance)
