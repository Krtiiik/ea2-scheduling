from dataclasses import dataclass
import math
import random

from evolution import SerialScheduleGenerationSchemeDecoder
from instances import ProblemInstance
from solvers import Solver

from deap_utils import generate_population

ActivityList = list[int]


class AnnealingSolver(Solver):
    """
    Simulated Annealing Solver for job scheduling problems.
    """
    def __init__(self, temp: float=100., cooling_rate=0.99, min_temp=1e-3):
        super().__init__()
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def solve(self, instance: ProblemInstance) -> tuple[ActivityList, int]:
        """
        Solve the problem using simulated annealing.
        """
        self.move = Move(instance)
        self.decoder = SerialScheduleGenerationSchemeDecoder()
        self.decoder.init(instance)
        best_state, best_fitness = simulated_annealing(
            initial_state=generate_population(instance, 1)[0],
            temp=self.temp,
            move_f=self.move,
            fitness=self.fitness,
            cooling_rate=self.cooling_rate,
            min_temp=self.min_temp
        )
        schedule, makespan = self.decoder(best_state)
        return schedule, makespan

    def fitness(self, state: ActivityList) -> int:
        """
        Calculate the fitness of a given state.
        """
        schedule, makespan = self.decoder(state)
        return makespan


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


def simulated_annealing(initial_state, temp, move_f, fitness, cooling_rate=0.99999, min_temp=1e-3):
    """
    Perform simulated annealing to optimize a given problem.

    :param initial_state: The starting state of the system.
    :param temp: Initial temperature.
    :param mutation: A function to generate a neighboring state.
    :param fitness: A function to evaluate the fitness of a state.
    :param cooling_rate: The rate at which the temperature decreases.
    :param min_temp: The minimum temperature to stop the algorithm.
    :return: The best state found and its fitness value.
    """
    current_state = initial_state
    current_fitness = fitness(current_state)
    best_state = current_state
    best_fitness = current_fitness

    while temp > min_temp:
        neighbor = move_f(current_state)
        neighbor_fitness = fitness(neighbor)
        print(f"best | neighbor: {best_fitness} | {neighbor_fitness}")

        delta_fitness = neighbor_fitness - current_fitness
        if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temp):
            current_state = neighbor
            current_fitness = neighbor_fitness

            if current_fitness > best_fitness:
                best_state = current_state
                best_fitness = current_fitness

        temp *= cooling_rate

    return best_state, best_fitness

# Example usage:
if __name__ == "__main__":
    # Define a sample fitness function
    def fitness_function(state):
        return -((state - 3) ** 2) + 10  # Example: Maximum at state = 3

    # Define a mutation function
    def mutate(state):
        return state + random.uniform(-1, 1)

    # Initial state
    initial = random.uniform(-10, 10)

    # Run simulated annealing
    best_state, best_fitness = simulated_annealing(
        initial_state=initial,
        temp=100,
        move_f=mutate,
        fitness=fitness_function
    )

    print(f"Best state: {best_state}, Best fitness: {best_fitness}")