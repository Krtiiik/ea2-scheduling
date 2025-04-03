from dataclasses import dataclass
import math
import random

from instances import ProblemInstance

ActivityList = list[int]

class Move:
    def __init__(self, instance: ProblemInstance):
        self.instance = instance
        self.swappable_pairs = set()
        V = set(job.id_job for job in instance.jobs)
        for job in instance.jobs:
            X = V - ({job.id_job} | instance.predecessors_closure[job.id_job] | instance.successors_closure[job.id_job])
            for x in X:
                self.swappable_pairs.add(tuple(sorted((job.id_job, x))))
        
    def __call__(self, state: ActivityList) -> ActivityList:
        """
        Generate a neighboring state by swapping two random jobs in the schedule.
        """
        new_state = state.copy()
        i, j = random.sample(self.swappable_pairs, 1)
        idx_i, idx_j = state.index(i), state.index(j)
        new_state[idx_i], new_state[idx_j] = new_state[idx_j], new_state[idx_i]
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