import math
import random
import time

from metaheuristic_repr import ActivityList, Move, SerialScheduleGenerationSchemeDecoder
from instances import ProblemInstance
from solvers import Solution, Solver

from deap_utils import generate_population


class AnnealingSolver(Solver):
    """
    Simulated Annealing Solver for job scheduling problems.
    """
    def __init__(self, temp: float=100., cooling_rate=0.99, min_temp=1e-3):
        super().__init__()
        self.temp = temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def _solve(self, instance: ProblemInstance) -> tuple[ActivityList, int]:
        """
        Solve the problem using simulated annealing.
        """
        self.move = Move(instance)
        self.decoder = SerialScheduleGenerationSchemeDecoder()
        self.decoder.init(instance)
        best_state, best_fitness, eval_log = simulated_annealing(
            initial_state=generate_population(instance, 1)[0],
            temp=self.temp,
            move_f=self.move,
            fitness=self.fitness,
            cooling_rate=self.cooling_rate,
            min_temp=self.min_temp,
            time_limit=self._config.time_limit
        )
        schedule, makespan = self.decoder(best_state)
        return Solution(schedule=schedule, makespan=makespan, eval_log=eval_log)

    def fitness(self, state: ActivityList) -> int:
        """
        Calculate the fitness of a given state.
        """
        schedule, makespan = self.decoder(state)
        return makespan


def simulated_annealing(initial_state, temp, move_f, fitness, cooling_rate=0.99999, min_temp=1e-3, time_limit=10):
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
    eval_log = []
    fitness_eval_count = 0

    current_state = initial_state
    current_fitness = fitness(current_state); fitness_eval_count += 1
    best_state = current_state
    best_fitness = current_fitness
    eval_log.append((fitness_eval_count, best_fitness))

    end_time = time.time() + time_limit
    while temp > min_temp and time.time() < end_time:
        neighbor = move_f(current_state)
        neighbor_fitness = fitness(neighbor); fitness_eval_count += 1
        # print(f"best | neighbor: {best_fitness} | {neighbor_fitness}")

        delta_fitness = current_fitness - neighbor_fitness
        if delta_fitness > 0 or random.random() < math.exp(delta_fitness / temp):
            current_state = neighbor
            current_fitness = neighbor_fitness

            if current_fitness < best_fitness:
                best_state = current_state
                best_fitness = current_fitness

        eval_log.append((fitness_eval_count, best_fitness))
        temp *= cooling_rate

    return best_state, best_fitness, eval_log

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