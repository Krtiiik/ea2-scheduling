from collections import Counter, defaultdict
from dataclasses import dataclass
import random
from typing import Callable

from deap import base, creator, tools, algorithms

from instances import ProblemInstance, parse_psplib

@dataclass
class CrossOver:
    problem_instance: ProblemInstance
    fitness_func: Callable[[list], float]

    def __call__(self, p1, p2):
        candidates = []
        current = p1.copy()
        for pos in range(len(p1)):
            if current[pos] == p2[pos]:
                continue
            p2_pos_frajer_idx = current.index(p2[pos])
            current = current[:pos] + [current[p2_pos_frajer_idx]] + current[pos:p2_pos_frajer_idx] + current[p2_pos_frajer_idx + 1:]
            candidates.append(current)
        print(len(candidates) - 1)
        return self.choose_best(candidates[:-1])
    
    def choose_best(self, candidates):
        best = None
        best_score = float('inf')
        for c in candidates:
            score = self.fitness_func(c)
            if score < best_score:
                best = c
                best_score = score
        return best

    def construct_next_on_path(self, current, move_after_idx, jobs_to_move):
        return [x for x in current[:move_after_idx + 1] if x not in jobs_to_move]\
            + jobs_to_move + current[move_after_idx + 1:]


def generate_population(instance: ProblemInstance, population_size: int):
    initial_counter, precedence_graph = build_precedence_graph(instance)
    population = [
        generate_individual_genotype(initial_counter, precedence_graph)
        for _ in range(population_size)
    ]
    return population


def build_precedence_graph(instance: ProblemInstance):
    """
    Constructs a precedence graph and counts the number of times
    each job appears as a parent in the precedence relations.
    """
    counter = {}
    precedence_graph = defaultdict(list)

    for relation in instance.precedences:
        # Map each child to its parent (or add to list if multiple)
        precedence_graph[relation.id_child].append(relation.id_parent)

        for v in (relation.id_child, relation.id_parent):
            if v not in counter:
                counter[v] = 0

        counter[relation.id_parent] += 1

    return counter, precedence_graph


def generate_individual_genotype(initial_counter: dict, precedence_graph: dict):
    """
    Generates a single genotype (job sequence) satisfying the given precedence.
    """
    counter = initial_counter.copy()
    genotype = []
    # Jobs with no prerequisites (or not appearing as parents) are initially available.
    available_jobs = [job for job, count in counter.items() if count == 0]

    while available_jobs:
        chosen_index = random.randint(0, len(available_jobs) - 1)
        # Swap chosen job with the last to efficiently pop it.
        available_jobs[-1], available_jobs[chosen_index] = available_jobs[chosen_index], available_jobs[-1]
        chosen_job = available_jobs.pop()
        genotype.append(chosen_job)

        # After scheduling a job, update the counter for its connected jobs.
        for parent in precedence_graph.get(chosen_job, []):
            counter[parent] -= 1
            if counter[parent] == 0:
                available_jobs.append(parent)

    return genotype

            
instance = parse_psplib("data/j302_10.sm")
pop = generate_population(instance, 100)
from pprint import pprint
print(pop)

cx = CrossOver(instance, lambda x: 1)
for i in range(100):
    for j in range(i + 1, 100):
        cx(pop[i], pop[j])

# def eval_func(individual):
#     return sum(individual),  

# # Set up the DEAP framework
# creator.create("FitnessMin", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# toolbox.register("attr_float", random.uniform, 0, 10)
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# toolbox.register("evaluate", eval_func)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # Main execution
# if __name__ == "__main__":
#     random.seed(42)
#     population = toolbox.population(n=50)
#     ngen, cxpb, mutpb = 100, 0.7, 0.2

#     algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
#                         stats=None, halloffame=None, verbose=True)

#     # Print the best individual and its fitness
#     best_ind = tools.selBest(population, k=100)[0]
#     print("Best individual is:", best_ind)
#     print("Fitness:", best_ind.fitness.values)
