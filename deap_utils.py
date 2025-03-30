from collections import Counter, defaultdict
import random

from deap import base, creator, tools, algorithms

from instances import ProblemInstance

class CrossOver:
    def __call__(self, p1, p2, problem, fitness_func):
        current = p1.copy()
        candidates = []
        j_in_current = 0
        while current != p2:
            # try next index
            j_in_current = (j_in_current + 1) % len(current)
            j_in_p2 = p2.index(current[j_in_current])
            Bj = set(current[:j_in_current])
            j_succeros = problem.successors(current[j_in_current])
            # Now we try to check whether B'(j) is a subset of B(j)
            for subset_breaker in p2[:j_in_p2]: 
                if subset_breaker in Bj:
                    continue
                # J and all its successors must be moved after subset_breaker
                move_after_idx = current.index(subset_breaker)
                jobs_to_move = [current[j_in_current]] + [x for x in current[:move_after_idx] if x in j_succeros]
                
                if len(candidates) > 0 or current != p1: # Makes sure we don't put p1 in the candidates
                    candidates.append(current.copy())
                
                current = self.construct_next_on_path(current, move_after_idx, jobs_to_move)
                break
        
        return self.choose_best(candidates[1:], fitness_func)
    
    def choose_best(candidates, fitness_func):
        best = None
        best_score = float('inf')
        for c in candidates:
            score = fitness_func(c)
            if score < best_score:
                best = c
                best_score = score
        return best

    def construct_next_on_path(self, current, move_after_idx, jobs_to_move):
        return [x for x in current[:move_after_idx] if x not in jobs_to_move]\
            + jobs_to_move + current[move_after_idx:]


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
    counter = defaultdict(int)
    precedence_graph = defaultdict(list)

    for relation in instance.precedences:
        # Map each child to its parent (or add to list if multiple)
        precedence_graph[relation.id_child].append(relation.id_parent)
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

            
        

def eval_func(individual):
    return sum(individual),  

# Set up the DEAP framework
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_func)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Main execution
if __name__ == "__main__":
    random.seed(42)
    population = toolbox.population(n=50)
    ngen, cxpb, mutpb = 100, 0.7, 0.2

    algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, 
                        stats=None, halloffame=None, verbose=True)

    # Print the best individual and its fitness
    best_ind = tools.selBest(population, k=100)[0]
    print("Best individual is:", best_ind)
    print("Fitness:", best_ind.fitness.values)
