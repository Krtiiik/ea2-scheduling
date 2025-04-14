from collections import deque
import itertools
import random
from typing import Generator, Literal

import networkx as nx

from instances import ProblemInstance, Job


def build_instance_graph(instance, reverse: bool = False) -> nx.DiGraph:
    """
    Builds a job-graph of the given problem instance.

    Args:
        instance: The instance to build the graph of. Any object with `jobs` and `precedences` properties can be given.
        reverse: Indicates whether to reverse the precedence edges.

    Returns:
        The oriented job-graph of the problem instance.
    """
    def build_edge(precedence): return ((precedence.id_child, precedence.id_parent)
                                        if not reverse else (precedence.id_parent, precedence.id_child))
    graph = nx.DiGraph()
    graph.add_nodes_from(job.id_job for job in instance.jobs)
    graph.add_edges_from(build_edge(precedence) for precedence in instance.precedences)
    return graph


def enumerate_topological_generations_nodes(graph):
    """
    Enumerates the topological generations of a given graph.

    Args:
        graph (networkx.DiGraph): The graph whose topological generations to enumerate.

    Yields:
        tuple: A tuple containing the generation index and the node in the topological order.
    """
    for i_gen, gen in enumerate(nx.topological_generations(graph)):
        for node in gen:
            yield i_gen, node


def topological_sort(graph: nx.DiGraph,
                     yield_state: bool = False):
    """
    Traverses the nodes of the graph in topological order.

    Args:
        graph (nx.DiGraph): The graph whose nodes to traverse.
        yield_state (bool, optional): Determines whether to yield the parent node together with the current node.

    Yields:
        The nodes in the topological order. If `yield_state` is True, yields a tuple containing the current node and its parent node.
    """
    degrees = {v: d for v, d in graph.in_degree if d > 0}
    no_in = deque()
    for v, d in graph.in_degree:
        if d == 0:
            no_in.append((None, v))

    while no_in:
        parent, n = no_in.popleft()
        for _n_from, n_to in graph.edges(n):
            degrees[n_to] -= 1
            if degrees[n_to] == 0:
                no_in.append((n, n_to))
                del degrees[n_to]
        yield (n, parent) if yield_state else n


def uniform_traversal(graph: nx.DiGraph,
                      yield_state: bool = False):
    """
    Traverses the nodes of the graph in a uniformly random order.

    Args:
        graph (nx.DiGraph): The graph whose nodes to traverse.
        yield_state (bool, optional): Determines whether to yield the node as a tuple `(node, )` or just the object node.

    Yields:
        node (tuple or object): The next node in the traversal.
    """
    def pop(f):  # Selects and pops a uniformly chosen node
        i_node = random.randint(0, len(f) - 1)
        n_rand = f[i_node]
        f[i_node], f[-1] = f[-1], None
        f.pop()
        return n_rand

    first_generation = nx.topological_generations(graph).send(None)  # `send` will yield the first generation
    frontier = list(first_generation)
    while frontier:
        node = pop(frontier)
        yield (node,) if yield_state else node
        frontier += [e[1] for e in graph.out_edges(node)]


def paths_traversal(graph: nx.DiGraph):
    """
    Traverses random vertex-disjoint paths in the given graph.

    Args:
        graph: The graph in which to traverse the paths.

    Returns:
        A collection of paths using all graph nodes. All the paths are vertex-disjoint.
    """
    successors: dict[Job, set[Job]] = {node: set(graph.successors(node)) for node in graph.nodes if len(set(graph.successors(node))) > 0}
    in_degrees = {node: d for node, d in graph.in_degree if d > 0}

    no_in = deque()  # Queue, initially nodes with no incoming edges
    for node, d in graph.in_degree:
        if d == 0:
            no_in.append(node)

    paths = []
    visited = set()
    while no_in:
        node = no_in.popleft()
        path = [node]  # Start a new path with the current node
        visited |= {node}  # Add the current node to the visited set
        while node in successors:  # while there are successors for node...
            successors[node] -= visited  # remove visited nodes from successors
            if not successors[node]:  # if there are no unvisited successors...
                break  # path is complete
            old_node, node = node, random.choice(list(successors[node]))  # node is a random successor
            for successor in successors[old_node]:  # update successors in-degrees
                in_degrees[successor] -= 1
                if in_degrees[successor] == 0:
                    del in_degrees[successor]
                    no_in.append(successor)
            del successors[old_node]
            visited |= {node}
            path.append(node)  # add this new node to the path

        paths.append(path)

    return paths


def traverse_instance_graph(problem_instance: ProblemInstance = None,
                            graph: nx.DiGraph = None,
                            search: Literal["topological generations", "components topological generations", "topological", "uniform", "paths"] = "topological generations",
                            yield_state: bool = False) -> Generator[Job, None, None]:
    """
    Traverses the job-graph of a given problem instance, yielding jobs in the order of visiting.

    The available search type options are:
    - topological generations (default): Traverse the graph by the order of topological generations.
    - components topological generations: Traverse topologically each weakly-connected component before traversing the next component.
    - topological: Traverse the whole graph in topological order.
    - uniform: Traverse from the first (child) jobs of each connected component, choosing a random reachable parent job.
    - paths: Traverse random paths in the graph. The paths are vertex-disjoint, each vertex is yielded as part of a path.

    Args:
        problem_instance (ProblemInstance, optional): The problem instance whose job-graph to traverse.
        graph (nx.DiGraph, optional): The existing job-graph to traverse.
        search (str, optional): Determines the type of search to use for the traversal. Options are "topological generations" (default),
                "components topological generations", "topological", "uniform", "paths".
        yield_state (bool, optional): Determines whether a search state is yielded with each node. The yielded search state is
                                      (node, i_gen) for topological generations, (node, i_comp, i_gen) for components topological generations,
                                      (node, parent) for topological, (node, ) for uniform, (node, i_path) for paths.
    Returns:
        Each job from the instance graph in an order given by the specified search type.
    """
    if problem_instance is None and graph is None:
        print_error("Neither problem instance nor job-graph were given to traverse")

    if search not in ["topological generations", "components topological generations", "topological", "uniform", "paths"]:
        print_error(f"Unrecognized search kind: {search}")
        return

    if graph is None:
        graph = build_instance_graph(problem_instance)

    match search:
        case "topological generations":
            for i_gen, node in enumerate_topological_generations_nodes(graph):
                yield (node, i_gen) if yield_state else node
        case "components topological generations":
            for i_comp, component in enumerate(nx.weakly_connected_components(graph)):
                for i_gen, node in enumerate_topological_generations_nodes(graph.subgraph(component)):
                    yield (node, i_comp, i_gen) if yield_state else node
        case "topological":
            yield from topological_sort(graph, yield_state=yield_state)
        case "uniform":
            yield from uniform_traversal(graph, yield_state=yield_state)
        case "paths":
            for node, i_path in paths_traversal(graph):
                yield (node, i_path) if yield_state else node


def compute_jobs_in_components(problem_instance: ProblemInstance) -> dict[int, list[Job]]:
    """
    Computes the partition of job-nodes of the given problem instance into individual components.

    Args:
        problem_instance (ProblemInstance): The problem instance whose jobs to assign to components.

    Returns:
        dict[int, list[Job]]: A dictionary mapping root jobs to jobs in the same component.
    """
    jobs_grouped = itertools.groupby(traverse_instance_graph(problem_instance, search="components topological generations", yield_state=True),
                                     key=lambda x: x[1])
    component_jobs_by_root_job: dict[int, list[Job]] = dict()
    for _i_comp, job_states in jobs_grouped:
        jobs = [job for job, _, _ in job_states]
        for component in problem_instance.components:
            root_job = component.id_root_job
            if any(job.id_job == root_job for job in jobs):
                component_jobs_by_root_job[root_job] = jobs
                break
            else:
                print_error("No root job specified for an existing component")
                return {}
    return component_jobs_by_root_job


def subtree_traversal(graph: nx.DiGraph,
                      root: Job | int,
                      kind: Literal["bfs", "dfs"] = "bfs") -> list[Job]:
    """
    Traverses the subtree of the given graph rooted at the given node.

    Args:
        graph (nx.DiGraph): The graph whose subtree to traverse.
        root (Job or int): The root of the subtree to traverse.
        kind (Literal["bfs", "dfs"], optional): The kind of traversal to use. Options are "bfs" (default) and "dfs".

    Returns:
        list[Job]: The subtree of the graph rooted at the given node.
    """
    if kind not in ["bfs", "dfs"]:
        print_error(f"unrecognized subtree traversal kind {kind}")
        return []

    frontier = deque()
    put = frontier.append
    pop = frontier.popleft if kind == "bfs" else frontier.pop
    subtree = []

    put(root)
    while frontier:
        node = pop()
        subtree.append(node)
        for child in graph.successors(node):
            put(child)

    return subtree


def compute_earliest_completion_times(instance: ProblemInstance,
                                      given: dict[int, int] = None,
                                      ) -> dict[Job, int]:
    """
    Computes the earliest completion times for each job in the given problem instance.

    Args:
        instance (ProblemInstance): The problem instance containing the jobs.
        given (dict[int, int], optional): A dictionary of job IDs and their corresponding completion times,
                                          if any are already known. Defaults to None.

    Returns:
        dict[Job, int]: A dictionary mapping each job to its earliest completion time.
    """
    if not given:
        given = {}

    durations = {j.id_job: j.duration for j in instance.jobs}
    graph = build_instance_graph(instance)
    earliest_completion_times: dict[int, int] = dict()
    for node in nx.topological_sort(graph):
        if node in given:
            earliest_completion_times[node] = given[node]
        else:
            earliest_bound = max((earliest_completion_times[predecessor] for predecessor, _ in graph.in_edges(node)), default=0)
            earliest_completion_times[node] = earliest_bound + durations[node]

    return {j: earliest_completion_times[j.id_job] for j in instance.jobs}