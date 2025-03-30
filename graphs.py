import networkx as nx


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


def compute_descendants(instance_or_graph) -> dict[int, set[int]]:
    """
    Computes the descendants of each job in the given instance or graph.

    Args:
        instance_or_graph: Either a problem instance (with `jobs` and `precedences` properties) or a pre-built graph.

    Returns:
        A dictionary mapping each job ID to a set of its descendants.
    """
    if isinstance(instance_or_graph, nx.DiGraph):
        graph = instance_or_graph
    else:
        graph = build_instance_graph(instance_or_graph, reverse=True)

    return {node: set(graph.successors(node)) for node in graph.nodes}


def compute_ancestors(instance_or_graph) -> dict[int, set[int]]:
    """
    Computes the ancestors of each job in the given instance or graph.

    Args:
        instance_or_graph: Either a problem instance (with `jobs` and `precedences` properties) or a pre-built graph.

    Returns:
        A dictionary mapping each job ID to a set of its ancestors.
    """
    if isinstance(instance_or_graph, nx.DiGraph):
        graph = instance_or_graph
    else:
        graph = build_instance_graph(instance_or_graph)

    return {node: set(graph.predecessors(node)) for node in graph.nodes}
