import argparse
import typing

from instances import Job, ProblemInstance
from enum import Enum


class Configuration(typing.TypedDict):
    time_limit: int  # for each instance, in seconds

class SolutionKind(Enum):
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"

Schedule = dict[Job|int, int]  # Job -> start time

class Solution(typing.TypedDict):
    kind: SolutionKind
    schedule: Schedule | None
    makespan: int | None


class Solver:
    def __init__(self,):
        pass

    def configure(self, config: Configuration, args: argparse.Namespace):
        """
        Configure the solver.
        """
        self._config = config
        self._args = args

    def solve(self, instance: ProblemInstance) -> Solution:
        """
        Solve the given instance.
        """
        pass

    def solve_all(self, instances: list[ProblemInstance]) -> list[Solution]:
        """
        Solve all instances.
        """
        print(f"Solving {len(instances)} instances")
        solutions = []
        for instance in instances:
            solution = self.solve(instance)
            solutions.append(solution)
        return solutions
