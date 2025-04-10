import argparse
from dataclasses import dataclass
import typing
import time

from instances import Job, ProblemInstance
from enum import Enum


@dataclass
class Configuration:
    time_limit: int # for each instance, in seconds


Schedule = dict[Job|int, int]  # Job -> start time


@dataclass
class Solution:
    schedule: Schedule | None
    makespan: int | None
    runtime: float = 0


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
        t_start = time.time()
        solution = self._solve(instance)
        t_end = time.time()
        runtime = t_end - t_start
        solution.runtime = runtime
        return solution

    def _solve(self, instance: ProblemInstance) -> Solution:
        raise NotImplementedError()

    def solve_all(self, instances: list[ProblemInstance]) -> list[Solution]:
        """
        Solve all instances.
        """
        print(f"Solving {len(instances)} instances")
        solutions = []
        for i, instance in enumerate(instances):
            print(f'{i + 1}/{len(instances)}: {instance.name}', end='\r')
            solution = self.solve(instance)
            solutions.append(solution)
        print("Done"+" "*10)
        return solutions
