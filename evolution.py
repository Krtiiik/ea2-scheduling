from collections import defaultdict
from instances import ProblemInstance
from solvers import Schedule, Solver


ActivityList = list[int]  # List of activities


class SerialScheduleGenerationSchemeDecoder:
    def __init__(self):
        pass

    def __call__(self, individual: ActivityList, instance: ProblemInstance) -> Schedule:
        """
        Decode the individual into a schedule.
        Using the Serial Schedule Generation Scheme.
        """
        self.instance = instance
        self.durations = {job.job_id: job.duration for job in instance.jobs}
        self.capacities = {resource.id_resource: [resource.capacity] * instance.horizon
                           for resource in instance.resources}

        predecessors = defaultdict(list)
        for precedence in instance.precedences:
            predecessors[precedence.id_parent].append(precedence.id_child)

        schedule: Schedule = {}
        for j in individual:
            earliest_start = max((schedule[pred] + self.durations[pred] for pred in predecessors[j]),
                                 default=0)
            assert earliest_start < instance.horizon, f"Job {j} cannot start at {earliest_start}"
            possible_starts = self._possible_starts(j, earliest_start)
            assert possible_starts, f"No possible start times for job {j}"
            start = min(t for t in possible_starts)
            schedule[j] = start
            self._decrease_capacities(j, start)
        return schedule

    def _sufficient_capacities(self, _j: int, _start: int) -> list[bool]:
        """Compute times when the resources are sufficient for consumption of job _j."""
        _r_capacities = {}
        for _resource in self.instance.resources:
            _capacities = [True] * self.instance.horizon
            _consumption = self.instance.jobs_by_id[_j].resource_consumption[_resource]
            for _t in range(_start, self.instance.horizon):
                if _capacities[_t] and self.capacities[_resource][_t] < _consumption:
                    _capacities[_t] = False
            _r_capacities[_resource] = _capacities
        return _r_capacities

    def _capacities_overhead(self, _r_capacities: list[bool], _start: int) -> int:
        """Compute the overhead of the capacities."""
        _overhead = 0
        _overheads = [0] * self.instance.horizon
        for _t in range(self.instance.horizon-1, _start-1, -1):
            if _r_capacities[_t]:
                _overhead += 1
            else:
                _overhead = 0
            _overheads[_t] = _overhead
        return _overheads

    def _possible_starts(self, _j: int, _start: int) -> list[int]:
        """Compute the possible start times for job _j."""
        _overheads = self._capacities_overhead(self._sufficient_capacities(_j, _start), _start)
        return [_t for _t in range(_start, self.instance.horizon) if _overheads[_t] >= self.durations[_j]]

    def _decrease_capacities(self, _j: int, _start: int) -> None:
        """Decrease the capacities of the resources."""
        _consumption = self.instance.jobs_by_id[_j].resource_consumption
        for _t in range(_start, self.instance.horizon):
            for _resource in self.instance.resources:
                self.capacities[_resource][_t] -= _consumption[_resource]


class EvolutionSolver(Solver):
    def __init__(self):
        super().__init__()
        self._decoder = SerialScheduleGenerationSchemeDecoder()

    def solve(self, instance):
        """
        Solve the given instance using the evolutionary method.
        """
        # TODO
        ...


if __name__ == "__main__":
    # Example usage

    instance = ProblemInstance().precedences_by_id_child

    instance = ProblemInstance()  # Replace with actual instance creation
    solver = EvolutionSolver()
    schedule = solver.solve(instance)
    print(schedule)
