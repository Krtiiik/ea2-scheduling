import typing
from evolution import SerialScheduleGenerationSchemeDecoder
from solvers import SolutionKind, Solver, Solution

from ortools.sat.python import cp_model


TVARIABLES = typing.TypedDict("TVariables", {
    "makespan": cp_model.IntVar,
    "job_vars": dict[int, dict[str, cp_model.IntVar]],
})


class CPSolver(Solver):
    def __init__(self):
        super().__init__()
        self._solver = cp_model.CpSolver()

    def solve(self, instance):
        """
        Solve the given instance using the exact method.
        """
        model, variables = self._build_model(instance)
        status = self._solver.solve(model)

        return Solution(
            kind={
                cp_model.OPTIMAL: SolutionKind.OPTIMAL,
                cp_model.FEASIBLE: SolutionKind.FEASIBLE,
                cp_model.INFEASIBLE: SolutionKind.INFEASIBLE,
            }[status],
            schedule={
                job: self._solver.value(variables["job_vars"][job.id_job]["start"])
                for job in instance.jobs
            },
            makespan=self._solver.value(variables["makespan"]),
        )

    def _build_model(self, instance) -> tuple[cp_model.CpModel, TVARIABLES]:
        """
        Build the model for the given instance.
        """
        horizon = instance.horizon

        model = cp_model.CpModel()

        job_vars = {}
        for job in instance.jobs:
            duration = job.duration
            suffix = f'_{job.id_job}'
            start_var = model.new_int_var(0, horizon, "start" + suffix)
            end_var = model.new_int_var(0, horizon, "end" + suffix)
            interval_var = model.new_interval_var(
                start_var, duration, end_var, "interval" + suffix
            )
            job_vars[job.id_job] = {
                "start": start_var,
                "end": end_var,
                "interval": interval_var,
            }

        for precedence in instance.precedences:
            child = precedence.id_child
            parent = precedence.id_parent
            end_child = job_vars[child]["end"]
            start_parent = job_vars[parent]["start"]
            model.add(end_child <= start_parent)

        for resource in instance.resources:
            model.AddCumulative(
                intervals=[job_vars[job.id_job]["interval"] for job in instance.jobs],
                demands=[job.resource_consumption[resource] for job in instance.jobs],
                capacity=resource.capacity,
            )

        makespan = model.new_int_var(0, horizon, "makespan")
        model.add_max_equality(
            makespan,
            [job_vars[job.id_job]['end'] for job in instance.jobs],
        )
        model.minimize(makespan)

        variables = TVARIABLES(makespan=makespan, job_vars=job_vars)
        return model, variables
