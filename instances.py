import os
import sys
from typing import IO, Iterable, Optional, Collection, Self, Sequence, TypeVar
from collections import defaultdict
from enum import StrEnum

from graphs import compute_ancestors, compute_descendants


T = TypeVar('T')


# ~~~~~~~ ResourceType ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceType(StrEnum):
    """
    Enumeration representing the type of a resource.

    Attributes:
        RENEWABLE (str): Represents a renewable resource.
        NONRENEWABLE (str): Represents a non-renewable resource.
        DOUBLY_CONSTRAINED (str): Represents a doubly constrained resource.
    """
    RENEWABLE = 'R'
    NONRENEWABLE = 'N'
    DOUBLY_CONSTRAINED = 'D'


# ~~~~~~~ Resource ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Resource:
    """
    Represents a resource in the problem instance.
    """

    _id_resource: int
    _type: ResourceType
    _capacity: int

    def __init__(self,
                 id_resource: int,
                 resource_type: ResourceType,
                 capacity: int,
                 ):
        """
        Initializes a new instance of the Resource class.

        Args:
            id_resource (int): The ID of the resource.
            resource_type (ResourceType): The type of the resource.
            capacity (int): The capacity of the resource.
            availability (ResourceAvailability, optional): The availability of the resource. Defaults to None.
        """
        self._id_resource = id_resource
        self._type = resource_type
        self._capacity = capacity

    @property
    def id_resource(self) -> int:
        """
        int: The ID of the resource.
        """
        return self._id_resource

    @id_resource.setter
    def id_resource(self, value: int):
        """
        Sets the ID of the resource.

        Args:
            value (int): The ID of the resource.
        """
        self._id_resource = value

    @property
    def type(self) -> ResourceType:
        """
        ResourceType: The type of the resource.
        """
        return self._type

    @type.setter
    def type(self, value: ResourceType):
        """
        Sets the type of the resource.

        Args:
            value (ResourceType): The type of the resource.
        """
        self._type = value

    @property
    def capacity(self) -> int:
        """
        int: The capacity of the resource.
        """
        return self._capacity

    @capacity.setter
    def capacity(self, value: int):
        """
        Sets the capacity of the resource.

        Args:
            value (int): The capacity of the resource.
        """
        self._capacity = value

    @property
    def key(self) -> str:
        """
        str: The key of the resource.
        """
        return f"{self.type}{self.id_resource}"

    def copy(self) -> 'Resource':
        """
        Creates a copy of the resource.

        Returns:
            Resource: A copy of the resource.
        """
        return Resource(id_resource=self.id_resource,
                        resource_type=self.type,
                        capacity=self.capacity,
                        availability=self.availability.copy() if self.availability is not None else None)

    def __hash__(self):
        """
        Returns the hash value of the resource.

        Returns:
            int: The hash value of the resource.
        """
        return hash((self._id_resource, self._type))

    def __eq__(self, other):
        """
        Checks if the resource is equal to another resource.

        Args:
            other (Resource): The other resource to compare.

        Returns:
            bool: True if the resources are equal, False otherwise.
        """
        return isinstance(other, Resource)\
            and self.id_resource == other.id_resource \
            and self.type == other.type

    def __str__(self):
        """
        Returns a string representation of the resource.

        Returns:
            str: A string representation of the resource.
        """
        return f"Resource{{id: {self.id_resource}, type: {self.type}}}"


# ~~~~~~~ ResourceConsumption ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ResourceConsumption:
    """
    Represents the resource consumption for a task in a problem instance.
    """

    _consumption_by_resource: dict[Resource, int]

    def __init__(self, consumption_by_resource: dict[Resource, int]):
        """
        Initializes a new instance of the ResourceConsumption class.

        Args:
            consumption_by_resource (dict[Resource, int]): A dictionary mapping resources to their consumption values.
        """
        self._consumption_by_resource = consumption_by_resource

    @property
    def consumption_by_resource(self) -> dict[Resource, int]:
        """
        Gets the dictionary of resource consumption values.

        Returns:
            dict[Resource, int]: A dictionary mapping resources to their consumption values.
        """
        return self._consumption_by_resource

    @consumption_by_resource.setter
    def consumption_by_resource(self, value: dict[Resource, int]):
        """
        Sets the dictionary of resource consumption values.

        Args:
            value (dict[Resource, int]): A dictionary mapping resources to their consumption values.
        """
        self._consumption_by_resource = value

    def copy(self) -> 'ResourceConsumption':
        """
        Creates a copy of the ResourceConsumption object.

        Returns:
            ResourceConsumption: A new instance of the ResourceConsumption class with the same consumption values.
        """
        return ResourceConsumption(consumption_by_resource=self.consumption_by_resource.copy())

    def __getitem__(self, resource: Resource | int):
        """
        Gets the consumption value for the specified resource.

        Args:
            resource (Resource | int): The resource or its index.

        Returns:
            int: The consumption value for the specified resource.
        """
        return self.consumption_by_resource[resource]

    def __str__(self):
        """
        Returns a string representation of the ResourceConsumption object.

        Returns:
            str: A string representation of the ResourceConsumption object.
        """
        return f"ResourceConsumption{{consumptions: {self.consumption_by_resource}}}"


# ~~~~~~~ Job ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Job:
    """
    Represents a job in a problem instance.
    """

    def __init__(self,
                 id_job: int,
                 duration: int,
                 resource_consumption: ResourceConsumption,
                 due_date: int = 0,
                 completed: bool = False):
        """
        Initializes a new instance of the Job class.

        Args:
            id_job (int): The ID of the job.
            duration (int): The duration of the job.
            resource_consumption (ResourceConsumption): The resource consumption of the job.
            due_date (int, optional): The due date of the job. Defaults to 0.
            completed (bool, optional): Indicates whether the job has been completed. Defaults to False.
        """
        self._id_job = id_job
        self._duration = duration
        self._resource_consumption = resource_consumption
        self._due_date = due_date
        self._completed = completed

    @property
    def id_job(self) -> int:
        """
        Gets the ID of the job.

        Returns:
            int: The ID of the job.
        """
        return self._id_job

    @id_job.setter
    def id_job(self, value: int):
        """
        Sets the ID of the job.

        Args:
            value (int): The ID of the job.
        """
        self._id_job = value

    @property
    def duration(self) -> int:
        """
        Gets the duration of the job.

        Returns:
            int: The duration of the job.
        """
        return self._duration

    @duration.setter
    def duration(self, value: int):
        """
        Sets the duration of the job.

        Args:
            value (int): The duration of the job.
        """
        self._duration = value

    @property
    def resource_consumption(self) -> ResourceConsumption:
        """
        Gets the resource consumption of the job.

        Returns:
            ResourceConsumption: The resource consumption of the job.
        """
        return self._resource_consumption

    @resource_consumption.setter
    def resource_consumption(self, value: ResourceConsumption):
        """
        Sets the resource consumption of the job.

        Args:
            value (ResourceConsumption): The resource consumption of the job.
        """
        self._resource_consumption = value

    @property
    def due_date(self) -> int | None:
        """
        Gets the due date of the job.

        Returns:
            int or None: The due date of the job, or None if there is no due date.
        """
        return self._due_date

    @due_date.setter
    def due_date(self, value: int):
        """
        Sets the due date of the job.

        Args:
            value (int): The due date of the job.
        """
        self._due_date = value

    @property
    def completed(self) -> bool:
        """
        Gets a value indicating whether the job has been completed.

        Returns:
            bool: True if the job has been completed, False otherwise.
        """
        return self._completed

    @completed.setter
    def completed(self, value: bool):
        """
        Sets a value indicating whether the job has been completed.

        Args:
            value (bool): True if the job has been completed, False otherwise.
        """
        self._completed = value

    def copy(self) -> 'Job':
        """
        Creates a copy of the job.

        Returns:
            Job: A copy of the job.
        """
        return Job(id_job=self.id_job,
                   duration=self.duration,
                   resource_consumption=self.resource_consumption.copy(),
                   due_date=self.due_date,
                   completed=self.completed)

    def __hash__(self):
        return self._id_job

    def __eq__(self, other):
        return isinstance(other, Job)\
            and self.id_job == other.id_job

    def __str__(self):
        return f"Job{{id: {self.id_job}}}"


# ~~~~~~~ Precedence ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class Precedence:
    """
    Represents a precedence relationship between two tasks in a project scheduling problem.
    """

    _id_child: int
    _id_parent: int

    def __init__(self,
                 id_child: int,
                 id_parent: int):
        """
        Initializes a new Precedence object.

        Args:
            id_child (int): The ID of the child task.
            id_parent (int): The ID of the parent task.
        """
        self._id_child = id_child
        self._id_parent = id_parent

    @property
    def id_child(self) -> int:
        """
        Gets the ID of the child task.

        Returns:
            int: The ID of the child task.
        """
        return self._id_child

    @id_child.setter
    def id_child(self, value: int):
        """
        Sets the ID of the child task.

        Args:
            value (int): The ID of the child task.
        """
        self._id_child = value

    @property
    def id_parent(self) -> int:
        """
        Gets the ID of the parent task.

        Returns:
            int: The ID of the parent task.
        """
        return self._id_parent

    @id_parent.setter
    def id_parent(self, value: int):
        """
        Sets the ID of the parent task.

        Args:
            value (int): The ID of the parent task.
        """
        self._id_parent = value

    def copy(self) -> 'Precedence':
        """
        Creates a copy of the Precedence object.

        Returns:
            Precedence: A new Precedence object with the same child and parent IDs.
        """
        return Precedence(id_child=self.id_child,
                          id_parent=self.id_parent)

    def __hash__(self):
        return hash((self._id_child, self._id_parent))

    def __eq__(self, other):
        return isinstance(other, Precedence)\
            and self.id_child == other.id_child\
            and self.id_parent == other.id_parent

    def __str__(self):
        return f"Precedence{{child: {self.id_child}, parent: {self.id_parent}}}"


# ~~~~~~~ ProblemInstance ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ProblemInstance:
    _name: Optional[str]

    _horizon: int

    _resources: list[Resource] = []
    _jobs: list[Job] = []
    _precedences: list[Precedence] = []

    _resources_by_id: dict[int, Resource] = {}
    _resources_by_key: dict[str, Resource] = {}
    _jobs_by_id: dict[int, Job] = {}
    _precedences_by_id_child: dict[int, Iterable[Precedence]] = {}
    _precedences_by_id_parent: dict[int, Iterable[Precedence]] = {}

    def __init__(self,
                 horizon: int,
                 resources: Collection[Resource],
                 jobs: Collection[Job],
                 precedences: Collection[Precedence],
                 name: str = None):
        """
        Initialize a ProblemInstance object.

        Args:
            horizon (int): The horizon of the problem instance.
            resources (Collection[Resource]): The collection of resources in the problem instance.
            jobs (Collection[Job]): The collection of jobs in the problem instance.
            precedences (Collection[Precedence]): The collection of precedences in the problem instance.
            name (str, optional): The name of the problem instance. Defaults to None.
        """
        self.name = name

        self.horizon = horizon

        self.resources = list_of(resources)
        self.jobs = list_of(jobs)
        self.precedences = list_of(precedences)

    @property
    def name(self) -> Optional[str]:
        """
        Get the name of the problem instance.

        Returns:
            Optional[str]: The name of the problem instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the problem instance.

        Args:
            value (str): The name of the problem instance.
        """
        self._name = value

    @property
    def horizon(self) -> int:
        """
        Get the horizon of the problem instance.

        Returns:
            int: The horizon of the problem instance.
        """
        return self._horizon

    @horizon.setter
    def horizon(self, value: int):
        """
        Set the horizon of the problem instance.

        Args:
            value (int): The horizon of the problem instance.
        """
        self._horizon = value

    @property
    def resources(self) -> list[Resource]:
        """
        Get the list of resources in the problem instance.

        Returns:
            list[Resource]: The list of resources in the problem instance.
        """
        return self._resources

    @resources.setter
    def resources(self, value: list[Resource]):
        """
        Set the list of resources in the problem instance.

        Args:
            value (list[Resource]): The list of resources in the problem instance.
        """
        self._resources = value

    @property
    def resources_by_id(self) -> dict[int, Resource]:
        """
        Get the dictionary of resources by ID in the problem instance.

        Returns:
            dict[int, Resource]: The dictionary of resources by ID in the problem instance.
        """
        if len(self._resources_by_id) != len(self._resources):
            self._resources_by_id = {r.id_resource: r for r in self._resources}
        return self._resources_by_id

    @property
    def resources_by_key(self) -> dict[str, Resource]:
        """
        Get the dictionary of resources by key in the problem instance.

        Returns:
            dict[str, Resource]: The dictionary of resources by key in the problem instance.
        """
        if len(self._resources_by_key) != len(self._resources):
            self._resources_by_key = {r.key: r for r in self._resources}
        return self._resources_by_key

    @property
    def jobs(self) -> list[Job]:
        """
        Get the list of jobs in the problem instance.

        Returns:
            list[Job]: The list of jobs in the problem instance.
        """
        return self._jobs

    @jobs.setter
    def jobs(self, value: list[Job]):
        """
        Set the list of jobs in the problem instance.

        Args:
            value (list[Job]): The list of jobs in the problem instance.
        """
        self._jobs = value

    @property
    def jobs_by_id(self) -> dict[int, Job]:
        """
        Get the dictionary of jobs by ID in the problem instance.

        Returns:
            dict[int, Job]: The dictionary of jobs by ID in the problem instance.
        """
        if len(self._jobs_by_id) != len(self._jobs):
            self._jobs_by_id = {j.id_job: j for j in self._jobs}
        return self._jobs_by_id

    @property
    def precedences(self) -> list[Precedence]:
        """
        Get the list of precedences in the problem instance.

        Returns:
            list[Precedence]: The list of precedences in the problem instance.
        """
        return self._precedences

    @precedences.setter
    def precedences(self, value: list[Precedence]):
        """
        Set the list of precedences in the problem instance.

        Args:
            value (list[Precedence]): The list of precedences in the problem instance.
        """
        self._precedences = value

        # Recompute precedence index as automatically checking and recomputing it in the appropriate properties is expensive
        self._precedences_by_id_child = defaultdict(list)
        self._precedences_by_id_parent = defaultdict(list)
        for p in self._precedences:
            self._precedences_by_id_child[p.id_child] += [p]
            self._precedences_by_id_parent[p.id_parent] += [p]

        self._successors_closure = compute_descendants(self)
        self._predecessors_closure = compute_ancestors(self)

    @property
    def precedences_by_id_child(self) -> dict[int, Iterable[Precedence]]:
        """
        Get the dictionary of precedences by child ID in the problem instance.

        Returns:
            dict[int, Iterable[Precedence]]: The dictionary of precedences by child ID in the problem instance.
        """
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_child

    @property
    def precedences_by_id_parent(self) -> dict[int, Iterable[Precedence]]:
        """
        Get the dictionary of precedences by parent ID in the problem instance.

        Returns:
            dict[int, Iterable[Precedence]]: The dictionary of precedences by parent ID in the problem instance.
        """
        # This is not recomputed automatically as it is hard to check
        return self._precedences_by_id_parent

    @property
    def successors_closure(self) -> dict[int, set[int]]:
        """
        Get the successors closure of the problem instance.

        Returns:
            dict[int, set[int]]: The successors closure of the problem instance.
        """
        return self._successors_closure

    @property
    def predecessors_closure(self) -> dict[int, set[int]]:
        """
        Get the predecessors closure of the problem instance.

        Returns:
            dict[int, set[int]]: The predecessors closure of the problem instance.
        """
        return self._predecessors_closure

    def copy(self) -> Self:
        """
        Create a copy of the problem instance.

        Returns:
            Self: The copy of the problem instance.
        """
        return ProblemInstance(horizon=self.horizon,
                               projects=[p.copy() for p in self.projects],
                               resources=[r.copy() for r in self.resources],
                               jobs=[j.copy() for j in self.jobs],
                               precedences=[p.copy() for p in self.precedences],
                               name=self.name)

    def __str__(self):
        """
        Get the string representation of the problem instance.

        Returns:
            str: The string representation of the problem instance.
        """
        return f"ProblemInstance{{name: {self._name}}}"


def parse_psplib(filename: str) -> ProblemInstance:
    """
    Parses a PSPLIB file and returns a ProblemInstance object.

    Args:
        filename (str): The path to the PSPLIB file.

    Returns:
        ProblemInstance: The parsed ProblemInstance object.
    """
    with open(filename, "rt") as f:
        return __parse_psplib_internal(f, name_as=os.path.basename(filename))


def __parse_psplib_internal(file: IO, name_as: str | None) -> ProblemInstance:
    """
    Parses a PSPLIB file and returns a ProblemInstance object.

    This function is... long. It reads the file line by line, parsing the data into a ProblemInstance object.
    Should a brave soul venture into this function, they will find a series of helper functions that
    make the parsing process more manageable. It follows the structure of the PSPLIB file format.
    """
    def read_line():
        nonlocal line_num
        line_num += 1
        return file.readline()

    def skip_lines(count: int) -> None:
        nonlocal line_num
        for _ in range(count):
            read_line()

    def asterisks() -> None:
        skip_lines(1)

    def process_split_line(line,
                           target_indices: tuple,
                           end_of_line_array_index: int = -1) -> tuple:
        content = line.split(maxsplit=end_of_line_array_index)
        if (len(content) - 1) < end_of_line_array_index:  # If the end-of-line array is empty...
            content.append("")  # ...insert empty array string
        elif (len(content) - 1) < max(target_indices):
            raise ParseError.in_file(file, line_num, "Line contains less values than expected")

        return tuple(content[i] for i in target_indices)

    def parse_split_line(target_indices: tuple,
                         end_of_line_array_index: int = -1,
                         move_line: bool = True) -> tuple:
        nonlocal line_num

        line = file.readline()
        result = process_split_line(line, target_indices, end_of_line_array_index)
        if move_line:
            line_num += 1

        return result

    def try_parse_value(value_str) -> int:
        nonlocal line_num

        if not value_str.isdecimal():
            raise ParseError.in_file(file, line_num, "Integer value expected on key-value line")

        value = int(value_str)
        return value

    def _check_key(expected_key: str,
                   key: str):
        nonlocal line_num

        if key != expected_key:
            raise ParseError.in_file(file, line_num, "Unexpected key on key-value line")

    def parse_key_value_line(key_value_indices: tuple[int, int],
                             expected_key: str) -> int:
        nonlocal line_num

        key: str
        value_str: str
        key, value_str = parse_split_line(key_value_indices, move_line=False)
        _check_key(expected_key, key)
        value = try_parse_value(value_str)

        line_num += 1
        return value

    def parse_colon_key_value_line(expected_key):
        PSPLIB_KEY_VALUE_SEPARATOR: str = ':'
        nonlocal line_num

        key: str
        value_str: str
        key, value_str = file.readline().split(sep=PSPLIB_KEY_VALUE_SEPARATOR)
        _check_key(expected_key, key.strip())
        value = try_parse_value(value_str.strip())

        line_num += 1
        return value

    def build():
        return ProblemInstance(
            horizon,
            sorted(resources, key=lambda r: r.key),
            sorted(jobs, key=lambda j: j.id_job),
            sorted(precedences, key=lambda p: (p.id_child, p.id_parent)),
            (name_as if (name_as is not None) else os.path.basename(file.name)),
        )

    line_num = 1

    asterisks()
    skip_lines(2)  # file with basedata   &   initial value random generator

    asterisks()
    project_count = parse_colon_key_value_line("projects")
    job_count = parse_colon_key_value_line("jobs (incl. supersource/sink )")
    horizon = parse_colon_key_value_line("horizon")
    skip_lines(1)  # RESOURCES list header
    _renewable_resource_count = parse_key_value_line((1, 3), "renewable")
    _nonrenewable_resource_count = parse_key_value_line((1, 3), "nonrenewable")
    _doubly_constrained_resource_count = parse_key_value_line((1, 4), "doubly")  # "doubly constrained" split as two...

    asterisks()
    skip_lines(2)  # PROJECT INFORMATION   &   projects header (pronr. #jobs rel.date duedate tardcost  MPM-Time)
    projects: list = []
    for _ in range(project_count):
        id_project_str, due_date_str, tardiness_cost_str = parse_split_line((0, 3, 4))  # ignore pronr, rel.date, MPM-Time
        projects.append((try_parse_value(id_project_str),
                         try_parse_value(due_date_str),
                         try_parse_value(tardiness_cost_str)))

    asterisks()
    skip_lines(2)  # PRECEDENCE RELATIONS   &   precedences header (jobnr. #modes #successors successors)
    precedences: list[Precedence] = []
    for _ in range(job_count):
        id_job_str, successor_count_str, successors_str = parse_split_line((0, 2, 3), end_of_line_array_index=3)  # ignore #mode
        if try_parse_value(successor_count_str) > 0:
            id_job = try_parse_value(id_job_str)
            precedences += [Precedence(id_child=id_job, id_parent=try_parse_value(successor_str))
                            for successor_str in successors_str.split()]

    asterisks()
    skip_lines(1)  # REQUESTS/DURATIONS
    resources_str, = parse_split_line((3,), end_of_line_array_index=3)
    resource_type_id_pairs = chunk(resources_str.split(), 2)
    resource_data: list[tuple[int, ResourceType]] = [(try_parse_value(id_resource_str), ResourceType(resource_type_str))
                                                     for resource_type_str, id_resource_str in resource_type_id_pairs]

    asterisks()
    job_data: list[tuple[int, int, dict[int, int]]] = []
    for _ in range(job_count):
        id_job_str, duration, consumptions = parse_split_line((0, 2, 3), end_of_line_array_index=3)  # ignore mode
        consumption_by_resource_id = {resource[0]: amount
                                      for resource, amount in zip(resource_data,
                                                                  map(try_parse_value, consumptions.split()))}
        job_data.append((try_parse_value(id_job_str), try_parse_value(duration), consumption_by_resource_id))

    asterisks()
    skip_lines(2)  # RESOURCE AVAILABILITIES   &   Resource name headers (assuming same order of resources as above)
    capacities_str, = parse_split_line((0,), end_of_line_array_index=0)
    capacities: Iterable[int] = map(try_parse_value, capacities_str.split())

    resources: list[Resource] = [Resource(id_resource, resource_type, capacity)
                                 for (id_resource, resource_type), capacity in zip(resource_data, capacities)]
    jobs: list[Job] = [Job(id_job,
                           duration,
                           ResourceConsumption({resource: consumption_by_resource_id[resource.id_resource]
                                                for resource in resources}))
                       for id_job, duration, consumptio in job_data]

    asterisks()

    return build()


class ParseError(Exception):
    """
    Exception raised for parsing errors.
    """

    def __init__(self, message):
        super().__init__(message)

    @staticmethod
    def in_file(file: IO,
                line_num: int,
                message: str):
        """
        Create a ParseError for an error in a file.

        Args:
            file (IO): The file object where the error occurred.
            line_num (int): The line number where the error occurred.
            message (str): The error message.

        Returns:
            ParseError: The ParseError object.
        """
        return ParseError(f"[{file.name}:{line_num}] {message}")


def list_of(items: Iterable[T]) -> list[T]:
    """
    Convert an iterable to a list.

    Args:
        items (Iterable[T]): The iterable to be converted.

    Returns:
        list[T]: The converted list.

    """
    return items if items is list else list(items)


def chunk(sequence: Sequence[T], chunk_size: int) -> Iterable[Iterable[T]]:
    """
    Splits a sequence into smaller chunks of a specified size.

    Args:
        sequence: The sequence to be chunked.
        chunk_size: The size of each chunk.

    Yields:
        An iterable of iterables, where each inner iterable represents a chunk of the original sequence.
    """
    for i in range(0, len(sequence), chunk_size):
        yield sequence[i:(i + chunk_size)]


def load_instances(dir: str) -> list[ProblemInstance]:
    """
    Loads all problem instances from the given data directory.

    Args:
        dir (str): The directory containing the problem instance files.

    Returns:
        list[ProblemInstance]: A list of ProblemInstance objects loaded from the files in the directory.
    """
    instances = []
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path) and filename.endswith(".sm"):
            instance = parse_psplib(file_path)
            instances.append(instance)
    return instances
