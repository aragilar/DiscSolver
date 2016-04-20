"""
Defines common data structures and how they are to be written to files
"""

from collections import namedtuple, defaultdict
from collections.abc import MutableMapping

from numpy import asarray, zeros

from h5preserve import Registry, new_registry_list, GroupContainer

ds_registry = Registry("disc_solver")
registries = new_registry_list(ds_registry)

ConfigInput = namedtuple("ConfigInput", [
    "start", "stop", "taylor_stop_angle", "max_steps", "num_angles", "label",
    "relative_tolerance", "absolute_tolerance", "β", "v_rin_on_c_s",
    "v_a_on_c_s", "c_s_on_v_k", "η_O", "η_H", "η_A",
])

SolutionInput = namedtuple("SolutionInput", [
    "start", "stop", "taylor_stop_angle", "max_steps", "num_angles",
    "relative_tolerance", "absolute_tolerance", "β", "v_rin_on_c_s",
    "v_a_on_c_s", "c_s_on_v_k", "η_O", "η_H", "η_A",
])

Solution = namedtuple("Solution", [
    "solution_input", "angles", "solution", "flag", "coordinate_system",
    "internal_data", "initial_conditions", "t_roots", "y_roots",
])


class Problems(MutableMapping):
    # pylint: disable=missing-docstring
    def __init__(self, **problems):
        self._problems = defaultdict(list)
        self.update(problems)

    def __getitem__(self, key):
        return self._problems[str(key)]

    def __setitem__(self, key, val):
        self._problems[str(key)].append(val)

    def __delitem__(self, key):
        del self._problems[key]

    def __iter__(self):
        for key in self._problems:
            yield key

    def __len__(self):
        return len(self._problems)

    def __repr__(self):
        return repr(self._problems)


class InternalData:
    # pylint: disable=missing-docstring,too-few-public-methods
    # pylint: disable=too-many-instance-attributes
    def __init__(
        self, *, derivs=None, params=None, angles=None, v_r_normal=None,
        v_φ_normal=None, ρ_normal=None, v_r_taylor=None, v_φ_taylor=None,
        ρ_taylor=None, problems=None
    ):
        if derivs is None:
            derivs = []
        if params is None:
            params = []
        if angles is None:
            angles = []
        if v_r_normal is None:
            v_r_normal = []
        if v_φ_normal is None:
            v_φ_normal = []
        if ρ_normal is None:
            ρ_normal = []
        if v_r_taylor is None:
            v_r_taylor = []
        if v_φ_taylor is None:
            v_φ_taylor = []
        if ρ_taylor is None:
            ρ_taylor = []
        if problems is None:
            problems = Problems()

        self.derivs = derivs
        self.params = params
        self.angles = angles
        self.v_r_normal = v_r_normal
        self.v_φ_normal = v_φ_normal
        self.ρ_normal = ρ_normal
        self.v_r_taylor = v_r_taylor
        self.v_φ_taylor = v_φ_taylor
        self.ρ_taylor = ρ_taylor
        self.problems = problems

    def finalise(self):
        self.derivs = asarray(self.derivs)
        self.params = asarray(self.params)
        self.angles = asarray(self.angles)
        self.v_r_normal = asarray(self.v_r_normal)
        self.v_φ_normal = asarray(self.v_φ_normal)
        self.ρ_normal = asarray(self.ρ_normal)
        self.v_r_taylor = asarray(self.v_r_taylor)
        self.v_φ_taylor = asarray(self.v_φ_taylor)
        self.ρ_taylor = asarray(self.ρ_taylor)

    def __repr__(self):
        return repr(self.__dict__)


class Run:
    # pylint: disable=missing-docstring,too-few-public-methods
    def __init__(
        self, *, config_input=None, config_filename=None, time=None,
        final_solution=None, solutions=None
    ):
        self.config_input = config_input
        self.config_filename = config_filename
        self.time = time
        self.final_solution = final_solution
        self.solutions = solutions if solutions is not None else {}


class InitialConditions:
    # pylint: disable=missing-docstring,too-few-public-methods
    # pylint: disable=too-many-branches,too-many-instance-attributes
    def __init__(
        self, *, norm_kepler_sq=None, c_s=None, η_O=None, η_A=None, η_H=None,
        β=None, init_con=None, angles=None
    ):
        if norm_kepler_sq is None:
            raise TypeError("norm_kepler_sq required")
        self.norm_kepler_sq = norm_kepler_sq
        if c_s is None:
            raise TypeError("c_s required")
        self.c_s = c_s
        if β is None:
            raise TypeError("β required")
        self.β = β
        if angles is None:
            raise TypeError("angles required")
        self.angles = angles

        if init_con is None:
            raise TypeError("init_con required")

        if len(init_con) == 8:
            self.init_con = zeros(11)
            self.init_con[:8] = init_con
            self.init_con[8] = η_O
            self.init_con[9] = η_A
            self.init_con[10] = η_H
        else:
            if η_O is None:
                η_O = init_con[8]
            elif η_O != init_con[8]:
                raise RuntimeError("Initial conditions for η_O inconstant")

            if η_A is None:
                η_A = init_con[9]
            elif η_A != init_con[9]:
                raise RuntimeError("Initial conditions for η_A inconstant")

            if η_H is None:
                η_H = init_con[10]
            elif η_H != init_con[10]:
                raise RuntimeError("Initial conditions for η_H inconstant")
            self.init_con = init_con
        self.η_O = η_O
        self.η_A = η_A
        self.η_H = η_H


@ds_registry.dumper(InternalData, "InternalData", version=1)
def _internal_dump(internal_data):
    # pylint: disable=missing-docstring
    return GroupContainer(
        derivs=internal_data.derivs,
        params=internal_data.params,
        angles=internal_data.angles,
        v_r_normal=internal_data.v_r_normal,
        v_φ_normal=internal_data.v_φ_normal,
        ρ_normal=internal_data.ρ_normal,
        v_r_taylor=internal_data.v_r_taylor,
        v_φ_taylor=internal_data.v_φ_taylor,
        ρ_taylor=internal_data.ρ_taylor,
    )


@ds_registry.dumper(InternalData, "InternalData", version=2)
def _internal_dump2(internal_data):
    # pylint: disable=missing-docstring
    return GroupContainer(
        derivs=internal_data.derivs,
        params=internal_data.params,
        angles=internal_data.angles,
        v_r_normal=internal_data.v_r_normal,
        v_φ_normal=internal_data.v_φ_normal,
        ρ_normal=internal_data.ρ_normal,
        v_r_taylor=internal_data.v_r_taylor,
        v_φ_taylor=internal_data.v_φ_taylor,
        ρ_taylor=internal_data.ρ_taylor,
        problems=internal_data.problems,
    )


@ds_registry.loader("InternalData", version=1)
def _internal_load(group):
    # pylint: disable=missing-docstring
    return InternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        v_r_normal=group["v_r_normal"]["data"],
        v_φ_normal=group["v_φ_normal"]["data"],
        ρ_normal=group["ρ_normal"]["data"],
        v_r_taylor=group["v_r_taylor"]["data"],
        v_φ_taylor=group["v_φ_taylor"]["data"],
        ρ_taylor=group["ρ_taylor"]["data"],
    )


@ds_registry.loader("InternalData", version=2)
def _internal_load2(group):
    # pylint: disable=missing-docstring
    return InternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        v_r_normal=group["v_r_normal"]["data"],
        v_φ_normal=group["v_φ_normal"]["data"],
        ρ_normal=group["ρ_normal"]["data"],
        v_r_taylor=group["v_r_taylor"]["data"],
        v_φ_taylor=group["v_φ_taylor"]["data"],
        ρ_taylor=group["ρ_taylor"]["data"],
        problems=group["problems"]
    )


@ds_registry.dumper(InitialConditions, "InitialConditions", version=1)
def _initial_dump(initial_conditions):
    # pylint: disable=missing-docstring
    return GroupContainer(
        attrs={
            "norm_kepler_sq": initial_conditions.norm_kepler_sq,
            "c_s": initial_conditions.c_s,
            "η_O": initial_conditions.η_O,
            "η_A": initial_conditions.η_A,
            "η_H": initial_conditions.η_H,
            "β": initial_conditions.β,
            "init_con": initial_conditions.init_con,
        }, angles=initial_conditions.angles,
    )


@ds_registry.loader("InitialConditions", version=1)
def _initial_load(group):
    # pylint: disable=missing-docstring
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        c_s=group.attrs["c_s"],
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        β=group.attrs["β"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.dumper(Solution, "Solution", version=1)
def _solution_dumper(solution):
    # pylint: disable=missing-docstring
    return GroupContainer(
        attrs={
            "flag": solution.flag,
            "coordinate_system": solution.coordinate_system,
        },
        angles=solution.angles,
        solution=solution.solution,
        internal_data=solution.internal_data,
        initial_conditions=solution.initial_conditions,
        t_roots=solution.t_roots,
        y_roots=solution.y_roots,
        solution_input=solution.solution_input
    )


@ds_registry.loader("Solution", version=1)
def _solution_loader(group):
    # pylint: disable=missing-docstring
    if group["t_roots"] is None:
        t_roots = None
    else:
        t_roots = group["t_roots"]["data"]
    if group["y_roots"] is None:
        y_roots = None
    else:
        y_roots = group["y_roots"]["data"]

    return Solution(
        flag=group.attrs["flag"],
        coordinate_system=group.attrs["coordinate_system"],
        angles=group["angles"]["data"],
        solution=group["solution"]["data"],
        internal_data=group["internal_data"],
        initial_conditions=group["initial_conditions"],
        solution_input=group["solution_input"],
        t_roots=t_roots,
        y_roots=y_roots,
    )


@ds_registry.dumper(ConfigInput, "ConfigInput", version=1)
def _config_dumper(config_input):
    # pylint: disable=missing-docstring
    return GroupContainer(attrs={
        "start": config_input.start,
        "stop": config_input.stop,
        "taylor_stop_angle": config_input.taylor_stop_angle,
        "max_steps": config_input.max_steps,
        "num_angles": config_input.num_angles,
        "label": config_input.label,
        "relative_tolerance": config_input.relative_tolerance,
        "absolute_tolerance": config_input.absolute_tolerance,
        "β": config_input.β,
        "v_rin_on_c_s": config_input.v_rin_on_c_s,
        "v_a_on_c_s": config_input.v_a_on_c_s,
        "c_s_on_v_k": config_input.c_s_on_v_k,
        "η_O": config_input.η_O,
        "η_H": config_input.η_H,
        "η_A": config_input.η_A,
    })


@ds_registry.loader("ConfigInput", version=1)
def _config_loader(group):
    # pylint: disable=missing-docstring
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        β=group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.dumper(SolutionInput, "SolutionInput", version=1)
def _input_dumper(solution_input):
    # pylint: disable=missing-docstring
    return GroupContainer(attrs={
        "start": solution_input.start,
        "stop": solution_input.stop,
        "taylor_stop_angle": solution_input.taylor_stop_angle,
        "max_steps": solution_input.max_steps,
        "num_angles": solution_input.num_angles,
        "relative_tolerance": solution_input.relative_tolerance,
        "absolute_tolerance": solution_input.absolute_tolerance,
        "β": solution_input.β,
        "v_rin_on_c_s": solution_input.v_rin_on_c_s,
        "v_a_on_c_s": solution_input.v_a_on_c_s,
        "c_s_on_v_k": solution_input.c_s_on_v_k,
        "η_O": solution_input.η_O,
        "η_H": solution_input.η_H,
        "η_A": solution_input.η_A,
    })


@ds_registry.loader("SolutionInput", version=1)
def _input_loader(group):
    # pylint: disable=missing-docstring
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        β=group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.dumper(Run, "Run", version=1)
def _run_dumper(run):
    # pylint: disable=missing-docstring
    return GroupContainer(
        time=run.time,
        config_filename=run.config_filename,
        config_input=run.config_input,
        final_solution=run.final_solution,
        solutions=run.solutions,
    )


@ds_registry.loader("Run", version=1)
def _run_loader(group):
    # pylint: disable=missing-docstring
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
    )


@ds_registry.dumper(Problems, "Problems", version=1)
def _problems_dumper(problems):
    # pylint: disable=missing-docstring
    group = GroupContainer()
    for key, item in problems.items():
        group[key] = asarray([
            s.encode("utf8") for s in item
        ])
    return group


@ds_registry.loader("Problems", version=1)
def _problems_loader(group):
    # pylint: disable=missing-docstring
    problems = Problems()
    for key, item in group.items():
        problems[key] = [
            s.decode("utf8") for s in item["data"]
        ]
    return problems
