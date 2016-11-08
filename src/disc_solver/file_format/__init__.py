"""
Defines common data structures and how they are to be written to files
"""

from collections import defaultdict
from collections.abc import MutableMapping
from fractions import Fraction

import attr

from numpy import asarray, concatenate, zeros

from h5preserve import Registry, new_registry_list, GroupContainer

from .old_dict_loading import dict_as_group_registry

ds_registry = Registry("disc_solver")
registries = new_registry_list(ds_registry, dict_as_group_registry)


@attr.s(cmp=False, hash=False)
class Solution:
    """
    Container for result from solver
    """
    solution_input = attr.ib()
    angles = attr.ib()
    solution = attr.ib()
    flag = attr.ib()
    coordinate_system = attr.ib()
    internal_data = attr.ib()
    initial_conditions = attr.ib()
    t_roots = attr.ib()
    y_roots = attr.ib()
    derivatives = attr.ib(default=None)


@attr.s
class ConfigInput:
    """
    Container for input from config file
    """
    start = attr.ib()
    stop = attr.ib()
    taylor_stop_angle = attr.ib()
    max_steps = attr.ib()
    num_angles = attr.ib()
    label = attr.ib()
    relative_tolerance = attr.ib()
    absolute_tolerance = attr.ib()
    γ = attr.ib()
    v_rin_on_c_s = attr.ib()
    v_a_on_c_s = attr.ib()
    c_s_on_v_k = attr.ib()
    η_O = attr.ib()
    η_H = attr.ib()
    η_A = attr.ib()
    η_derivs = attr.ib(default="True")
    jump_before_sonic = attr.ib(default=None)


@attr.s
class SolutionInput:
    """
    Container for parsed input for solution
    """
    start = attr.ib()
    stop = attr.ib()
    taylor_stop_angle = attr.ib()
    max_steps = attr.ib()
    num_angles = attr.ib()
    relative_tolerance = attr.ib()
    absolute_tolerance = attr.ib()
    γ = attr.ib()
    v_rin_on_c_s = attr.ib()
    v_a_on_c_s = attr.ib()
    c_s_on_v_k = attr.ib()
    η_O = attr.ib()
    η_H = attr.ib()
    η_A = attr.ib()
    η_derivs = attr.ib(default=True)
    jump_before_sonic = attr.ib(default=None)


class Problems(MutableMapping):
    """
    Container for storing the problems occurring during solving
    """
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
        return "Problems(" + ', '.join(
            "{key}={val}".format(key=key, val=val)
            for key, val in self._problems.items()
        ) + ")"


@attr.s(cmp=False, hash=False)
class InternalData:
    """
    Container for values computed internally during the solution
    """
    derivs = attr.ib(default=attr.Factory(list))
    params = attr.ib(default=attr.Factory(list))
    angles = attr.ib(default=attr.Factory(list))
    v_r_normal = attr.ib(default=attr.Factory(list))
    v_φ_normal = attr.ib(default=attr.Factory(list))
    ρ_normal = attr.ib(default=attr.Factory(list))
    v_r_taylor = attr.ib(default=attr.Factory(list))
    v_φ_taylor = attr.ib(default=attr.Factory(list))
    ρ_taylor = attr.ib(default=attr.Factory(list))
    problems = attr.ib(default=attr.Factory(Problems))

    def _finalise(self):
        """
        Finalise data for storage in hdf5 files
        """
        self.derivs = asarray(self.derivs)
        self.params = asarray(self.params)
        self.angles = asarray(self.angles)
        self.v_r_normal = asarray(self.v_r_normal)
        self.v_φ_normal = asarray(self.v_φ_normal)
        self.ρ_normal = asarray(self.ρ_normal)
        self.v_r_taylor = asarray(self.v_r_taylor)
        self.v_φ_taylor = asarray(self.v_φ_taylor)
        self.ρ_taylor = asarray(self.ρ_taylor)

    def __add__(self, other):
        self._finalise()
        # pylint: disable=protected-access
        other._finalise()
        # pylint: enable=protected-access
        # pylint: disable=not-a-mapping
        problems = Problems(**self.problems)
        # pylint: enable=not-a-mapping
        problems.update(other.problems)
        return InternalData(
            derivs=concatenate((self.derivs, other.derivs)),
            params=concatenate((self.params, other.params)),
            angles=concatenate((self.angles, other.angles)),
            v_r_normal=concatenate((self.v_r_normal, other.v_r_normal)),
            v_φ_normal=concatenate((self.v_φ_normal, other.v_φ_normal)),
            ρ_normal=concatenate((self.ρ_normal, other.ρ_normal)),
            v_r_taylor=concatenate((self.v_r_taylor, other.v_r_taylor)),
            v_φ_taylor=concatenate((self.v_φ_taylor, other.v_φ_taylor)),
            ρ_taylor=concatenate((self.ρ_taylor, other.ρ_taylor)),
            problems=problems,
        )


@attr.s(cmp=False, hash=False)
class DAEInternalData:
    """
    Container for values computed internally during the solution
    """
    derivs = attr.ib(default=attr.Factory(list))
    params = attr.ib(default=attr.Factory(list))
    angles = attr.ib(default=attr.Factory(list))
    residuals = attr.ib(default=attr.Factory(list))
    problems = attr.ib(default=attr.Factory(Problems))

    def _finalise(self):
        """
        Finalise data for storage in hdf5 files
        """
        self.derivs = asarray(self.derivs)
        self.params = asarray(self.params)
        self.angles = asarray(self.angles)
        self.residuals = asarray(self.residuals)

    def __add__(self, other):
        self._finalise()
        # pylint: disable=protected-access
        other._finalise()
        # pylint: enable=protected-access
        # pylint: disable=not-a-mapping
        problems = Problems(**self.problems)
        # pylint: enable=not-a-mapping
        problems.update(other.problems)
        return InternalData(
            derivs=concatenate((self.derivs, other.derivs)),
            params=concatenate((self.params, other.params)),
            angles=concatenate((self.angles, other.angles)),
            residuals=concatenate((self.residuals, other.residuals)),
            problems=problems,
        )


@attr.s(cmp=False, hash=False)
class Run:
    """
    Container holding a single run of the solver code
    """
    config_input = attr.ib()
    config_filename = attr.ib()
    time = attr.ib(default=None)
    final_solution = attr.ib(default=None)
    solutions = attr.ib(default=attr.Factory(dict))


@attr.s(
    these={
        'norm_kepler_sq': attr.ib(), 'η_O': attr.ib(), 'η_A': attr.ib(),
        'η_H': attr.ib(), 'γ': attr.ib(), 'init_con': attr.ib(),
        'angles': attr.ib(), 'a_0': attr.ib()
    }, cmp=False, hash=False, init=False
)
class InitialConditions:
    """
    Container holding the initial conditions for the solver
    """
    def __init__(
        self, *, norm_kepler_sq, η_O=None, η_A=None, η_H=None,
        γ, init_con, angles, a_0
    ):
        self.norm_kepler_sq = norm_kepler_sq
        self.a_0 = a_0
        self.γ = γ
        self.angles = angles

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


@attr.s(cmp=False, hash=False)
class DAEInitialConditions:
    """
    Container holding the initial conditions for the solver
    """
    norm_kepler_sq = attr.ib()
    a_0 = attr.ib()
    γ = attr.ib()
    angles = attr.ib()
    init_con = attr.ib()
    deriv_init_con = attr.ib()


def _str_β_to_γ(β):
    """
    Convert β to γ where it appears as a string
    """
    return str(Fraction("5/4") - Fraction(β))


# pylint: disable=missing-docstring
@ds_registry.dumper(DAEInternalData, "DAEInternalData", version=1)
def _dae_internal_dump(internal_data):
    # pylint: disable=protected-access
    internal_data._finalise()
    # pylint: enable=protected-access
    return GroupContainer(
        derivs=internal_data.derivs,
        params=internal_data.params,
        angles=internal_data.angles,
        residuals=internal_data.residuals,
        problems=internal_data.problems,
    )


@ds_registry.loader("DAEInternalData", version=1)
def _dae_internal_load(group):
    return DAEInternalData(
        derivs=group["derivs"]["data"],
        params=group["params"]["data"],
        angles=group["angles"]["data"],
        residuals=group["residuals"]["data"],
        problems=group["problems"]
    )


@ds_registry.dumper(InternalData, "InternalData", version=2)
def _internal_dump2(internal_data):
    # pylint: disable=protected-access
    internal_data._finalise()
    # pylint: enable=protected-access
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


@ds_registry.dumper(DAEInitialConditions, "DAEInitialConditions", version=1)
def _dae_initial_dump(initial_conditions):
    return GroupContainer(
        attrs={
            "norm_kepler_sq": initial_conditions.norm_kepler_sq,
            "a_0": initial_conditions.a_0,
            "γ": initial_conditions.γ,
            "init_con": initial_conditions.init_con,
            "deriv_init_con": initial_conditions.deriv_init_con,
        }, angles=initial_conditions.angles,
    )


@ds_registry.loader("DAEInitialConditions", version=1)
def _dae_initial_load(group):
    return DAEInitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=group.attrs["a_0"],
        γ=group.attrs["γ"],
        init_con=group.attrs["init_con"],
        deriv_init_con=group.attrs["deriv_init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.dumper(InitialConditions, "InitialConditions", version=3)
def _initial_dump(initial_conditions):
    return GroupContainer(
        attrs={
            "norm_kepler_sq": initial_conditions.norm_kepler_sq,
            "a_0": initial_conditions.a_0,
            "η_O": initial_conditions.η_O,
            "η_A": initial_conditions.η_A,
            "η_H": initial_conditions.η_H,
            "γ": initial_conditions.γ,
            "init_con": initial_conditions.init_con,
        }, angles=initial_conditions.angles,
    )


@ds_registry.loader("InitialConditions", version=3)
def _initial_load3(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=group.attrs["a_0"],
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=group.attrs["γ"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.loader("InitialConditions", version=2)
def _initial_load(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=group.attrs["a_0"],
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=5/4 - group.attrs["β"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.loader("InitialConditions", version=1)
def _initial_load_old(group):
    return InitialConditions(
        norm_kepler_sq=group.attrs["norm_kepler_sq"],
        a_0=float('nan'),
        η_O=group.attrs["η_O"],
        η_A=group.attrs["η_A"],
        η_H=group.attrs["η_H"],
        γ=5/4 - group.attrs["β"],
        init_con=group.attrs["init_con"],
        angles=group["angles"]["data"],
    )


@ds_registry.dumper(Solution, "Solution", version=2)
def _solution_dumper(solution):
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
        solution_input=solution.solution_input,
        derivatives=solution.derivatives,
    )


@ds_registry.loader("Solution", version=1)
def _solution_loader(group):
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


@ds_registry.loader("Solution", version=2)
def _solution_loader_2(group):
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
        derivatives=group["derivatives"],
    )


@ds_registry.dumper(ConfigInput, "ConfigInput", version=4)
def _config_dumper(config_input):
    return GroupContainer(
        attrs={
            "start": config_input.start,
            "stop": config_input.stop,
            "taylor_stop_angle": config_input.taylor_stop_angle,
            "max_steps": config_input.max_steps,
            "num_angles": config_input.num_angles,
            "label": config_input.label,
            "relative_tolerance": config_input.relative_tolerance,
            "absolute_tolerance": config_input.absolute_tolerance,
            "γ": config_input.γ,
            "v_rin_on_c_s": config_input.v_rin_on_c_s,
            "v_a_on_c_s": config_input.v_a_on_c_s,
            "c_s_on_v_k": config_input.c_s_on_v_k,
            "η_O": config_input.η_O,
            "η_H": config_input.η_H,
            "η_A": config_input.η_A,
            "η_derivs": config_input.η_derivs,
        }, jump_before_sonic=config_input.jump_before_sonic
    )


@ds_registry.loader("ConfigInput", version=1)
def _config_loader(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.loader("ConfigInput", version=2)
def _config_loader2(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"]
    )


@ds_registry.loader("ConfigInput", version=3)
def _config_loader3(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=_str_β_to_γ(group.attrs["β"]),
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("ConfigInput", version=4)
def _config_loader4(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.dumper(SolutionInput, "SolutionInput", version=4)
def _input_dumper(solution_input):
    return GroupContainer(
        attrs={
            "start": solution_input.start,
            "stop": solution_input.stop,
            "taylor_stop_angle": solution_input.taylor_stop_angle,
            "max_steps": solution_input.max_steps,
            "num_angles": solution_input.num_angles,
            "relative_tolerance": solution_input.relative_tolerance,
            "absolute_tolerance": solution_input.absolute_tolerance,
            "γ": solution_input.γ,
            "v_rin_on_c_s": solution_input.v_rin_on_c_s,
            "v_a_on_c_s": solution_input.v_a_on_c_s,
            "c_s_on_v_k": solution_input.c_s_on_v_k,
            "η_O": solution_input.η_O,
            "η_H": solution_input.η_H,
            "η_A": solution_input.η_A,
            "η_derivs": solution_input.η_derivs,
        }, jump_before_sonic=solution_input.jump_before_sonic
    )


@ds_registry.loader("SolutionInput", version=1)
def _input_loader(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
    )


@ds_registry.loader("SolutionInput", version=2)
def _input_loader2(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"]
    )


@ds_registry.loader("SolutionInput", version=3)
def _input_loader3(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=5/4 - group.attrs["β"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.loader("SolutionInput", version=4)
def _input_loader4(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        taylor_stop_angle=group.attrs["taylor_stop_angle"],
        max_steps=group.attrs["max_steps"],
        num_angles=group.attrs["num_angles"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        γ=group.attrs["γ"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        c_s_on_v_k=group.attrs["c_s_on_v_k"],
        η_O=group.attrs["η_O"],
        η_H=group.attrs["η_H"],
        η_A=group.attrs["η_A"],
        jump_before_sonic=group["jump_before_sonic"],
        η_derivs=group.attrs["η_derivs"],
    )


@ds_registry.dumper(Run, "Run", version=1)
def _run_dumper(run):
    return GroupContainer(
        time=run.time,
        config_filename=run.config_filename,
        config_input=run.config_input,
        final_solution=run.final_solution,
        solutions=run.solutions,
    )


@ds_registry.loader("Run", version=1)
def _run_loader(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
    )


@ds_registry.dumper(Problems, "Problems", version=1)
def _problems_dumper(problems):
    group = GroupContainer()
    for key, item in problems.items():
        group[key] = asarray([
            s.encode("utf8") for s in item
        ])
    return group


@ds_registry.loader("Problems", version=1)
def _problems_loader(group):
    problems = Problems()
    for key, item in group.items():
        problems[key] = [
            s.decode("utf8") for s in item["data"]
        ]
    return problems


# pylint: enable=missing-docstring
