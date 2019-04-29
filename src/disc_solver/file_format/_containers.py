"""
Defines the common data structures
"""

from collections import defaultdict
from collections.abc import MutableMapping

import attr
from numpy import asarray, concatenate, zeros

from h5preserve import wrap_on_demand, OnDemandWrapper, DelayedContainer

from ..float_handling import float_type
from ..utils import (
    ODEIndex, str_to_float, str_to_int, str_to_bool, CaseDependentConfigParser,
)


def mcmc_vars_str_to_obj(mcmc_str):
    """
    Convert mcmc_vars string to MCMCVars
    """
    if mcmc_str is None:
        return None
    if isinstance(mcmc_str, MCMCVars):
        return mcmc_str
    split_str = mcmc_str.strip().split(',')
    return MCMCVars(
        with_v_r="v_r" in split_str,
        with_v_a="v_a" in split_str,
        with_v_k="v_k" in split_str,
    )


def replace_empty_string(string):
    """
    Helper for attrs which expect "None".
    """
    if string == '':
        return "None"
    return string


# pylint: disable=too-few-public-methods

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
    sonic_point = attr.ib()
    sonic_point_values = attr.ib()
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
    nwalkers = attr.ib()
    iterations = attr.ib()
    threads = attr.ib()
    target_velocity = attr.ib()
    split_method = attr.ib()
    v_rin_on_c_s = attr.ib()
    v_a_on_c_s = attr.ib()
    c_s_on_v_k = attr.ib()
    η_O = attr.ib()
    η_H = attr.ib()
    η_A = attr.ib()
    η_derivs = attr.ib(default="True")
    jump_before_sonic = attr.ib(default=None, converter=replace_empty_string)
    use_taylor_jump = attr.ib(default="True")
    mcmc_vars = attr.ib(default=None)
    v_θ_sonic_crit = attr.ib(default=None, converter=replace_empty_string)

    def asdict(self):
        """
        Convert ConfigInput to dict
        """
        return attr.asdict(self, recurse=False)

    def to_conf_file(self, filename):
        """
        Convert ConfigInput to cfg file
        """
        cfg = CaseDependentConfigParser()

        cfg["config"]["start"] = self.start
        cfg["config"]["stop"] = self.stop
        cfg["config"]["taylor_stop_angle"] = self.taylor_stop_angle
        cfg["config"]["max_steps"] = self.max_steps
        cfg["config"]["num_angles"] = self.num_angles
        cfg["config"]["label"] = self.label
        cfg["config"]["relative_tolerance"] = self.relative_tolerance
        cfg["config"]["absolute_tolerance"] = self.absolute_tolerance
        cfg["config"]["nwalkers"] = self.nwalkers
        cfg["config"]["iterations"] = self.iterations
        cfg["config"]["threads"] = self.threads
        cfg["config"]["target_velocity"] = self.target_velocity
        cfg["config"]["split_method"] = self.split_method

        cfg["initial"]["γ"] = self.γ
        cfg["initial"]["v_rin_on_c_s"] = self.v_rin_on_c_s
        cfg["initial"]["v_a_on_c_s"] = self.v_a_on_c_s
        cfg["initial"]["c_s_on_v_k"] = self.c_s_on_v_k
        cfg["initial"]["η_O"] = self.η_O
        cfg["initial"]["η_H"] = self.η_H
        cfg["initial"]["η_A"] = self.η_A

        cfg["config"]["η_derivs"] = self.η_derivs
        cfg["config"]["use_taylor_jump"] = self.use_taylor_jump

        if self.jump_before_sonic is not None:
            cfg["config"]["jump_before_sonic"] = self.jump_before_sonic
        if self.mcmc_vars is not None:
            cfg["config"]["mcmc_vars"] = self.mcmc_vars
        if self.v_θ_sonic_crit is not None:
            cfg["config"]["v_θ_sonic_crit"] = self.v_θ_sonic_crit

        with open(filename, 'w') as f:
            cfg.write(f)

    def to_soln_input(self):
        """
        Convert user input into solver input
        """
        return SolutionInput(
            start=float_type(str_to_float(self.start)),
            stop=float_type(str_to_float(self.stop)),
            taylor_stop_angle=float_type(str_to_float(self.taylor_stop_angle)),
            max_steps=str_to_int(self.max_steps),
            num_angles=str_to_int(self.num_angles),
            relative_tolerance=float_type(
                str_to_float(self.relative_tolerance)
            ),
            absolute_tolerance=float_type(
                str_to_float(self.absolute_tolerance)
            ),
            jump_before_sonic=(
                None if self.jump_before_sonic == "None" or
                self.jump_before_sonic is None
                else float_type(str_to_float(self.jump_before_sonic))
            ),
            v_θ_sonic_crit=(
                None if self.v_θ_sonic_crit == "None" or
                self.v_θ_sonic_crit is None
                else float_type(str_to_float(self.v_θ_sonic_crit))
            ),
            η_derivs=str_to_bool(self.η_derivs),
            nwalkers=str_to_int(self.nwalkers),
            iterations=str_to_int(self.iterations),
            threads=str_to_int(self.threads),
            target_velocity=float_type(str_to_float(self.target_velocity)),
            split_method=self.split_method,
            use_taylor_jump=str_to_bool(self.use_taylor_jump),
            mcmc_vars=mcmc_vars_str_to_obj(self.mcmc_vars),
            γ=float_type(str_to_float(self.γ)),
            v_rin_on_c_s=float_type(str_to_float(self.v_rin_on_c_s)),
            v_a_on_c_s=float_type(str_to_float(self.v_a_on_c_s)),
            c_s_on_v_k=float_type(str_to_float(self.c_s_on_v_k)),
            η_O=float_type(str_to_float(self.η_O)),
            η_H=float_type(str_to_float(self.η_H)),
            η_A=float_type(str_to_float(self.η_A)),
        )


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
    nwalkers = attr.ib()
    iterations = attr.ib()
    threads = attr.ib()
    target_velocity = attr.ib()
    split_method = attr.ib()
    v_rin_on_c_s = attr.ib()
    v_a_on_c_s = attr.ib()
    c_s_on_v_k = attr.ib()
    η_O = attr.ib()
    η_H = attr.ib()
    η_A = attr.ib()
    η_derivs = attr.ib(default=True)
    jump_before_sonic = attr.ib(default=None)
    use_taylor_jump = attr.ib(default=True)
    mcmc_vars = attr.ib(default=None, converter=mcmc_vars_str_to_obj)
    v_θ_sonic_crit = attr.ib(default=None)

    def asdict(self):
        """
        Convert SolutionInput to dict
        """
        return attr.asdict(self, recurse=False)


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
        if isinstance(val, str):
            self._problems[str(key)].append(val)
        else:
            self._problems[str(key)].extend(val)

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
class JacobianData:
    """
    Container for jacobian associated data generated by the solver
    """
    jacobians = attr.ib(default=attr.Factory(list))
    derivs = attr.ib(default=attr.Factory(list))
    params = attr.ib(default=attr.Factory(list))
    angles = attr.ib(default=attr.Factory(list))

    def _finalise(self):
        """
        Finalise data for storage in hdf5 files
        """
        self.angles = asarray(self.angles)

        self.derivs = asarray(self.derivs)
        if self.derivs.size == 0:
            self.derivs.shape = (0, len(ODEIndex))

        self.params = asarray(self.params)
        if self.params.size == 0:
            self.params.shape = (0, len(ODEIndex))

        self.jacobians = asarray(self.jacobians)
        if self.jacobians.size == 0:
            self.jacobians.shape = (0, len(ODEIndex), len(ODEIndex))

    def __add__(self, other):
        self._finalise()
        # pylint: disable=protected-access
        other._finalise()
        # pylint: enable=protected-access
        return JacobianData(
            derivs=concatenate((self.derivs, other.derivs)),
            params=concatenate((self.params, other.params)),
            angles=concatenate((self.angles, other.angles)),
            jacobians=concatenate((self.jacobians, other.jacobians)),
        )

    def flip(self):
        """
        Flip contents
        """
        self.derivs = self.derivs[::-1]
        self.params = self.params[::-1]
        self.angles = self.angles[::-1]
        self.jacobians = self.jacobians[::-1]
        return self


@attr.s(cmp=False, hash=False)
class InternalData:
    """
    Container for values computed internally during the solution
    """
    # pylint: disable=too-many-instance-attributes
    derivs = attr.ib(default=attr.Factory(list))
    params = attr.ib(default=attr.Factory(list))
    angles = attr.ib(default=attr.Factory(list))
    jacobian_data = attr.ib(default=attr.Factory(JacobianData))
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
        if self.derivs.size == 0:
            self.derivs.shape = (0, len(ODEIndex))

        self.params = asarray(self.params)
        if self.params.size == 0:
            self.params.shape = (0, len(ODEIndex))

        self.angles = asarray(self.angles)
        self.v_r_normal = asarray(self.v_r_normal)
        self.v_φ_normal = asarray(self.v_φ_normal)
        self.ρ_normal = asarray(self.ρ_normal)
        self.v_r_taylor = asarray(self.v_r_taylor)
        self.v_φ_taylor = asarray(self.v_φ_taylor)
        self.ρ_taylor = asarray(self.ρ_taylor)
        # pylint: disable=protected-access,no-member
        self.jacobian_data._finalise()

    def __add__(self, other):
        self._finalise()
        other._finalise()  # pylint: disable=protected-access
        problems = Problems(**self.problems)  # pylint: disable=not-a-mapping
        problems.update(other.problems)
        return InternalData(
            derivs=concatenate((self.derivs, other.derivs)),
            params=concatenate((self.params, other.params)),
            angles=concatenate((self.angles, other.angles)),
            jacobian_data=self.jacobian_data + other.jacobian_data,
            v_r_normal=concatenate((self.v_r_normal, other.v_r_normal)),
            v_φ_normal=concatenate((self.v_φ_normal, other.v_φ_normal)),
            ρ_normal=concatenate((self.ρ_normal, other.ρ_normal)),
            v_r_taylor=concatenate((self.v_r_taylor, other.v_r_taylor)),
            v_φ_taylor=concatenate((self.v_φ_taylor, other.v_φ_taylor)),
            ρ_taylor=concatenate((self.ρ_taylor, other.ρ_taylor)),
            problems=problems,
        )

    def flip(self):
        """
        Flip contents
        """
        self._finalise()
        self.derivs = self.derivs[::-1]
        self.params = self.params[::-1]
        self.angles = self.angles[::-1]
        self.v_r_normal = self.v_r_normal[::-1]
        self.v_φ_normal = self.v_φ_normal[::-1]
        self.ρ_normal = self.ρ_normal[::-1]
        self.v_r_taylor = self.v_r_taylor[::-1]
        self.v_φ_taylor = self.v_φ_taylor[::-1]
        self.ρ_taylor = self.ρ_taylor[::-1]
        self.jacobian_data.flip()  # pylint: disable=no-member
        return self


class Solutions(MutableMapping):
    """
    Container holding the different solutions generated
    """
    def __init__(self, **solutions):
        self._solutions = {}
        self.update(solutions)

    def __getitem__(self, key):
        value = self._solutions[key]
        if isinstance(value, OnDemandWrapper):
            value = value()
            self._solutions[key] = value
        return value

    def __setitem__(self, key, val):
        self._solutions[key] = wrap_on_demand(self, key, val)

    def __delitem__(self, key):
        del self._solutions[key]

    def __iter__(self):
        for key in self._solutions:
            yield key

    def __len__(self):
        return len(self._solutions)

    def _h5preserve_update(self):
        """
        Support for h5preserve on demand use
        """
        for key, val in self.items():
            self._solutions[key] = wrap_on_demand(self, key, val)

    def __repr__(self):
        return "Solutions(" + ', '.join(
            "{key}={val}".format(key=key, val=val)
            for key, val in self._solutions.items()
        ) + ")"

    def add_solution(self, soln):
        """
        Add a solution returning the index of the solution
        """
        index = self._get_next_index()
        self[index] = soln
        return index

    def _get_next_index(self):
        """
        Get the next available index
        """
        if str(len(self._solutions)) not in self._solutions:
            return str(len(self._solutions))
        else:
            raise RuntimeError("Failed to guess a solution location")


@attr.s(cmp=False, hash=False)
class Run:
    """
    Container holding a single run of the solver code
    """
    config_input = attr.ib()
    config_filename = attr.ib()
    disc_solver_version = attr.ib()
    float_type = attr.ib()
    sonic_method = attr.ib()
    time = attr.ib(default=None)
    use_E_r = attr.ib(default=False)
    _final_solution = attr.ib(default=attr.Factory(DelayedContainer))
    solutions = attr.ib(default=attr.Factory(Solutions))

    @property
    def final_solution(self):
        """
        The best solution found
        """
        return self._final_solution

    @final_solution.setter
    def final_solution(self, soln):
        if isinstance(self._final_solution, DelayedContainer):
            self._final_solution.write_container(soln)
            self._final_solution = soln
        else:
            raise RuntimeError("Cannot change final solution")


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
    # pylint: disable=too-many-instance-attributes
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
class MCMCVars:
    """
    Container holding which variables to use inside mcmc
    """
    with_v_r = attr.ib()
    with_v_a = attr.ib()
    with_v_k = attr.ib()

    def __str__(self):
        true_vars = []
        if self.with_v_r:
            true_vars.append("v_r")
        if self.with_v_a:
            true_vars.append("v_a")
        if self.with_v_k:
            true_vars.append("v_k")
        return ','.join(true_vars)

# pylint: enable=too-few-public-methods
