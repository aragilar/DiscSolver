"""
Defines the common data structures
"""

from collections import defaultdict
from collections.abc import MutableMapping

import attr
from numpy import asarray, concatenate, zeros


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
