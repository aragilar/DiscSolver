# -*- coding: utf-8 -*-
"""
Stepper related logic
"""

from enum import IntEnum, unique, auto

import logbook

import emcee

from numpy import any as np_any, diff, errstate, isfinite
from numpy.random import randn

from .config import define_conditions
from .solution import solution
from .utils import SolverError

from ..file_format import SolutionInput
from ..utils import ODEIndex

log = logbook.Logger(__name__)

INITIAL_SPREAD = 0.1
TARGETED_PROB_WEIGHTING = 1
OUTFLOW_RATE_PROB_WEIGHTING = 1


def generate_sysvars_enum(soln_input):
    """
    Create SysVars enum based on which parameters to mcmc
    """
    if getattr(soln_input, "mcmc_vars", None) is None:
        with_v_r = True
        with_v_a = True
        with_v_k = True
    else:
        with_v_r = soln_input.mcmc_vars.with_v_r
        with_v_a = soln_input.mcmc_vars.with_v_a
        with_v_k = soln_input.mcmc_vars.with_v_k

    @unique
    class SysVars(IntEnum):
        """
        Variables for MCMC
        """
        γ = 0
        if with_v_r:
            v_rin_on_c_s = auto()
        if with_v_a:
            v_a_on_c_s = auto()
        if with_v_k:
            c_s_on_v_k = auto()

    return SysVars


def solution_input_to_sys_vars(soln_input, sys_vars_enum):
    """
    Convert instance of SolutionInput to mcmc variables
    """
    return [
        getattr(soln_input, v.name) for v in sorted(sys_vars_enum)
    ]


def sys_vars_to_solution_input(sys_vars, old_soln_input, sys_vars_enum):
    """
    Convert mcmc variables to instance of SolutionInput
    """
    soln_input = SolutionInput(**vars(old_soln_input))
    for v in sys_vars_enum:
        setattr(soln_input, v.name, sys_vars[v])
    return soln_input


def solver(soln_input, run, store_internal=True):
    """
    MCMC solver
    """
    sys_vars_enum = generate_sysvars_enum(soln_input)
    log.notice("SysVars is {}".format(list(sys_vars_enum)))
    logprobfunc = LogProbGenerator(
        soln_input, run, store_internal=store_internal,
        sys_vars_enum=sys_vars_enum,
    )
    sampler = emcee.EnsembleSampler(
        soln_input.nwalkers, len(sys_vars_enum), logprobfunc,
    )

    with errstate(invalid="ignore"):
        sampler.run_mcmc(
            generate_initial_positions(soln_input, sys_vars_enum),
            soln_input.iterations, progress=True,
        )
    run.final_solution = logprobfunc.best_solution


class LogProbGenerator:
    """
    Generate log probability function
    """
    # pylint: disable=too-few-public-methods
    def __init__(self, soln_input, run, *, store_internal, sys_vars_enum):
        self._soln_input = soln_input
        self._run = run
        self._store_internal = store_internal
        self._best_logprob = - float("inf")
        self._best_logprob_index = None
        self._sys_vars_enum = sys_vars_enum

    def __call__(self, sys_vars):
        new_soln_input = sys_vars_to_solution_input(
            sys_vars, self._soln_input, self._sys_vars_enum
        )
        if not solution_input_valid(new_soln_input):
            log.warning("MCMC input invalid")
            return - float("inf")
        try:
            cons = define_conditions(new_soln_input, use_E_r=self._run.use_E_r)
        except SolverError:
            log.warning(
                "MCMC input could not be converted to initial conditions."
            )
            return - float("inf")
        try:
            soln = solution(
                new_soln_input, cons, store_internal=self._store_internal,
            )
        except SolverError as e:
            log.exception(e)
            return - float("inf")
        logprob = get_logprob_of_soln(soln)
        if not isfinite(logprob):
            log.warning("Invalid solution found")
            return - float("inf")
        soln_index = self._run.solutions.add_solution(soln)
        if logprob > self._best_logprob:
            self._best_logprob = logprob
            self._best_logprob_index = soln_index
        return logprob

    @property
    def best_solution(self):
        """
        Return the solution with the highest probablity
        """
        if self._best_logprob_index is None:
            return None
        return self._run.solutions[self._best_logprob_index]


def get_logprob_of_soln(soln):
    """
    Return log probability of solution
    """
    if soln.flag < 0:
        return - float("inf")
    if np_any(diff(soln.solution[:, ODEIndex.v_θ]) < 0):
        return - float("inf")
    targeted_prob = - (
        soln.solution_input.target_velocity - max(
            soln.solution[:, ODEIndex.v_θ]
        )
    ) ** 2
    return (
        targeted_prob * TARGETED_PROB_WEIGHTING +
        get_outflow_rate_probability(soln) * OUTFLOW_RATE_PROB_WEIGHTING
    )


def get_outflow_rate_probability(soln):
    """
    Get likelihood for outflow structure
    """
    Δ_v_θ = diff(soln.solution[:, ODEIndex.v_θ])
    return - sum(diff(Δ_v_θ))


def generate_initial_positions(soln_input, sys_vars_enum):
    """
    Generate initial positions of walkers
    """
    sys_vars = solution_input_to_sys_vars(soln_input, sys_vars_enum)
    return sys_vars * (
        INITIAL_SPREAD * randn(soln_input.nwalkers, len(sys_vars_enum)) + 1
    )


def solution_input_valid(soln_input):
    """
    Check whether the solution input is valid.
    """
    if soln_input.v_rin_on_c_s <= 0:
        return False
    if soln_input.γ <= 0:
        return False
    if soln_input.v_a_on_c_s <= 0:
        return False
    if soln_input.c_s_on_v_k <= 0:
        return False
    return True
