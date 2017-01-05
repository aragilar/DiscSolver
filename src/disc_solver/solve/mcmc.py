# -*- coding: utf-8 -*-
"""
Stepper related logic
"""

from enum import IntEnum, unique
from math import isfinite

import logbook

import emcee

from numpy import all as np_all, diff
from numpy.random import randn

from .config import define_conditions
from .solution import solution
from .utils import onroot_continue
from ..file_format import SolutionInput
from ..utils import ODEIndex

log = logbook.Logger(__name__)


@unique
class SysVars(IntEnum):
    """
    Variables for MCMC
    """
    γ = 0
    v_rin_on_c_s = 1
    v_a_on_c_s = 2
    c_s_on_v_k = 3


def solution_input_to_sys_vars(soln_input):
    """
    Convert instance of SolutionInput to mcmc variables
    """
    return [
        getattr(soln_input, v.name) for v in sorted(SysVars)
    ]


def sys_vars_to_solution_input(sys_vars, old_soln_input):
    """
    Convert mcmc variables to instance of SolutionInput
    """
    soln_input = SolutionInput(**vars(old_soln_input))
    for v in SysVars:
        setattr(soln_input, v.name, sys_vars[v])
    return soln_input


def solver(soln_input, run, store_internal=True):
    """
    MCMC solver
    """
    logprobfunc = LogProbGenerator(
        soln_input, run, store_internal=store_internal
    )
    sampler = emcee.EnsembleSampler(
        soln_input.nwalkers, len(SysVars), logprobfunc,
        threads=soln_input.threads,
    )
    sampler.run_mcmc(
        generate_initial_positions(soln_input),
        soln_input.iterations
    )
    run.final_solution = logprobfunc.best_solution


class LogProbGenerator:
    """
    Generate log probability function
    """
    def __init__(self, soln_input, run, *, store_internal):
        self._soln_input = soln_input
        self._run = run
        self._store_internal = store_internal
        self._best_logprob = - float("inf")
        self._best_logprob_index = None
        self._root_func, self._root_func_args = velocity_stop_generator(
            self._soln_input
        )

    def __call__(self, sys_vars):
        new_soln_input = sys_vars_to_solution_input(sys_vars, self._soln_input)
        if not solution_input_valid(new_soln_input):
            log.info("MCMC input invalid")
            return - float("inf")
        try:
            cons = define_conditions(new_soln_input)
        except ValueError:
            log.info(
                "MCMC input could not be converted to initial conditions."
            )
            return - float("inf")
        try:
            soln = solution(
                new_soln_input, cons, onroot_func=onroot_continue,
                store_internal=self._store_internal, root_func=self._root_func,
                root_func_args=self._root_func_args,
            )
        except RuntimeError as e:
            log.exception(e)
            return - float("inf")
        logprob = get_logprob_of_soln(new_soln_input, soln)
        if not isfinite(logprob):
            log.info("Solution invalid")
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


def get_logprob_of_soln(new_soln_input, soln):
    """
    Return log probability of solution
    """
    if soln.flag < 0:
        return - float("inf")
    if np_all((
        soln.solution[1:, ODEIndex.v_θ] - soln.solution[:-1, ODEIndex.v_θ]
    ) < 0):
        return - float("inf")
    return - (
        new_soln_input.target_velocity - soln.solution[-1, ODEIndex.v_θ]
    ) ** 2 - diff(diff(soln.solution[:, ODEIndex.v_θ])[0, -1])


def generate_initial_positions(soln_input):
    """
    Generate initial positions of walkers
    """
    sys_vars = solution_input_to_sys_vars(soln_input)
    return sys_vars * (0.01 * randn(soln_input.nwalkers, len(SysVars)) + 1)


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


def velocity_stop_generator(soln_input):
    """
    return function to stop at target velocity
    """
    def rootfn(θ, params, out):
        """
        root function to stop at required velocity
        """
        # pylint: disable=unused-argument
        out[0] = soln_input.target_velocity - params[ODEIndex.v_θ]
        return 0
    return rootfn, 1
