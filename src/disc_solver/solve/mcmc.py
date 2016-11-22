# -*- coding: utf-8 -*-
"""
Stepper related logic
"""

from enum import IntEnum, unique
from itertools import chain

import logbook

import emcee

from numpy import argmax
from numpy.random import randn

from .config import define_conditions
from .solution import solution
from .utils import onroot_continue
from ..file_format import SolutionInput
from ..utils import ODEIndex

log = logbook.Logger(__name__)
IDEAL_VELOCITY = 1000


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
    logprobfunc = logprobgenerator(soln_input, store_internal=store_internal)
    sampler = emcee.EnsembleSampler(
        soln_input.nwalkers, len(SysVars), logprobfunc,
        threads=soln_input.threads,
    )
    sampler.run_mcmc(
        generate_initial_positions(soln_input),
        soln_input.iterations
    )
    write_solutions(sampler, run)


def logprobgenerator(soln_input, *, store_internal):
    """
    Generate log probability function
    """
    def logprobfunc(sys_vars):
        """
        log probability function
        """
        new_soln_input = sys_vars_to_solution_input(sys_vars, soln_input)
        if not solution_input_valid(new_soln_input):
            return - float("inf"), None
        try:
            cons = define_conditions(new_soln_input)
        except ValueError:
            return - float("inf"), None
        try:
            soln = solution(
                new_soln_input, cons, onroot_func=onroot_continue,
                find_sonic_point=True, store_internal=store_internal,
            )
        except RuntimeError:
            return - float("inf"), None
        return get_logprob_of_soln(soln), soln

    return logprobfunc


def get_logprob_of_soln(soln):
    """
    Return log probability of solution
    """
    return - (IDEAL_VELOCITY - soln.solution[-1, ODEIndex.v_θ]) ** 2


def generate_initial_positions(soln_input):
    """
    Generate initial positions of walkers
    """
    sys_vars = solution_input_to_sys_vars(soln_input)
    return sys_vars * (randn(soln_input.nwalkers, len(SysVars)) + 1)


def write_solutions(sampler, run):
    """
    Include solutions
    """
    run.solutions.update(
        (str(i), soln) for i, soln in enumerate(
            chain.from_iterable(sampler.blobs)
        )
    )
    run.final_solution = run.solutions[str(argmax(sampler.flatlnprobability))]


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
