# -*- coding: utf-8 -*-
"""
dae solver which uses taylor series for first part of solution
"""

from .config import define_conditions
from .dae_config import define_dae_conditions
from .dae_solution import solution


def solver(inp, run, *, store_internal=True):
    """
    Single run solver
    """
    single_solution = solution(
        inp, define_dae_conditions(inp, define_conditions(inp)),
        store_internal=store_internal
    )
    run.solutions["0"] = single_solution
    run.final_solution = run.solutions["0"]

