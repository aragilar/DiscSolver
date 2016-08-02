# -*- coding: utf-8 -*-
"""
Single run solver
"""

from .config import define_conditions
from .solution import solution


def solver(inp, run, *, store_internal=True):
    """
    Single run solver
    """
    single_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal
    )
    run.solutions["0"] = single_solution
    run.final_solution = run.solutions["0"]
