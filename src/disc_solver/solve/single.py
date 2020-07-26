# -*- coding: utf-8 -*-
"""
Single run solver
"""

from .config import define_conditions
from .solution import solution


def solver(inp, run, *, store_internal=True, no_final_solution=False):
    """
    Single run solver
    """
    single_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal,
    )
    run.solutions.add_solution(single_solution)
    if no_final_solution:
        return True
    run.final_solution = run.solutions.get_last_solution()
    return True
