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
        inp, define_conditions(inp, use_E_r=run.use_E_r),
        store_internal=store_internal, use_E_r=run.use_E_r,
    )
    run.solutions.add_solution(single_solution)
    if no_final_solution:
        return True
    run.final_solution = run.solutions.get_last_solution()
    return True
