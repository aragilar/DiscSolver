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
        inp, define_conditions(inp, use_E_r=run.use_E_r),
        store_internal=store_internal, use_E_r=run.use_E_r,
    )
    run.solutions["0"] = single_solution
    run.final_solution = run.solutions["0"]
