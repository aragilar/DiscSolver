# -*- coding: utf-8 -*-
"""
Single run solver
"""

from .config import define_conditions
from .solution import solution

from ..file_format import Solution


def solver(inp, run):
    """
    Single run solver
    """
    cons = define_conditions(inp)
    soln, internal_data, coords = solution(
        cons.angles, cons.init_con, cons.β, cons.c_s, cons.norm_kepler_sq,
        absolute_tolerance=inp.absolute_tolerance,
        relative_tolerance=inp.relative_tolerance,
        max_steps=inp.max_steps, taylor_stop_angle=inp.taylor_stop_angle,
        η_derivs=inp.η_derivs,
    )
    final_solution = Solution(
        solution_input=inp, initial_conditions=cons, flag=soln.flag,
        coordinate_system=coords, internal_data=internal_data,
        angles=soln.values.t, solution=soln.values.y, t_roots=soln.roots.t,
        y_roots=soln.roots.y,
    )
    run.final_solution = final_solution
