# -*- coding: utf-8 -*-
"""
Get around sonic point by jumping
"""
# pylint: skip-file
from math import degrees, radians
import logbook

from numpy import concatenate
from scikits.odes.sundials.cvode import StatusEnum

from .config import define_conditions
from .solution import solution
from .utils import (
    onroot_stop, validate_solution, closest_less_than_index,
)
from ..file_format import Solution
from ..utils import ODEIndex

log = logbook.Logger(__name__)


def solver(inp, run, store_internal=False):
    """
    Find solution by jumping over sonic point
    """
    raise RuntimeError("Broken by changes, needs fixing")
    cons = define_conditions(inp)
    sonic_point = find_sonic_point(inp, cons)
    log.notice("sonic point: {}".format(degrees(sonic_point)))
    if inp.jump_before_sonic is None:
        raise RuntimeError("Missing sonic jump location")
    jump_point = sonic_point - radians(inp.jump_before_sonic)
    log.notice("jump_point: {}".format(degrees(jump_point)))

    soln_before_jump, internal_data_before_jump, coords_before_jump = solution(
        cons.angles, cons.init_con, cons.β, cons.c_s, cons.norm_kepler_sq,
        relative_tolerance=inp.relative_tolerance,
        absolute_tolerance=inp.absolute_tolerance,
        max_steps=inp.max_steps, taylor_stop_angle=inp.taylor_stop_angle,
        tstop=jump_point, η_derivs=inp.η_derivs,
        store_internal=store_internal,
    )

    if not validate_solution(Solution(
        solution_input=inp, initial_conditions=cons,
        flag=soln_before_jump.flag, coordinate_system=coords_before_jump,
        internal_data=internal_data_before_jump,
        angles=soln_before_jump.values.t, solution=soln_before_jump.values.y,
        t_roots=soln_before_jump.roots.t, y_roots=soln_before_jump.roots.y,
    )):
        raise RuntimeError("Failed to reach jump point correctly")

    jump_derivs = internal_data_before_jump.derivs
    jump_all_angles = internal_data_before_jump.angles
    internal_index = closest_less_than_index(
        jump_all_angles, jump_point
    )
    log.notice("Stopped before jump point: {}".format(
        degrees(soln_before_jump.values.t[-1]) < degrees(jump_point)
    ))
    log.notice("Solver finished with flag: {!r}".format(
        StatusEnum(soln_before_jump.flag)
    ))
    log.notice("Final velocity: {}".format(
        soln_before_jump.values.y[-1][ODEIndex.v_θ]
    ))
    log.notice(jump_derivs[internal_index][ODEIndex.v_θ])

    end_jump_point = jump_point + (
        cons.c_s - soln_before_jump.values.y[-1][ODEIndex.v_θ] + 0.1
    ) / jump_derivs[internal_index][ODEIndex.v_θ]

    log.notice("end_jump_point: {}".format(degrees(end_jump_point)))
    if degrees(end_jump_point) > inp.stop:
        raise RuntimeError("Jump too big")
    if end_jump_point < jump_point:
        raise RuntimeError("Crossed sonic point in presonic solution")

    jump_vals = soln_before_jump.values.y[-1] + jump_derivs[internal_index] * (
        end_jump_point - jump_point
    )

    jump_angles = cons.angles[cons.angles > end_jump_point]

    log.notice(degrees(jump_angles[0]))

    soln_after_jump, internal_data_after_jump, coords_after_jump = solution(
        jump_angles, jump_vals, cons.β, cons.c_s, cons.norm_kepler_sq,
        relative_tolerance=inp.relative_tolerance,
        absolute_tolerance=inp.absolute_tolerance,
        max_steps=inp.max_steps, taylor_stop_angle=0,
        find_sonic_point=False,
        store_internal=store_internal,
    )

    if not validate_solution(Solution(
        solution_input=inp, initial_conditions=cons, flag=soln_after_jump.flag,
        coordinate_system=coords_after_jump,
        internal_data=internal_data_after_jump,
        angles=soln_after_jump.values.t, solution=soln_after_jump.values.y,
        t_roots=soln_after_jump.roots.t, y_roots=soln_after_jump.roots.y,
    )):
        raise RuntimeError("Solution after jump is invalid")

    if coords_after_jump != coords_before_jump:
        raise RuntimeError("Different coordinate systems used")
    coords = coords_after_jump

    internal_data = internal_data_before_jump + internal_data_after_jump

    angles = concatenate(
        (soln_before_jump.values.t, soln_after_jump.values.t),
        axis=0
    )

    final_vals = concatenate(
        (soln_before_jump.values.y, soln_after_jump.values.y),
        axis=0
    )

    t_roots = None
    y_roots = None

    final_solution = Solution(
        solution_input=inp, initial_conditions=cons, flag=soln_after_jump.flag,
        coordinate_system=coords, internal_data=internal_data, angles=angles,
        solution=final_vals, t_roots=t_roots, y_roots=y_roots,
    )
    run.final_solution = final_solution


def find_sonic_point(inp, cons):
    """
    Find the sonic point based on the input and initial conditions
    """
    soln, _, _ = solution(
        cons.angles, cons.init_con, cons.β, cons.c_s, cons.norm_kepler_sq,
        relative_tolerance=inp.relative_tolerance,
        absolute_tolerance=inp.absolute_tolerance,
        max_steps=inp.max_steps, taylor_stop_angle=inp.taylor_stop_angle,
        onroot_func=onroot_stop, find_sonic_point=True, η_derivs=inp.η_derivs,
    )
    if soln.roots.t is None:
        raise RuntimeError("Solver failed to reach sonic point")
    return soln.roots.t[0]
