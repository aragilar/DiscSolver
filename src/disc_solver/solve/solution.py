# -*- coding: utf-8 -*-
"""
The system of odes
"""

from math import pi, sqrt, tan, degrees, radians
from collections import namedtuple

import logbook

from numpy import concatenate, copy, insert

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)

from .deriv_funcs import dderiv_B_φ_soln, taylor_series
from .utils import gen_sonic_point_rootfn, ode_error_handler

from ..file_format import InternalData, Solution
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "lapackdense"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)

TaylorSolution = namedtuple(
    "TaylorSolution", [
        "angles", "params", "new_angles", "internal_data",
        "new_initial_conditions", "angle_stopped",
    ]
)


def ode_system(
    β, c_s, central_mass, init_con, *, with_taylor=False, η_derivs=True,
    store_internal=True
):
    """
    Set up the system we are solving for.
    """
    dderiv_ρ_M, dderiv_v_rM, dderiv_v_φM = taylor_series(
        β, c_s, init_con
    )
    if η_derivs:
        η_O_scale = init_con[ODEIndex.η_O] / sqrt(init_con[ODEIndex.ρ])
        η_A_scale = init_con[ODEIndex.η_A] / sqrt(init_con[ODEIndex.ρ])
        η_H_scale = init_con[ODEIndex.η_H] / sqrt(init_con[ODEIndex.ρ])
    else:
        η_O_scale = 0
        η_A_scale = 0
        η_H_scale = 0

    if store_internal:
        internal_data = InternalData()
        derivs_list = internal_data.derivs
        params_list = internal_data.params
        angles_list = internal_data.angles
        v_r_nlist = internal_data.v_r_normal
        v_φ_nlist = internal_data.v_φ_normal
        ρ_nlist = internal_data.ρ_normal
        v_r_taylist = internal_data.v_r_taylor
        v_φ_taylist = internal_data.v_φ_taylor
        ρ_taylist = internal_data.ρ_taylor
        problems = internal_data.problems
    else:
        internal_data = None

    def rhs_equation(θ, params, derivs):
        """
        Compute the ODEs
        """
        B_r = params[ODEIndex.B_r]
        B_φ = params[ODEIndex.B_φ]
        B_θ = params[ODEIndex.B_θ]
        v_r = params[ODEIndex.v_r]
        v_φ = params[ODEIndex.v_φ]
        v_θ = params[ODEIndex.v_θ]
        ρ = params[ODEIndex.ρ]
        B_φ_prime = params[ODEIndex.B_φ_prime]
        η_O = params[ODEIndex.η_O]
        η_A = params[ODEIndex.η_A]
        η_H = params[ODEIndex.η_H]

        # check sanity of input values
        if ρ < 0:
            if store_internal:
                problems[θ].append("negative density")
            return 1
        if v_θ < 0:
            if store_internal:
                problems[θ].append("negative velocity")
            return -1

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
        norm_B_r, norm_B_φ, norm_B_θ = (
            B_r/B_mag, B_φ/B_mag, B_θ/B_mag
        )

        deriv_B_φ = B_φ_prime
        deriv_B_r = (
            B_θ * (1 - β) +
            (
                v_θ * B_r - v_r * B_θ + deriv_B_φ * (
                    η_H * norm_B_θ -
                    η_A * norm_B_r * norm_B_φ
                ) + B_φ * (
                    η_H * (
                        norm_B_r * (1 - β) -
                        norm_B_θ * tan(θ)
                    ) - η_A * norm_B_φ * (
                        norm_B_θ * (1 - β) -
                        norm_B_r * tan(θ)
                    )
                )
            ) / (
                η_O + η_A * (1 - norm_B_φ**2)
            )
        )

        deriv_B_θ = (β - 2) * B_r + B_θ * tan(θ)

        deriv_v_r_taylor = θ * dderiv_v_rM
        deriv_v_r_normal = (
            v_r**2 / 2 +
            v_θ**2 + v_φ**2 +
            (2 * β * c_s**2) - central_mass +
            (
                B_θ * deriv_B_r + (β - 1)*(B_θ**2 + B_φ**2)
            ) / (
                4 * pi * ρ
            )
        ) / v_θ
        deriv_v_r = (
            deriv_v_r_taylor if with_taylor else deriv_v_r_normal
        )

        deriv_v_φ_taylor = θ * dderiv_v_φM
        deriv_v_φ_normal = (
            (
                B_θ * B_φ_prime +
                (1-β) * B_r * B_φ -
                tan(θ) * B_θ * B_φ
            ) / (
                4 * pi * ρ
            ) + tan(θ) * v_φ * v_θ -
            (v_φ * v_r) / 2
        ) / v_θ
        deriv_v_φ = (
            deriv_v_φ_taylor if with_taylor else deriv_v_φ_normal
        )

        deriv_v_θ = (
            v_r * (
                v_θ**2 / 2 + c_s**2 * (2 * β - 5/2)
            ) + v_θ * (
                tan(θ) * (v_φ**2 + c_s**2) + (
                    (β-1) * B_θ * B_r + B_r * deriv_B_r + B_φ * B_φ_prime -
                    B_φ**2 * tan(θ)
                ) / (4*pi * ρ)
            )
        ) / ((c_s - v_θ) * (c_s + v_θ))

        deriv_ρ_taylor = θ * dderiv_ρ_M
        deriv_ρ_normal = - ρ * (
            (
                (5/2 - 2 * β) * v_r + deriv_v_θ
            ) / v_θ - tan(θ)
        )
        deriv_ρ = deriv_ρ_taylor if with_taylor else deriv_ρ_normal

        deriv_ρ_scale = deriv_ρ / sqrt(ρ) / 2
        deriv_η_O = deriv_ρ_scale * η_O_scale
        deriv_η_A = deriv_ρ_scale * η_A_scale
        deriv_η_H = deriv_ρ_scale * η_H_scale

        dderiv_B_φ = dderiv_B_φ_soln(
            B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r,
            v_θ, v_φ, deriv_v_r, deriv_v_θ, deriv_v_φ,
            deriv_B_r, deriv_B_θ, deriv_B_φ, β, deriv_η_O, deriv_η_A,
            deriv_η_H
        )

        derivs[ODEIndex.B_r] = deriv_B_r
        derivs[ODEIndex.B_φ] = deriv_B_φ
        derivs[ODEIndex.B_θ] = deriv_B_θ
        derivs[ODEIndex.v_r] = deriv_v_r
        derivs[ODEIndex.v_φ] = deriv_v_φ
        derivs[ODEIndex.v_θ] = deriv_v_θ
        derivs[ODEIndex.ρ] = deriv_ρ
        derivs[ODEIndex.B_φ_prime] = dderiv_B_φ
        derivs[ODEIndex.η_O] = deriv_η_O
        derivs[ODEIndex.η_A] = deriv_η_A
        derivs[ODEIndex.η_H] = deriv_η_H
        if __debug__:
            log.debug("θ: {}, {}", θ, degrees(θ))

        if store_internal:
            params_list.append(copy(params))
            derivs_list.append(copy(derivs))
            angles_list.append(θ)
            v_r_nlist.append(deriv_v_r_normal)
            v_φ_nlist.append(deriv_v_φ_normal)
            ρ_nlist.append(deriv_ρ_normal)
            v_r_taylist.append(deriv_v_r_taylor)
            v_φ_taylist.append(deriv_v_φ_taylor)
            ρ_taylist.append(deriv_ρ_taylor)

            if len(params_list) != len(angles_list):
                log.error(
                    "Internal data not consistent, "
                    "params is {}, angles is {}".format(
                        len(params_list), len(angles_list)
                    )
                )

        return 0
    return rhs_equation, internal_data


def taylor_solution(
    angles, initial_conditions, β, c_s, central_mass, *, taylor_stop_angle,
    relative_tolerance=1e-6, absolute_tolerance=1e-10, max_steps=500,
    η_derivs=True, store_internal=True
):
    """
    Compute solution using taylor series
    """
    system, internal_data = ode_system(
        β, c_s, central_mass, initial_conditions, with_taylor=True,
        η_derivs=η_derivs, store_internal=store_internal,
    )

    solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=ode_error_handler,
        tstop=taylor_stop_angle
    )

    try:
        soln = solver.solve(angles, initial_conditions)
    except CVODESolveFailed as e:
        RuntimeError(
            "Taylor solver stopped in at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
    else:
        raise RuntimeError("Taylor solver did not find taylor_stop_angle")

    new_angles = angles[angles > taylor_stop_angle]
    insert(new_angles, 0, taylor_stop_angle)

    return TaylorSolution(
        angles=soln.values.t, params=soln.values.y, new_angles=new_angles,
        internal_data=internal_data, new_initial_conditions=soln.tstop.y[0],
        angle_stopped=soln.tstop.t[0],
    )


def main_solution(
    angles, system_initial_conditions, ode_initial_conditions, β, c_s,
    central_mass, *, relative_tolerance=1e-6, absolute_tolerance=1e-10,
    max_steps=500, onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, η_derivs=True, store_internal=True
):
    """
    Find solution
    """
    extra_args = {}
    if find_sonic_point:
        extra_args["rootfn"] = gen_sonic_point_rootfn(c_s)
        extra_args["nr_rootfns"] = 1

    system, internal_data = ode_system(
        β, c_s, central_mass, system_initial_conditions,
        η_derivs=η_derivs, store_internal=store_internal, with_taylor=False,
    )
    solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=ode_error_handler,
        onroot=onroot_func,
        tstop=tstop,
        ontstop=ontstop_func,
        **extra_args
    )

    try:
        soln = solver.solve(angles, ode_initial_conditions)
    except CVODESolveFailed as e:
        soln = e.soln
        log.warn(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    except CVODESolveFoundRoot as e:
        soln = e.soln
        for root in soln.roots.t:
            log.notice("Found sonic point at {}".format(degrees(root)))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        for tstop in soln.tstop.t:
            log.notice("Stopped at {}".format(degrees(tstop)))

    return soln, internal_data


def solution(
    input, initial_conditions, *,
    onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, store_internal=True
):
    """
    Find solution
    """
    angles = initial_conditions.angles
    init_con = initial_conditions.init_con
    β = initial_conditions.β
    c_s = initial_conditions.c_s
    central_mass = initial_conditions.norm_kepler_sq
    absolute_tolerance = input.absolute_tolerance
    relative_tolerance = input.relative_tolerance
    max_steps = input.max_steps
    taylor_stop_angle = radians(input.taylor_stop_angle)
    η_derivs = input.η_derivs

    if taylor_stop_angle is None:
        post_taylor_angles = angles
        post_taylor_initial_conditions = init_con
    else:
        taylor_soln = taylor_solution(
            angles, init_con, β, c_s, central_mass,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, max_steps=max_steps,
            taylor_stop_angle=taylor_stop_angle, η_derivs=η_derivs,
            store_internal=store_internal
        )
        post_taylor_angles = taylor_soln.new_angles
        post_taylor_initial_conditions = taylor_soln.new_initial_conditions
        taylor_internal = taylor_soln.internal_data

    soln, internal_data = main_solution(
        post_taylor_angles, init_con, post_taylor_initial_conditions, β, c_s,
        central_mass, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal,
        find_sonic_point=find_sonic_point
    )

    if store_internal:
        internal_data.finalise()
        if taylor_stop_angle is not None:
            taylor_internal.finalise()
            internal_data = taylor_internal + internal_data

    if taylor_stop_angle is None:
        joined_angles = soln.values.t
        joined_solution = soln.values.y
    else:
        joined_angles = concatenate((taylor_soln.angles, soln.values.t))
        joined_solution = concatenate((taylor_soln.params, soln.values.y))

    return Solution(
        solution_input=input, initial_conditions=initial_conditions,
        flag=soln.flag, coordinate_system=COORDS, internal_data=internal_data,
        angles=joined_angles, solution=joined_solution, t_roots=soln.roots.t,
        y_roots=soln.roots.y
    )
