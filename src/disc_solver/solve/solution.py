# -*- coding: utf-8 -*-
"""
The system of odes
"""

from math import pi, sqrt, tan, degrees, radians

import logbook

from numpy import copy

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)

from .deriv_funcs import dderiv_B_φ_soln, taylor_series
from .utils import gen_sonic_point_rootfn, ode_error_handler

from ..file_format import InternalData
from ..utils import ODEIndex

INTEGRATOR = "cvode"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)


def ode_system(
    β, c_s, central_mass, taylor_stop_angle, init_con, η_derivs=True,
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

    def rhs_equation(θ, params, derivs):
        """
        Compute the ODEs
        """
        nonlocal derivs_list, params_list, angles_list
        nonlocal v_r_nlist, v_r_taylist
        nonlocal v_φ_nlist, v_φ_taylist
        nonlocal ρ_nlist, ρ_taylist

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
            problems[θ].append("negative density")
            return 1
        if v_θ < 0:
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
            deriv_v_r_taylor if θ < taylor_stop_angle else deriv_v_r_normal
        )

        deriv_v_φ_taylor = θ * dderiv_v_φM
        deriv_v_φ_normal = (
            (
                B_θ * B_φ_prime +
                (1-β) * B_r * B_φ +
                tan(θ) * B_θ * B_φ
            ) / (
                4 * pi * ρ
            ) - tan(θ) * v_φ * v_θ -
            (v_φ * v_r) / 2
        ) / v_θ
        deriv_v_φ = (
            deriv_v_φ_taylor if θ < taylor_stop_angle else deriv_v_φ_normal
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
        deriv_ρ = deriv_ρ_taylor if θ < taylor_stop_angle else deriv_ρ_normal

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

        params_list.append(copy(params))
        derivs_list.append(copy(derivs))
        angles_list.append(θ)
        v_r_nlist.append(deriv_v_r_normal)
        v_φ_nlist.append(deriv_v_φ_normal)
        ρ_nlist.append(deriv_ρ_normal)
        v_r_taylist.append(deriv_v_r_taylor)
        v_φ_taylist.append(deriv_v_φ_taylor)
        ρ_taylist.append(deriv_ρ_taylor)

        if __debug__:
            if len(params_list) != len(angles_list):
                log.error(
                    "Internal data not consistent, "
                    "params is {}, angles is {}".format(
                        len(params_list), len(angles_list)
                    )
                )

        return 0
    return rhs_equation, internal_data


def solution(
        angles, initial_conditions, β, c_s, central_mass,
        relative_tolerance=1e-6, absolute_tolerance=1e-10,
        max_steps=500, taylor_stop_angle=0, onroot_func=None,
        find_sonic_point=False, tstop=None, ontstop_func=None,
        η_derivs=True,
):
    """
    Find solution
    """
    extra_args = {}
    if find_sonic_point:
        extra_args["rootfn"] = gen_sonic_point_rootfn(c_s)
        extra_args["nr_rootfns"] = 1

    system, internal_data = ode_system(
        β, c_s, central_mass, radians(taylor_stop_angle), initial_conditions,
        η_derivs=η_derivs,
    )
    solver = ode(
        INTEGRATOR, system,
        linsolver="lapackdense",
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
        soln = solver.solve(angles, initial_conditions)
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

    internal_data.finalise()

    return (
        soln, internal_data, COORDS,
    )
