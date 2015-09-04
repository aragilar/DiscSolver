# -*- coding: utf-8 -*-
"""
The system of odes
"""

from math import pi, sqrt, tan, degrees, radians

import logbook

import numpy as np

from scikits.odes import ode
import scikits.odes.sundials as sundials

from .deriv_funcs import (
    dderiv_B_φ_soln, taylor_series,
)

from ..file_format import LATEST_NAMESPACE as namespace

INTEGRATOR = "cvode"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)


def ode_system(
    β, c_s, central_mass, η_O, η_A, η_H,
    taylor_stop_angle, init_con,
):
    """
    Set up the system we are solving for.
    """
    dderiv_ρ_M, dderiv_v_rM, dderiv_v_φM = taylor_series(
        β, c_s, η_O, η_A, η_H, init_con
    )

    internal_data = namespace.internal_data(  # pylint: disable=no-member
        derivs=[], params=[], angles=[], v_r_normal=[], v_φ_normal=[],
        ρ_normal=[], v_r_taylor=[], v_φ_taylor=[], ρ_taylor=[],
    )

    derivs_list = internal_data.derivs
    params_list = internal_data.params
    angles_list = internal_data.angles
    v_r_nlist = internal_data.v_r_normal
    v_φ_nlist = internal_data.v_φ_normal
    ρ_nlist = internal_data.ρ_normal
    v_r_taylist = internal_data.v_r_taylor
    v_φ_taylist = internal_data.v_φ_taylor
    ρ_taylist = internal_data.ρ_taylor

    def rhs_equation(θ, params, derivs):
        """
        Compute the ODEs
        """
        nonlocal derivs_list, params_list, angles_list
        nonlocal v_r_nlist, v_r_taylist
        nonlocal v_φ_nlist, v_φ_taylist
        nonlocal ρ_nlist, ρ_taylist

        params_list.append(np.copy(params))

        B_r = params[0]
        B_φ = params[1]
        B_θ = params[2]
        v_r = params[3]
        v_φ = params[4]
        v_θ = params[5]
        ρ = params[6]
        B_φ_prime = params[7]

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
        if __debug__:
            if central_mass < (v_θ * deriv_v_r_normal + central_mass):
                log.info("not grav dominant at {}".format(degrees(θ)))

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

        dderiv_B_φ = dderiv_B_φ_soln(
            B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r,
            v_θ, v_φ, deriv_v_r, deriv_v_θ, deriv_v_φ,
            deriv_B_r, deriv_B_θ, deriv_B_φ, β
        )

        derivs[0] = deriv_B_r
        derivs[1] = deriv_B_φ
        derivs[2] = deriv_B_θ
        derivs[3] = deriv_v_r
        derivs[4] = deriv_v_φ
        derivs[5] = deriv_v_θ
        derivs[6] = deriv_ρ
        derivs[7] = dderiv_B_φ
        if __debug__:
            log.debug("θ: {}, {}", θ, degrees(θ))

        derivs_list.append(np.copy(derivs))
        angles_list.append(θ)
        v_r_nlist.append(deriv_v_r_normal)
        v_φ_nlist.append(deriv_v_φ_normal)
        ρ_nlist.append(deriv_ρ_normal)
        v_r_taylist.append(deriv_v_r_taylor)
        v_φ_taylist.append(deriv_v_φ_taylor)
        ρ_taylist.append(deriv_ρ_taylor)

        return 0
    return rhs_equation, internal_data


def solution(
        angles, initial_conditions, β, c_s, central_mass,
        η_O, η_A, η_H, relative_tolerance=1e-6,
        absolute_tolerance=1e-10,
        max_steps=500, taylor_stop_angle=0
):
    """
    Find solution
    """
    system, internal_data = ode_system(
        β, c_s, central_mass, η_O, η_A, η_H,
        radians(taylor_stop_angle), initial_conditions
    )
    solver = ode(
        INTEGRATOR, system,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        err_handler=ode_error_handler
    )
    try:
        soln = solver.solve(angles, initial_conditions)
    except sundials.CVODESolveFailed as e:
        soln = e.soln
        log.error(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    return (
        soln.values.t, soln.values.y, internal_data,
        namespace.solution_properties(  # pylint: disable=no-member
            flag=soln.flag,
            coordinate_system=COORDS,
        )
    )


def ode_error_handler(error_code, module, func, msg, user_data):
    """ drop all CVODE messages """
    # pylint: disable=unused-argument
    pass
