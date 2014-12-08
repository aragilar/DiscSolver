# -*- coding: utf-8 -*-
"""
The system of odes
"""

from math import pi, sqrt

import logbook

import numpy as np
from scikits.odes import ode

from .utils import cot, is_supersonic
from .deriv_funcs import (
    dderiv_B_φ_soln, dderiv_v_φ_midplane, dderiv_v_r_midplane,
    dderiv_ρ_midplane,
)

INTEGRATOR = "cvode"

log = logbook.Logger(__name__)


def ode_system(
    β, c_s, central_mass, η_O, η_A, η_H,
    taylor_stop_angle, init_con,
):
    """
    Set up the system we are solving for.
    """
    B_θM = init_con[2]
    v_rM = init_con[3]
    v_φM = init_con[4]
    ρ_M = init_con[6]
    deriv_B_φM = init_con[7]

    deriv_v_θM = v_rM * (2 * β - 5/2)
    deriv_B_rM = - (
        B_θM * (1 - β) + (v_rM * B_θM + deriv_B_φM * η_H) / (η_O + η_A)
    )

    dderiv_B_θM = (β - 2) * deriv_B_rM + B_θM

    dderiv_v_rM = dderiv_v_r_midplane(
        ρ_M, B_θM, v_rM, v_φM, deriv_v_θM, deriv_B_rM, deriv_B_φM, dderiv_B_θM,
        β, η_O, η_H, η_A, c_s
    )
    dderiv_ρ_M = dderiv_ρ_midplane(
        ρ_M, B_θM, v_rM, deriv_v_θM, deriv_B_rM, deriv_B_φM, dderiv_v_rM,
        β, c_s
    )
    dderiv_v_φM = dderiv_v_φ_midplane(
        ρ_M, B_θM, v_rM, v_φM, deriv_v_θM, deriv_B_rM, deriv_B_φM, dderiv_B_θM,
        dderiv_v_rM, β, η_O, η_H, η_A, c_s
    )

    log.info("B_θ'': {}".format(dderiv_B_θM))
    log.info("v_r'': {}".format(dderiv_v_rM))
    log.info("v_φ'': {}".format(dderiv_v_φM))
    log.info("ρ'': {}".format(dderiv_ρ_M))

    def rhs_equation(θ, params, derivs):
        """
        Compute the ODEs
        """
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
        deriv_B_r = - (
            B_θ * (1 - β) +
            (
                v_r * B_θ - v_θ * B_r + deriv_B_φ * (
                    η_H * norm_B_θ -
                    η_A * norm_B_r * norm_B_φ
                ) + B_φ * (
                    η_H * (
                        norm_B_θ * cot(θ) -
                        norm_B_r * (1 - β)
                    ) + η_A * norm_B_φ * (
                        norm_B_θ * (1 - β) -
                        norm_B_r * cot(θ)
                    )
                )
            ) / (
                η_O + η_A * (1 - norm_B_φ**2)
            )
        )

        deriv_B_θ = (β - 2) * B_r - B_θ * cot(θ)

        if θ > taylor_stop_angle:
            deriv_v_r = (pi/2 - θ) * dderiv_v_rM
        else:
            deriv_v_r = (
                v_r**2 / (2 * v_θ) +
                v_θ +
                v_φ**2 / v_θ +
                (2 * β * c_s**2) / v_θ -
                central_mass / v_θ +
                (
                    B_θ * deriv_B_r + (β - 1)*(B_θ**2 + B_φ**2)
                ) / (
                    4 * pi * v_θ * ρ
                )
            )

        if θ > taylor_stop_angle:
            deriv_v_φ = (pi/2 - θ) * dderiv_v_φM
        else:
            deriv_v_φ = (
                cot(θ) * v_φ -
                (v_φ * v_r) / (2 * v_θ) +
                (
                    B_θ * B_φ_prime +
                    (1-β) * B_r * B_φ -
                    cot(θ) * B_θ * B_φ
                ) / (
                    4 * pi * v_θ * ρ
                )
            )

        deriv_v_θ = (
            2 * (β-1) * v_r * c_s**2 / (c_s**2 - v_θ**2) - v_r / 2 -
            v_θ / (c_s**2 - v_θ**2) * (
                cot(θ) * v_φ**2 + c_s**2 * cot(θ) - (
                    (β-1) * B_θ * B_r + B_r * deriv_B_r + B_φ * B_φ_prime +
                    B_φ**2 * cot(θ)
                ) / (4*pi * ρ)
            )
        )

        if θ > taylor_stop_angle:
            deriv_ρ = (pi/2 - θ) * dderiv_ρ_M
        else:
            deriv_ρ = - ρ * (
                (
                    (5/2 - 2 * β) * v_r + deriv_v_θ
                ) / v_θ + cot(θ)
            )

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
            v = np.array([0, 0, v_θ])
            B = np.array([B_r, B_φ, B_θ])
            log.info("slow supersonic: {}".format(
                is_supersonic(v, B, ρ, c_s, "slow")))
            log.info("alfven supersonic: {}".format(
                is_supersonic(v, B, ρ, c_s, "alfven")))
            log.info("fast supersonic: {}".format(
                is_supersonic(v, B, ρ, c_s, "fast")))
            log.debug("θ: " + str(θ) + ", " + str(θ/pi*180))
            log.debug("params: " + str(params))
            log.debug("derivs: " + str(derivs))

        return 0
    return rhs_equation


def solution(
        angles, initial_conditions, β, c_s, central_mass,
        η_O, η_A, η_H, relative_tolerance=1e-6,
        absolute_tolerance=1e-10,
        max_steps=500, taylor_stop_angle=pi/2
):
    """
    Find solution
    """
    solver = ode(
        INTEGRATOR,
        ode_system(
            β, c_s, central_mass, η_O, η_A,
            η_H, taylor_stop_angle, initial_conditions
        ),
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
    )
    return solver.solve(angles, initial_conditions)
