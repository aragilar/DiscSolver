# -*- coding: utf-8 -*-
"""
Solver for PHD Project
"""

__version__ = "0.1"
from math import pi, sqrt

import logbook

import numpy as np
from scikits.odes import ode
import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from .utils import cot, is_supersonic
from .constants import G, AU, M_SUN, KM
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


def main():
    """
    The main function
    """

    start = 90
    stop = 85
    angles = np.linspace(start, stop, 10000) / 180 * pi
    taylor_stop_angle = 89.99

    # pick a radii, 1AU makes it easy to calculate
    radius = 1 * AU

    central_mass = 1 * M_SUN

    β = 4/3

    # B_θ is the equipartition field
    B_θ = 18  # G

    # from wardle 2007 for 1 AU
    # assume B ~ 1 G
    η_O = 5e15  # cm^2/s
    η_H = 5e16  # cm^2/s
    η_A = 1e14  # cm^2/s

    c_s = 0.99 * KM  # actually in cm/s for conversions
    # ρ computed from Wardle 2007
    ρ = 1.5e-9  # g/cm^3

    keplerian_velocity = sqrt(G * central_mass / radius)  # cm/s

    v_r = - 0.1 * keplerian_velocity

    v_θ = 0  # symmetry across disc
    B_r = 0  # symmetry across disc
    B_φ = 0  # symmetry across disc

    # solution for A * v_φ**2 + B * v_φ + C = 0
    A_v_φ = 1
    B_v_φ = - (v_r * η_H) / (2 * (η_O + η_A))
    C_v_φ = (
        v_r**2 / 2 + 2 * β * c_s**2 -
        keplerian_velocity**2 +
        B_θ**2 * (
            2 * (β - 1) - v_r / (η_O + η_A)
        ) / (4 * pi * ρ)
    )
    v_φ = - 1/2 * (B_v_φ - sqrt(B_v_φ**2 - 4 * A_v_φ * C_v_φ))

    B_φ_prime = (v_φ * v_r * 4 * pi * ρ) / (2 * B_θ)

    log.debug("A_v_φ: {}".format(A_v_φ))
    log.debug("B_v_φ: {}".format(B_v_φ))
    log.debug("C_v_φ: {}".format(C_v_φ))
    log.info("v_φ: {}".format(v_r))
    log.info("B_φ_prime: {}".format(B_φ_prime))

    if v_r > 0:
        log.error("v_r > 0")
        exit(1)

    v_norm = c_s
    B_norm = B_θ
    diff_norm = v_norm * radius

    ρ_norm = B_norm**2 / (4 * pi * v_norm**2)

    init_con = np.zeros(8)

    init_con[0] = B_r / B_norm
    init_con[1] = B_φ / B_norm
    init_con[2] = B_θ / B_norm
    init_con[3] = v_r / v_norm
    init_con[4] = v_φ / v_norm
    init_con[5] = v_θ / v_norm
    init_con[6] = ρ / ρ_norm
    init_con[7] = B_φ_prime / B_norm

    norm_kepler_sq = keplerian_velocity**2 / v_norm**2
    c_s = c_s / v_norm
    η_O = η_O / diff_norm
    η_A = η_A / diff_norm
    η_H = η_H / diff_norm

    try:
        soln = solution(
            angles, init_con, β, c_s, norm_kepler_sq, η_O,
            η_A, η_H, max_steps=10000,
            taylor_stop_angle=taylor_stop_angle / 180 * pi
        )
    except RuntimeError as e:
        # pylint: disable=no-member
        angles = e.x_vals
        soln = e.y_vals

    param_names = [
        {
            "name": "B_r",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_φ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "B_θ",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
        {
            "name": "v_r",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
        },
        {
            "name": "v_φ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
        },
        {
            "name": "v_θ",
            "y_label": "Velocity Field (km/s)",
            "normalisation": v_norm / KM,  # km/s
        },
        {
            "name": "ρ",
            "y_label": "Density ($g cm^{-3}$)",
            "normalisation": ρ_norm,
            "scale": "log",
        },
        {
            "name": "B_φ_prime",
            "y_label": "Magnetic Field (Gauss)",
            "normalisation": B_norm,
        },
    ]

    fig, axes = plt.subplots(nrows=2, ncols=4, tight_layout=True)
    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            90 - (angles * 180 / pi),
            soln[:, i] * settings["normalisation"]
        )
        ax.set_xlabel("angle from plane (°)")
        ax.set_ylabel(settings["y_label"])
        ax.set_yscale(settings.get("scale", "linear"))
        ax.set_title(settings["name"])
    fig.savefig("plot.png")
    plt.show()
