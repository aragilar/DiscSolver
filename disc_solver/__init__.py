# -*- coding: utf-8 -*-
"""
"""

import sys
from math import pi, cos, sin, sqrt
from functools import wraps

import logbook
from logbook.compat import redirected_warnings, redirected_logging

import numpy as np
from scikits.odes import ode
import matplotlib as mpl
mpl.use("Qt4Agg")
mpl.rcParams["backend.qt4"] = "PySide"
import matplotlib.pyplot as plt

from ode_wrapper import validate_cvode_flags

INTEGRATOR = "cvode"

# physical constants
G = 6.6742e-8 # (cm^3 ∕ (g*s^2))
AU = 1.4957871e13 # cm
M_SUN = 2e33 # g
KM = 1e5 # cm

log = logbook.Logger(__name__)

def take_multiple_dims(*multi_dim_pos):
    def decorator(func):
        @wraps(func)
        def wrapper(*args):
            have_multi_dims = {
                pos: bool(getattr(args[pos], "ndim", 0) > 1)
                for pos in multi_dim_pos
            }
            return func(*args, have_multi_dims=have_multi_dims)
        return wrapper
    return decorator


@take_multiple_dims(0,1)
def is_supersonic_slow(v, B, rho, sound_speed, have_multi_dims):
    v_axis = 1 if have_multi_dims[0] else 0
    B_axis = 1 if have_multi_dims[1] else 0

    v_sq = np.sum(v**2, axis=v_axis)
    B_sq = np.sum(B**2, axis=B_axis)
    cos_sq_psi = np.sum(
            (v.T/v_sq).T * (B.T/B_sq).T, axis=max(v_axis, B_axis)
        ) ** 2
    v_a_sq = B_sq /(4*pi*rho)
    v_sq_slow = 1/2 * (
        v_a_sq + sound_speed**2 - np.sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    return v_sq > v_sq_slow

@take_multiple_dims(0,1)
def is_supersonic_alfven(v, B, rho, have_multi_dims):
    v_axis = 1 if have_multi_dims[0] else 0
    B_axis = 1 if have_multi_dims[1] else 0

    v_sq = np.sum(v**2, axis=v_axis)
    B_sq = np.sum(B**2, axis=B_axis)
    cos_sq_psi = np.sum(
            (v.T/v_sq).T * (B.T/B_sq).T, axis=max(v_axis, B_axis)
        ) ** 2
    v_a_sq = B_sq /(4*pi*rho) * cos_sq_psi
    return v_sq > v_a_sq

@take_multiple_dims(0,1)
def is_supersonic_fast(v, B, rho, sound_speed, have_multi_dims):
    v_axis = 1 if have_multi_dims[0] else 0
    B_axis = 1 if have_multi_dims[1] else 0

    v_sq = np.sum(v**2, axis=v_axis)
    B_sq = np.sum(B**2, axis=B_axis)
    cos_sq_psi = np.sum(
            (v.T/v_sq).T * (B.T/B_sq).T, axis=max(v_axis, B_axis)
        ) ** 2
    v_a_sq = B_sq /(4*pi*rho)
    v_sq_fast = 1/2 * (
        v_a_sq + sound_speed**2 + np.sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    return v_sq > v_sq_fast

def cot(angle):
    if angle == pi:
        return float('inf')
    elif angle == pi/2:
        return 0
    return cos(angle)/sin(angle)

def B_unit_derivs(B_r, B_phi, B_theta, deriv_B_r, deriv_B_phi, deriv_B_theta):
    B_mag = sqrt(B_r**2 + B_phi**2 + B_theta **2)
    norm_B_r, norm_B_phi, norm_B_theta = B_r/B_mag, B_phi/B_mag, B_theta/B_mag
    return [
            1/B_mag * (
                B_deriv + B_unit/B_mag * (
                    B_r * deriv_B_r + B_phi * deriv_B_phi +
                    B_theta * deriv_B_theta
            ))

            for B_unit, B_deriv in (
                (norm_B_r, deriv_B_r), (norm_B_phi, deriv_B_phi),
                (norm_B_theta, deriv_B_theta)
        )]

def dderiv_B_phi_soln(
    B_r, B_phi, B_theta, ohm_diff, hall_diff, abi_diff, angle, v_r, v_theta,
    v_phi, deriv_v_r, deriv_v_theta, deriv_v_phi, deriv_B_r, deriv_B_theta,
    deriv_B_phi, B_power
):
    B_mag = sqrt(B_r**2 + B_phi**2 + B_theta **2)
    norm_B_r, norm_B_phi, norm_B_theta = B_r/B_mag, B_phi/B_mag, B_theta/B_mag
    deriv_norm_B_r, deriv_norm_B_phi, deriv_norm_B_theta = B_unit_derivs(
            B_r, B_phi, B_theta, deriv_B_r, deriv_B_phi, deriv_B_theta)

    # Solution to \frac{∂}{∂θ}\frac{η_{H0}b_{θ} + η_{A0} b_{r}b_{φ}}{η_{O0} +
    #  η_{A0}\left(1-b_{φ}^{2}\right)}$
    magic_deriv = (
            (
                hall_diff * deriv_norm_B_theta + abi_diff * (
                    norm_B_r * deriv_norm_B_phi +
                    norm_B_phi * deriv_norm_B_r
            ) * (
                ohm_diff + abi_diff * (1 - norm_B_phi)
            ) + (
                abi_diff * 2 * deriv_norm_B_phi * norm_B_phi
            ) * (
                hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
            )
        ) / (ohm_diff + abi_diff * (1 - norm_B_phi))**2)

    return (
            ohm_diff + abi_diff * (1-norm_B_r**2) + (
                hall_diff**2 * norm_B_theta**2 -
                abi_diff**2 * norm_B_phi**2 * norm_B_r**2
            ) / (
                ohm_diff + abi_diff * (1-norm_B_phi**2)
            )
        ) ** -1 * (
            deriv_B_r * (
                v_theta * (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) - hall_diff * norm_B_r + abi_diff * norm_B_theta * norm_B_phi
            ) + B_r * (
                v_phi + deriv_v_theta * (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) + v_theta * magic_deriv
            ) - B_theta * (
                deriv_v_phi * 2 / (2 * B_power - 1) + deriv_v_r * (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) + v_r * magic_deriv + (1-B_power) * (
                        hall_diff * norm_B_r -
                        abi_diff * norm_B_theta * norm_B_phi
                    )
            ) - deriv_B_theta * (
                v_r * (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) + 2 / (2 * B_power - 1) * v_phi
            ) +
            deriv_B_phi * (
                hall_diff * norm_B_phi + abi_diff * norm_B_r * norm_B_theta +
                v_theta * 2 / (2 * B_power - 1) - ohm_diff * cot(angle) -
                hall_diff * norm_B_phi * (1-B_power) - abi_diff * (
                    cot(angle) * (1 - norm_B_r **2) -
                    (1 - B_power) * norm_B_r * norm_B_theta
                ) - (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) * (
                    hall_diff * (
                        norm_B_theta * cot(angle) - norm_B_r * (1 - B_power)
                    ) + abi_diff * norm_B_phi * (
                        norm_B_theta * (1 - B_power) - norm_B_r * cot(angle)
                    )
                )
            ) - B_phi * (
                ohm_diff * (1 - B_power) -
                hall_diff * norm_B_phi * cot(angle) +
                abi_diff * (
                    (1 - B_power)* (1 - norm_B_theta**2) - 
                    cot(angle) * norm_B_r * norm_B_theta
                ) + v_r - 2 / (2 * B_power - 1) * deriv_v_theta +
                ohm_diff * sin(angle)**-2 -
                hall_diff * deriv_norm_B_phi * (1 - B_power) +
                abi_diff * (
                    2 * cot(angle) * norm_B_r * deriv_norm_B_r -
                    sin(angle)**-2 * norm_B_r **2 +
                    (1 - B_power)* (
                        norm_B_r * deriv_norm_B_theta +
                        norm_B_theta * deriv_norm_B_r
                    )
                ) - magic_deriv * (
                        hall_diff * (
                            norm_B_theta * cot(angle) - norm_B_r * (1 - B_power)
                        ) + abi_diff * norm_B_phi * (
                            norm_B_theta * (1 - B_power) - norm_B_r * cot(angle)
                        )
                ) - (
                    hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
                ) / (
                    ohm_diff + abi_diff * (1-norm_B_phi**2)
                ) * (
                    hall_diff * (
                        deriv_norm_B_theta * cot(angle) -
                        sin(angle) **-2 * norm_B_theta -
                        deriv_norm_B_r * (1-B_power)
                    ) + abi_diff * (
                        deriv_norm_B_phi * (
                            norm_B_theta * (1 - B_power) - norm_B_r * cot(angle)
                        ) + norm_B_phi * (
                            deriv_norm_B_theta * (1 - B_power) -
                            deriv_norm_B_r * cot(angle) +
                            sin(angle)**-2 * norm_B_r
                        )
                    )
                )
            )
        )


def ode_system(B_power, sound_speed, central_mass, ohm_diff, abi_diff, hall_diff):
    def rhs_equation(angle, params, derivs):
        B_r = params[0]
        B_phi = params[1]
        B_theta = params[2]
        v_r = params[3]
        v_phi = params[4]
        v_theta = params[5]
        rho = params[6]
        B_phi_prime = params[7]

        B_mag = sqrt(B_r**2 + B_phi**2 + B_theta **2)
        norm_B_r, norm_B_phi, norm_B_theta = B_r/B_mag, B_phi/B_mag, B_theta/B_mag

        deriv_B_phi = B_phi_prime
        deriv_B_r = - (
                B_theta * (1 - B_power) +
                (
                    v_r * B_theta - v_theta * B_r + deriv_B_phi * (
                        hall_diff * norm_B_theta -
                        abi_diff * norm_B_r * norm_B_phi
                    ) + B_phi * (
                        hall_diff * (
                            norm_B_theta * cot(angle) -
                            norm_B_r * (1 - B_power)
                        ) + abi_diff * norm_B_phi * (
                            norm_B_theta * (1 - B_power) -
                            norm_B_r * cot(angle)
                        )
                    )
                ) / (
                    ohm_diff + abi_diff * (1 - norm_B_phi**2)
                )
            )
        deriv_B_theta = (B_power - 2) * B_r - B_theta * cot(angle)

        deriv_v_r = (
                v_r**2 / (2 * v_theta) +
                v_theta +
                v_phi**2 / v_theta +
                (2 * B_power * sound_speed**2) / v_theta -
                central_mass / v_theta +
                (
                    B_theta * deriv_B_r + (B_power - 1)*(B_theta**2 + B_phi**2)
                ) / (
                    4 * pi * v_theta * rho
                )
            )

        deriv_v_phi = (
                cot(angle) * v_phi -
                (v_phi * v_r) / (2 * v_theta) +
                (
                    B_theta * B_phi_prime +
                    (1-B_power) * B_r * B_phi -
                    cot(angle) * B_theta * B_phi
                ) / (
                    4 * pi * v_theta * rho
                )
            )

        deriv_v_theta = v_theta / (sound_speed**2 - v_theta**2) * (
                (v_r * v_theta) / 2 -
                cot(angle) * v_phi **2 -
                sound_speed**2 * (
                    (5/2 - 2*B_power)* v_r / v_theta + cot(angle)
                ) + (
                    (B_power - 1) *B_theta * B_r +
                    B_r * deriv_B_r +
                    B_phi * B_phi_prime +
                    B_phi ** 2 * cot(angle)
                ) / (
                    4 * pi * rho
                )
            )

        deriv_rho = - rho * (
                (
                    (5/2 - 2 * B_power) * v_r + deriv_v_theta
                ) / v_theta + cot(angle)
            )

        dderiv_B_phi = dderiv_B_phi_soln(
                B_r, B_phi, B_theta, ohm_diff, hall_diff, abi_diff, angle, v_r,
                v_theta, v_phi, deriv_v_r, deriv_v_theta, deriv_v_phi,
                deriv_B_r, deriv_B_theta, deriv_B_phi, B_power
            )

        derivs[0] = deriv_B_r
        derivs[1] = deriv_B_phi
        derivs[2] = deriv_B_theta
        derivs[3] = deriv_v_r
        derivs[4] = deriv_v_phi
        derivs[5] = deriv_v_theta
        derivs[6] = deriv_rho
        derivs[7] = dderiv_B_phi

        if __debug__:
            v = np.array([v_r, 0, v_theta])
            B = np.array([B_r, B_phi, B_theta])
            log.info("slow supersonic: {}".format(
                is_supersonic_slow(v, B, rho, sound_speed)))
            log.info("alfven supersonic: {}".format(
                is_supersonic_alfven(v, B, rho)))
            log.info("fast supersonic: {}".format(
                is_supersonic_fast(v, B, rho, sound_speed)))
            log.debug("angle: " + str(angle) + ", " + str(angle/pi*180))
            log.debug("params: " + str(params))
            log.debug("derivs: " + str(derivs))

        return 0
    return rhs_equation

def solution(
        angles, initial_conditions, B_power, sound_speed, central_mass,
        ohm_diff, abi_diff, hall_diff, relative_tolerance=1e-6,
        absolute_tolerance=1e-10,
        max_steps=500,
):
    solver = ode(INTEGRATOR,
            ode_system(
                B_power, sound_speed, central_mass, ohm_diff, abi_diff,
                hall_diff
            ),
            rtol=relative_tolerance, atol=absolute_tolerance,
            max_steps=max_steps,
    )
    return validate_cvode_flags(*solver.solve(
            angles, initial_conditions
        ))

def main():
    angles = np.linspace(90, 10, 10000) / 180 * pi

    param_names = [
        "B_r",
        "B_phi",
        "B_theta",
        "v_r",
        "v_phi",
        "v_theta",
        "rho",
        "B_phi_prime",
    ]

    #rho needs computing
    rho = 1.5e-9 # g/cm^3
    #v_phi needs computing
    v_phi = 28 * KM # actually in cm/s for conversions
    #B_theta need computing
    B_theta = 1 # G
    # pick a radii, 1AU makes it easy to calculate
    radius = 1 * AU

    # from wardle 2007 for 1 AU
    # assume B ~ 1 G
    sound_speed = 0.99 * KM # actually in cm/s for conversions
    central_mass = 1 * M_SUN
    B_power = 1
    ohm_diff = 5e15 # cm^2/s
    hall_diff = 5e16 # cm^2/s
    abi_diff = 1e14 # cm^2/s

    #v_theta = 0 # symmetry across disc
    v_theta = 0.2 * KM # actually in cm/s for conversions
    B_r = 0 # symmetry across disc
    B_phi = 0 # symmetry across disc


    # solution for A * v_r**2 + B * v_r + C = 0
    A_v_r = 1/2
    B_v_r = - (
            B_theta **2 / (4 * pi * rho) +
            (v_phi * hall_diff) / 2
        ) / (ohm_diff + abi_diff)
    C_v_r = (
        v_phi**2 + 2 * B_power * sound_speed **2 -
        G * central_mass / radius +
        (B_power - 1) * B_theta**2 / (4 * pi * rho)
    )
    v_r = 1/2 * ( - B_v_r - sqrt(B_v_r**2 - 4 * A_v_r * C_v_r))

    B_phi_prime = (v_phi * v_r * 4 * pi * rho) / (2 * B_theta)

    log.debug("A_v_r: {}".format(A_v_r))
    log.debug("B_v_r: {}".format(B_v_r))
    log.debug("C_v_r: {}".format(C_v_r))
    log.info("v_r: {}".format(v_r))
    log.info("B_phi_prime: {}".format(B_phi_prime))

    if v_r > 0:
        log.error("v_r > 0")
        exit(1)

    v_norm = v_phi
    B_norm = B_theta
    diff_norm = ohm_diff

    rho_norm = B_norm **2 / (4 * pi * v_norm**2)

    init_con = np.zeros(8)

    init_con[0] = B_r / B_norm
    init_con[1] = B_phi / B_norm
    init_con[2] = B_theta / B_norm
    init_con[3] = v_r / v_norm
    init_con[4] = v_phi / v_norm
    init_con[5] = v_theta / v_norm
    init_con[6] = rho / rho_norm
    init_con[7] = B_phi_prime / B_norm

    central_mass = (central_mass * G / radius) / v_norm**2
    sound_speed = sound_speed / v_norm
    ohm_diff = ohm_diff / diff_norm
    abi_diff = abi_diff / diff_norm
    hall_diff = hall_diff / diff_norm

    soln = solution(
        angles, init_con, B_power, sound_speed, central_mass, ohm_diff,
        abi_diff, hall_diff, max_steps=10000
    )

    cmap = plt.get_cmap("Dark2")

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_color_cycle(cmap(np.linspace(0,1, num=len(param_names))))
    ax.plot(angles, soln)
    ax.legend(param_names, loc=3)
    fig.savefig("plot.png")
    plt.show()

    #return angles, soln

if __name__ == '__main__':
    file_handler = logbook.FileHandler('ode.log', mode="w", level=logbook.DEBUG)
    stdout_handler = logbook.StreamHandler(sys.stdout, level=logbook.INFO)
    null_handler = logbook.NullHandler()
    with redirected_warnings(), redirected_logging():
        with null_handler.applicationbound(), file_handler.applicationbound(), stdout_handler.applicationbound():
            main()
