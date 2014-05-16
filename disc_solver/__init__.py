# -*- coding: utf-8 -*-
"""
"""

from math import pi, cos, sin, sqrt

import numpy as np
from scikits.odes import ode

from ode_wrapper import validate_cvode_flags

INTEGRATOR = "cvode"

G = 1 # FIX

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

def dderiv_B_phi(
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
                ohm_diff * cos(angle)**-2 -
                hall_diff * deriv_norm_B_phi * (1 - B_power) +
                abi_diff * (
                    2 * cot(angle) * norm_B_r * deriv_norm_B_r -
                    cos(angle)**-2 * norm_B_r **2 +
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
                        cos(angle) **-2 * norm_B_theta -
                        deriv_norm_B_r * (1-B_power)
                    ) + abi_diff * (
                        deriv_norm_B_phi * (
                            norm_B_theta * (1 - B_power) - norm_B_r * cot(angle)
                        ) + norm_B_phi * (
                            deriv_norm_B_theta * (1 - B_power) -
                            deriv_norm_B_r * cot(angle) +
                            cos(angle)**-2 * norm_B_r
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

        derivs[1] = B_phi_prime
        derivs[0] = - (
                B_theta * (1 - B_power) +
                (
                    v_r * B_theta - v_theta * B_r + derivs[1] * (
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
        derivs[2] = (B_power - 2) * B_r - B_theta * cot(angle)

        derivs[3] = (
                v_r**2 / (2 * v_theta) +
                v_theta +
                v_phi**2 / v_theta +
                (2 * B_power * sound_speed**2) / v_theta + 
                (G * central_mass) / v_theta +
                (
                    B_theta * derivs[0] + (B_power - 1)*(B_theta**2 + B_phi**2)
                ) / (
                    4 * pi * v_theta * rho
                )
            )

        derivs[4] = (
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

        derivs[5] = v_theta / (sound_speed**2 - v_theta**2) * (
                (v_r * v_theta) / 2 -
                cot(angle) * v_phi **2 -
                sound_speed**2 * (
                    (5/2 - 2*B_power)* v_r / v_theta + cot(angle)
                ) + (
                    (B_power - 1) *B_theta * B_r +
                    B_r * derivs[0] +
                    B_phi * B_phi_prime +
                    B_phi ** 2 * cot(angle)
                ) / (
                    4 * pi * rho
                )
            )

        derivs[6] = - rho * (
                (
                    (5/2 - 2 * B_power) * v_r + derivs[5]
                ) / v_theta + cot(angle)
            )

        derivs[7] = dderiv_B_phi(
                B_r, B_phi, B_theta, ohm_diff, hall_diff, abi_diff, angle, v_r,
                v_theta, v_phi, derivs[3], derivs[5], derivs[4], derivs[0], derivs[2],
                derivs[1], B_power
            )

        return 0
    return rhs_equation

def solution(
        angles, initial_conditions, B_power, sound_speed, central_mass,
        ohm_diff, abi_diff, hall_diff, relative_tolerance=1e-10,
        absolute_tolerance=1e-14
):
    solver = ode(INTEGRATOR,
            ode_system(
                B_power, sound_speed, central_mass, ohm_diff, abi_diff,
                hall_diff
            ),
            rtol=relative_tolerance, atol=absolute_tolerance
    )
    return validate_cvode_flags(*solver.solve(
            angles, initial_conditions
        ))

