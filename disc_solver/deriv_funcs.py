# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

from math import sin, sqrt
from .utils import cot


def B_unit_derivs(B_r, B_phi, B_theta, deriv_B_r, deriv_B_phi, deriv_B_theta):
    """
    Compute the derivatives of the unit vector of B.
    """
    B_mag = sqrt(B_r**2 + B_phi**2 + B_theta**2)
    norm_B_r, norm_B_phi, norm_B_theta = B_r/B_mag, B_phi/B_mag, B_theta/B_mag
    return [
        1/B_mag * (
            B_deriv + B_unit/B_mag * (
                B_r * deriv_B_r + B_phi * deriv_B_phi +
                B_theta * deriv_B_theta
            )
        )

        for B_unit, B_deriv in (
            (norm_B_r, deriv_B_r), (norm_B_phi, deriv_B_phi),
            (norm_B_theta, deriv_B_theta)
        )
    ]


def dderiv_B_phi_soln(
    B_r, B_phi, B_theta, ohm_diff, hall_diff, abi_diff, angle, v_r, v_theta,
    v_phi, deriv_v_r, deriv_v_theta, deriv_v_phi, deriv_B_r, deriv_B_theta,
    deriv_B_phi, B_power
):
    """
    Compute the derivative of B_phi
    """
    B_mag = sqrt(B_r**2 + B_phi**2 + B_theta**2)
    norm_B_r, norm_B_phi, norm_B_theta = B_r/B_mag, B_phi/B_mag, B_theta/B_mag
    deriv_norm_B_r, deriv_norm_B_phi, deriv_norm_B_theta = B_unit_derivs(
        B_r, B_phi, B_theta, deriv_B_r, deriv_B_phi, deriv_B_theta
    )

    # Solution to \frac{∂}{∂θ}\frac{η_{H0}b_{θ} + η_{A0} b_{r}b_{φ}}{η_{O0} +
    #  η_{A0}\left(1-b_{φ}^{2}\right)}$
    magic_deriv = (
        (
            hall_diff * deriv_norm_B_theta * (
                ohm_diff + abi_diff * (1 - norm_B_phi**2)
            ) + abi_diff * (
                norm_B_r * deriv_norm_B_phi +
                norm_B_phi * deriv_norm_B_r
            ) * (
                ohm_diff + abi_diff * (1 - norm_B_phi**2)
            ) - (
                abi_diff * 2 * deriv_norm_B_phi * norm_B_phi
            ) * (
                hall_diff * norm_B_theta + abi_diff * norm_B_r * norm_B_phi
            )
        ) / (ohm_diff + abi_diff * (1 - norm_B_phi**2))**2)

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
                cot(angle) * (1 - norm_B_r**2) -
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
                (1 - B_power) * (1 - norm_B_theta**2) -
                cot(angle) * norm_B_r * norm_B_theta
            ) + v_r - 2 / (2 * B_power - 1) * deriv_v_theta +
            ohm_diff * sin(angle)**-2 -
            hall_diff * deriv_norm_B_phi * (1 - B_power) +
            abi_diff * (
                2 * cot(angle) * norm_B_r * deriv_norm_B_r -
                sin(angle)**-2 * norm_B_r**2 +
                (1 - B_power) * (
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
                    sin(angle)**-2 * norm_B_theta -
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
