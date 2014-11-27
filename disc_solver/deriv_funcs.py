# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

from math import sin, sqrt
from .utils import cot


def B_unit_derivs(B_r, B_φ, B_θ, deriv_B_r, deriv_B_φ, deriv_B_θ):
    """
    Compute the derivatives of the unit vector of B.
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    norm_B_r, norm_B_φ, norm_B_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag
    return [
        1/B_mag * (
            B_deriv + B_unit/B_mag * (
                B_r * deriv_B_r + B_φ * deriv_B_φ +
                B_θ * deriv_B_θ
            )
        )

        for B_unit, B_deriv in (
            (norm_B_r, deriv_B_r), (norm_B_φ, deriv_B_φ),
            (norm_B_θ, deriv_B_θ)
        )
    ]


def dderiv_B_φ_soln(
    B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_θ,
    v_φ, deriv_v_r, deriv_v_θ, deriv_v_φ, deriv_B_r, deriv_B_θ,
    deriv_B_φ, β
):
    """
    Compute the derivative of B_φ
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    norm_B_r, norm_B_φ, norm_B_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag
    deriv_norm_B_r, deriv_norm_B_φ, deriv_norm_B_θ = B_unit_derivs(
        B_r, B_φ, B_θ, deriv_B_r, deriv_B_φ, deriv_B_θ
    )

    # Solution to \frac{∂}{∂θ}\frac{η_{H0}b_{θ} + η_{A0} b_{r}b_{φ}}{η_{O0} +
    #  η_{A0}\left(1-b_{φ}^{2}\right)}$
    magic_deriv = (
        (
            η_H * deriv_norm_B_θ * (
                η_O + η_A * (1 - norm_B_φ**2)
            ) + η_A * (
                norm_B_r * deriv_norm_B_φ +
                norm_B_φ * deriv_norm_B_r
            ) * (
                η_O + η_A * (1 - norm_B_φ**2)
            ) - (
                η_A * 2 * deriv_norm_B_φ * norm_B_φ
            ) * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            )
        ) / (η_O + η_A * (1 - norm_B_φ**2))**2)

    return (
        η_O + η_A * (1-norm_B_r**2) + (
            η_H**2 * norm_B_θ**2 -
            η_A**2 * norm_B_φ**2 * norm_B_r**2
        ) / (
            η_O + η_A * (1-norm_B_φ**2)
        )
    ) ** -1 * (
        deriv_B_r * (
            v_θ * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) - η_H * norm_B_r + η_A * norm_B_θ * norm_B_φ
        ) + B_r * (
            v_φ + deriv_v_θ * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) + v_θ * magic_deriv
        ) - B_θ * (
            deriv_v_φ * 2 / (2 * β - 1) + deriv_v_r * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) + v_r * magic_deriv + (1-β) * (
                η_H * norm_B_r -
                η_A * norm_B_θ * norm_B_φ
            )
        ) - deriv_B_θ * (
            v_r * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) + 2 / (2 * β - 1) * v_φ
        ) +
        deriv_B_φ * (
            η_H * norm_B_φ + η_A * norm_B_r * norm_B_θ +
            v_θ * 2 / (2 * β - 1) - η_O * cot(θ) -
            η_H * norm_B_φ * (1-β) - η_A * (
                cot(θ) * (1 - norm_B_r**2) -
                (1 - β) * norm_B_r * norm_B_θ
            ) - (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) * (
                η_H * (
                    norm_B_θ * cot(θ) - norm_B_r * (1 - β)
                ) + η_A * norm_B_φ * (
                    norm_B_θ * (1 - β) - norm_B_r * cot(θ)
                )
            )
        ) - B_φ * (
            η_O * (1 - β) -
            η_H * norm_B_φ * cot(θ) +
            η_A * (
                (1 - β) * (1 - norm_B_θ**2) -
                cot(θ) * norm_B_r * norm_B_θ
            ) + v_r - 2 / (2 * β - 1) * deriv_v_θ +
            η_O * sin(θ)**-2 -
            η_H * deriv_norm_B_φ * (1 - β) +
            η_A * (
                2 * cot(θ) * norm_B_r * deriv_norm_B_r -
                sin(θ)**-2 * norm_B_r**2 +
                (1 - β) * (
                    norm_B_r * deriv_norm_B_θ +
                    norm_B_θ * deriv_norm_B_r
                )
            ) - magic_deriv * (
                η_H * (
                    norm_B_θ * cot(θ) - norm_B_r * (1 - β)
                ) + η_A * norm_B_φ * (
                    norm_B_θ * (1 - β) - norm_B_r * cot(θ)
                )
            ) - (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) * (
                η_H * (
                    deriv_norm_B_θ * cot(θ) -
                    sin(θ)**-2 * norm_B_θ -
                    deriv_norm_B_r * (1-β)
                ) + η_A * (
                    deriv_norm_B_φ * (
                        norm_B_θ * (1 - β) - norm_B_r * cot(θ)
                    ) + norm_B_φ * (
                        deriv_norm_B_θ * (1 - β) -
                        deriv_norm_B_r * cot(θ) +
                        sin(θ)**-2 * norm_B_r
                    )
                )
            )
        )
    )
