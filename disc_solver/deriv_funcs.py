# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

from math import pi, sin, sqrt
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


def dderiv_v_r_midplane(
    ρ, B_θ, v_r, v_φ, deriv_v_θ, deriv_B_r, deriv_B_φ, dderiv_B_θ,
    β, η_O, η_H, η_A, c_s
):
    """
    Compute v_r'' around the midplane
    """
    A = (1 - 2 * β) * (η_O + η_A + (η_H**2) / (η_O + η_A)) * (
        pi * ρ * v_φ + (β-1) * B_θ * deriv_B_φ / v_r +
        B_θ**2 * η_H / (4 * (η_H**2 + (η_O + η_A)**2))) / (
        B_θ**2 + 4*pi * ρ * v_r * (β-1) * (2*β-1) *
            (η_O + η_A + (η_H**2) / (η_O + η_A)))

    C = (B_θ / (η_O + η_A)) * (
        2*pi * ρ * η_H * (β-1) * (
            2 * η_H * v_r * (2*β-1) / (η_O + η_A) + v_φ / (β-1) +
            B_θ * deriv_B_φ / (v_r * ρ * pi)
        ) / (
            B_θ**2 + 4*pi * ρ * v_r * (β-1) * (2*β-1) *
            (η_O + η_A + η_H**2/(η_O + η_A))
        ) - 1
    )

    D = 1 + 4 * v_r * (β-1) * deriv_v_θ / c_s**2 + (
        (β-1) * B_θ * deriv_B_r + deriv_B_r**2 + deriv_B_φ**2) / (
            2*pi * ρ * c_s**2)

    A_dash = (1 + B_θ**2 / (4*pi * ρ * v_r * (β-1) * (2*β-1) * (
        η_O + η_A + η_H**2/(η_O + η_A))))**-1 * (
            deriv_B_φ / (2*pi * ρ) * (
                dderiv_B_θ - 2*(β-1) * deriv_B_r + B_θ -
                B_θ / c_s**2 * (
                    4 * v_r * (β-1) * deriv_v_θ + (
                        (β-1) * B_θ * deriv_B_r + deriv_B_r**2 + deriv_B_φ**2
                    ) / (2*pi * ρ)
                ) + B_θ / (2*pi * ρ * (η_O + η_A + η_H**2/(η_O + η_A))) * (
                    - 2 * dderiv_B_θ * (
                        η_H * v_r / (η_O + η_A) + 2 * v_φ / (2*β-1)
                    ) + deriv_B_r * (
                        v_φ + 2 * deriv_v_θ * η_H / (η_O + η_A) +
                        (β-1) * η_H + B_θ**-1 * (
                            deriv_B_φ * η_A * (3 - β - v_r / (η_O + η_A)) +
                            deriv_B_r * η_H * (v_r / (η_O + η_A) - 1)
                        ) + deriv_B_φ / (B_θ * (η_O + η_A))
                        * (v_r * η_A - η_H**2)
                    ) + deriv_B_φ * (
                        η_O - β * η_A + η_H**2 / (η_O + η_A) +
                        2 * deriv_v_θ / (2*β - 1) +
                        β * η_H * deriv_B_φ / B_θ +
                        η_H / (B_θ * (η_O + η_A)) * (
                            (β-1) * η_A + deriv_B_φ * v_r * (
                                1 + 2 * η_A / (η_O + η_A)))))))

    C_dash = dderiv_B_θ * (β-1) + (η_O + η_A)**-1 * (
        2 * deriv_B_r * deriv_v_θ -
        v_r * dderiv_B_θ + 2 * η_A * deriv_B_φ**2 / B_θ * (
            β - 1 + deriv_B_r / B_θ - v_r / (η_O + η_A)
        ) + η_H * deriv_B_φ * (
            2 * (1 - deriv_B_r * (β-1) / B_θ) + (
                deriv_B_r**2 + 2 * deriv_B_φ**2 * η_O / (η_O + η_A)
            ) / B_θ**2
        ) + 4*pi * ρ * v_r * η_H * (β-1) * (2*β-1) * (
            η_O + η_A + η_H**2 / (η_O + η_A)
        ) / (B_θ**2 + 4*pi * ρ * v_r * (β-1) * (2*β-1) * (
            η_O + η_A + η_H**2 / (η_O + η_A))
        ) * (
            2 * dderiv_B_θ * (η_H * v_r / (η_O + η_A) + 2 * v_φ / (2*β-1)) -
            deriv_B_r * (
                v_φ + 2 * deriv_v_θ * η_H / (η_O + η_A) + (β-1) * η_H +
                deriv_B_φ * (v_r * η_A - η_H**2) / (B_θ * (η_O + η_A)) +
                B_θ**-1 * (
                    deriv_B_φ * η_A * (3 - β - v_r / (η_O + η_A)) +
                    deriv_B_r * η_H * (v_r / (η_O + η_A) - 1)
                ) + B_θ * deriv_B_φ / (2*pi * ρ * (2*β-1) * v_r) +
                B_θ**2 * deriv_B_φ * (deriv_B_r + (β-1) * B_θ) / (
                    4 * pi**2 * ρ**2 * v_r * (β-1) * c_s**2 * (2*β-1)
                )
            ) - deriv_B_φ * (
                η_O - β * η_A + η_H**2 / (η_O + η_A) +
                2 * deriv_v_θ / (2*β-1) + β * η_H * deriv_B_φ / B_θ +
                η_H * (
                    η_A * (β-1) + deriv_B_φ * v_r * (1 - 2 * η_A / (η_O + η_A))
                ) / (B_θ * (η_O + η_A)) - B_θ * (
                    dderiv_B_θ + B_θ - B_θ / (c_s**2) * (
                        4 * v_r * (β-1) * deriv_v_θ + deriv_B_φ**2 / (2*pi * ρ)
                    )
                ) / 4*pi * ρ * v_r * (β-1) * (2*β-1)
            )
        )
    )

    return (
        4 * v_φ * A_dash + (
            dderiv_B_θ * deriv_B_r + B_θ * C_dash +
            2 * (β-1) * (B_θ * dderiv_B_θ + deriv_B_φ**2) +
            B_θ * D * (deriv_B_r + (β-1) * B_θ)
        ) / (2*pi * ρ)
    ) / (
        v_r * (4*β - 7) - 4 * v_φ * A -
        B_θ * (C + 2 * (β-1) * (deriv_B_φ + (β-1) * B_θ))
    )


def dderiv_ρ_midplane(
    ρ, B_θ, v_r, deriv_v_θ, deriv_B_r, deriv_B_φ, dderiv_v_r, β, c_s
):
    """
    Compute ρ'' around the midplane
    """
    return - ρ * (
        1 + 2 * (β-1) * dderiv_v_r / v_r +
        4 * v_r * deriv_v_θ * (β-1) / (c_s**2) + (
            (β-1) * B_θ * deriv_B_r + deriv_B_r**2 + deriv_B_φ**2
        ) / (2*pi * ρ * c_s**2)
    )


def dderiv_v_φ_midplane(
    ρ, B_θ, v_r, v_φ, deriv_v_θ, deriv_B_r, deriv_B_φ, dderiv_B_θ, dderiv_v_r,
    β, η_O, η_H, η_A, c_s
):
    """
    Compute v_φ'' around the midplane
    """
    return (
        1 + B_θ**2 / (4*pi * ρ * v_r * (β-1) * (2*β-1) * (
            η_O + η_A + η_H**2 / (η_O + η_A)
        ))
    )**-1 * (
        deriv_B_φ / (2*pi * ρ) * (
            dderiv_B_θ - 2 * (β-1) * deriv_B_r + B_θ -
            B_θ / (c_s**2) * (4 * v_r * (β-1) * deriv_v_θ + (
                (β-1) * B_θ * deriv_B_r + deriv_B_r**2 + deriv_B_φ**2
            ) / (2*pi * ρ))
        ) - dderiv_v_r / (4 * v_r * (β-1)) * (
            v_φ + (β-1) * B_θ * deriv_B_φ / (v_r * pi * ρ) + B_θ**2 * η_H / (
                4*pi * ρ * (η_H**2 + (η_O + η_A)**2)
            )
        ) + B_θ / (2*pi * ρ * (η_O + η_A + η_H**2 / (η_O + η_A))) +
        deriv_B_r * (
            v_φ + (β-1) * η_H + 2 * deriv_v_θ * η_H / (η_O + η_A) + (
                deriv_B_φ * η_A * (3 - β - v_r / (η_O + η_A)) +
                deriv_B_r * η_H * (v_r / (η_O + η_A) - 1)
            ) / B_θ
        ) - 2 * dderiv_B_θ * (
            η_H * v_r / (η_O + η_A) + 2 * v_φ / (2*β-1)
        ) + deriv_B_φ * (
            η_O - β * η_A + η_H**2 / (η_O + η_A) +
            2 * deriv_v_θ / (2*β-1) + β * η_H * deriv_B_r / B_θ +
            η_H / (B_θ * (η_O + η_A)) * (
                (β-1) * η_A + deriv_B_φ * v_r * (1 + 2 * η_A / (η_O + η_A))
            )
        )
    )
