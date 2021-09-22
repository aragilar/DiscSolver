# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT

from numpy import tan as np_tan, sqrt as np_sqrt

from ..utils import sec as np_sec

from .j_e_funcs import E_θ_func


def Z_5_func(*, η_O, η_A, η_H, b_r, b_θ, b_φ, C):
    """
    Compute the value of the variable Z_5
    """
    return η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)


def deriv_B_φ_func(
    *, θ, γ, B_r, B_θ, B_φ, v_r, v_φ, v_θ, η_O, η_A, η_H, E_r, b_r, b_φ, b_θ,
    C, Z_5, tan=np_tan,
):
    """
    Compute the derivative of B_φ, assuming E_r is being used
    """
    return (
        C * (v_θ * B_r - v_r * B_θ) - E_r - v_φ * B_θ + B_φ * (
            v_θ + tan(θ) * (η_O + η_A * (1 - b_r ** 2)) + (1 / 4 - γ) * (
                η_H * b_φ + η_A * b_r * b_θ
            ) - C * (
                η_A * b_φ * (
                    b_r * tan(θ) -
                    b_θ * (1/4 - γ)
                ) + η_H * (
                    b_r * (1/4 - γ) +
                    b_θ * tan(θ)
                )
            )
        )
    ) / Z_5


def deriv_B_r_func(
    *, θ, γ, B_r, B_θ, B_φ, v_r, v_θ, deriv_B_φ, η_O, η_A, η_H, b_r, b_φ, b_θ,
    tan=np_tan,
):
    """
    Compute the derivative of B_r
    """
    return (
        (
            v_θ * B_r - v_r * B_θ - deriv_B_φ * (
                η_H * b_θ +
                η_A * b_r * b_φ
            ) + B_φ * (
                η_A * b_φ * (
                    b_r * tan(θ) -
                    b_θ * (1/4 - γ)
                ) + η_H * (
                    b_r * (1/4 - γ) +
                    b_θ * tan(θ)
                )
            )
        ) / (
            η_O + η_A * (1 - b_φ) * (1 + b_φ)
        ) - B_θ * (1/4 - γ)
    )


def deriv_E_r_func(
    *, γ, v_r, v_φ, B_r, B_φ, η_O, η_A, η_H, b_r, b_θ, b_φ, J_r, J_θ, J_φ,
):
    """
    Compute the derivative of E_r
    """
    return (γ - 3 / 4) * E_θ_func(
        v_r=v_r, v_φ=v_φ, B_r=B_r, B_φ=B_φ, J_r=J_r, J_θ=J_θ, J_φ=J_φ, η_O=η_O,
        η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ
    )


def deriv_η_skw_func(
    *, ρ, B_r, B_φ, B_θ, deriv_ρ, deriv_B_r, deriv_B_φ, deriv_B_θ
):
    """
    Compute the scaled part of the derivative of η assuming the same form as in
    SKW
    """
    B_sq = B_r ** 2 + B_φ ** 2 + B_θ ** 2
    return (
        2 * (B_r * deriv_B_r + B_φ * deriv_B_φ + B_θ * deriv_B_θ) -
        B_sq * deriv_ρ / ρ
    ) / ρ


def B_unit_derivs(
    *, B_r, B_φ, B_θ, deriv_B_r, deriv_B_φ, deriv_B_θ, b_r, b_φ, b_θ,
    sqrt=np_sqrt,
):
    """
    Compute the derivatives of the unit vector of B.
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    return [
        1/B_mag * (
            B_deriv - B_unit/B_mag * (
                B_r * deriv_B_r + B_φ * deriv_B_φ +
                B_θ * deriv_B_θ
            )
        )

        for B_unit, B_deriv in (
            (b_r, deriv_B_r), (b_φ, deriv_B_φ),
            (b_θ, deriv_B_θ)
        )
    ]


def A_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O, deriv_η_A, deriv_η_H,
    deriv_b_θ, deriv_b_r, deriv_b_φ
):
    """
    Compute the value of the variable A
    """
    return (
        deriv_η_H * b_θ + η_H * deriv_b_θ - deriv_η_A * b_r * b_φ -
        η_A * deriv_b_r * b_φ - η_A * b_r * deriv_b_φ
    ) / (
        η_O + η_A * (1 - b_φ ** 2)
    ) - (
        (
            deriv_η_O + deriv_η_A * (1 - b_φ ** 2) - 2 * η_A * b_φ * deriv_b_φ
        ) * (η_H * b_θ - η_A * b_r * b_φ)
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    )


def C_func(*, η_O, η_A, η_H, b_θ, b_r, b_φ):
    """
    Compute the value of the variable C
    """
    return (η_H * b_θ - η_A * b_r * b_φ) / (η_O + η_A * (1 - b_φ ** 2))


def dderiv_B_φ_soln(
    *, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_θ,
    v_φ, deriv_v_r, deriv_v_θ, deriv_v_φ, deriv_B_r, deriv_B_θ,
    deriv_B_φ, γ, deriv_η_O, deriv_η_A, deriv_η_H, b_r, b_φ, b_θ, deriv_b_r,
    deriv_b_φ, deriv_b_θ, C, A, tan=np_tan, sec=np_sec,
):
    """
    Compute the second derivative of B_φ (when E_r is not used)
    """
    return (
        deriv_v_θ * B_φ + v_θ * deriv_B_φ - deriv_v_φ * B_θ - v_φ * deriv_B_θ +
        A * (
            v_θ * B_r - v_r * B_θ + B_φ * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            )
        ) + C * (
            deriv_v_θ * B_r - deriv_v_r * B_θ + v_θ * deriv_B_r -
            v_r * deriv_B_θ + deriv_B_φ * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                )
            ) + B_φ * (
                deriv_η_A * b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_A * deriv_b_φ * (
                    b_r * tan(θ) - b_θ * (1 / 4 - γ)
                ) + η_A * b_φ * (
                    deriv_b_r * tan(θ) - deriv_b_θ * (1 / 4 - γ) +
                    b_r * (1 + tan(θ) ** 2)
                ) + deriv_η_H * (
                    b_r * (1 / 4 - γ) + b_θ * tan(θ)
                ) + η_H * (
                    deriv_b_r * (1 / 4 - γ) + deriv_b_θ * tan(θ) +
                    b_θ * (1 + tan(θ) ** 2)
                )
            )
        ) - deriv_B_φ * (
            deriv_η_O + deriv_η_A * (1 - b_r ** 2) -
            2 * η_A * b_r * deriv_b_r + A * (
                η_H * b_θ + η_A * b_r * b_φ
            ) + C * (
                deriv_η_H * b_θ + η_H * deriv_b_θ + deriv_η_A * b_r * b_φ +
                η_A * deriv_b_r * b_φ + η_A * b_r * deriv_b_φ
            ) + η_O * tan(θ) + η_A * (
                (1 - b_r ** 2) * tan(θ) + b_r * b_θ * (1 / 4 - γ)
            ) + (1 / 4 - γ) * η_H * b_φ
        ) - B_φ * (
            deriv_η_O * tan(θ) + η_O * sec(θ) ** 2 + deriv_η_A * (
                (1 - b_r ** 2) * tan(θ) + b_r * b_θ * (1 / 4 - γ)
            ) + (1 / 4 - γ) * deriv_η_H * b_φ - η_A * (
                2 * b_r * deriv_b_r * tan(θ) - (1 - b_r ** 2) * sec(θ) ** 2 -
                deriv_b_r * b_θ * (1 / 4 - γ) - b_r * deriv_b_θ * (1 / 4 - γ)
            ) + (1 / 4 - γ) * η_H * deriv_b_φ
        ) - (3 / 4 - γ) * (
            (
                deriv_B_r + B_θ * (1 / 4 - γ)
            ) * (
                η_H * b_r + η_A * b_θ * b_φ
            ) + deriv_B_φ * (
                η_H * b_φ - η_A * b_r * b_θ
            ) + B_φ * (
                η_O * (1 / 4 - γ) + η_A * (
                    (1 / 4 - γ) * (1 - b_θ ** 2) + tan(θ) * b_r * b_θ
                ) - η_H * b_φ * tan(θ)
            ) + v_r * B_φ - v_φ * B_r
        )
    ) / (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ + η_A * b_r * b_φ)
    )
