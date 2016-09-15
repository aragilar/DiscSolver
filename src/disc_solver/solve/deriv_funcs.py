# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

from math import sqrt, tan

import logbook

from ..utils import sec, ODEIndex

log = logbook.Logger(__name__)


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
    deriv_B_φ, β, deriv_η_O, deriv_η_A, deriv_η_H,
):
    """
    Compute the derivative of B_φ
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag
    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r, B_φ, B_θ, deriv_B_r, deriv_B_φ, deriv_B_θ
    )

    C = (η_H * b_θ + η_A * b_r * b_φ) / (η_O + η_A * (1 - b_φ ** 2))

    A = (
        (
            deriv_η_O + deriv_η_A * (1 - b_φ ** 2) - 2 * η_A * b_φ * deriv_b_φ
        ) * (η_H * b_θ + η_A * b_r * b_φ)
    )/(
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    ) - (
        deriv_η_H * b_θ + η_H * deriv_b_θ + deriv_η_A * b_r * b_φ +
        η_A * deriv_b_r * b_φ + η_A * b_r * deriv_b_φ
    ) / (
        η_O + η_A * (1 - b_φ ** 2)
    )

    return (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
    ) ** -1 * (
        v_φ * B_r - 2 / (2 * β - 1) * (
            deriv_v_φ * B_θ + v_φ * deriv_B_θ
        ) - (B_θ * (1-β) - deriv_B_r) * (
            η_H * b_r - η_A * b_θ * b_φ
        ) + deriv_B_φ * (
            v_θ * 2 / (2 * β - 1) + η_O * tan(θ) - deriv_η_O -
            deriv_η_H * C * b_θ - deriv_η_A * (
                1 - b_r ** 2 - C * b_r * b_φ
            ) + η_H * (
                b_φ * (2 - β) - C * (
                    b_r * (1 - β) - b_θ * tan(θ) - deriv_b_θ
                ) - A * b_θ
            ) + η_A * (
                tan(θ) * (1 - b_r ** 2) + (β - 1) * b_r * b_φ + C * b_φ * (
                    b_θ * (1 - β) - b_r * tan(θ)
                ) + 2 * b_r * deriv_b_r - A * b_r * b_φ -
                C * deriv_b_r * b_φ - C * b_r * deriv_b_φ + b_r * b_θ
            )
        ) - A * v_θ * B_r - C * deriv_v_θ * B_r - C * v_θ * deriv_B_r +
        A * v_r * B_θ + C * deriv_v_r * B_θ + C * v_r * deriv_B_θ + B_φ * (
            deriv_η_O * tan(θ) + η_O * (sec(θ) ** 2 + (1 - β)) - v_r +
            2 / (2 * β - 1) * deriv_v_θ + η_H * (
                deriv_b_φ * (1 - β) - A * (
                    b_r * (1 - β) - b_θ * tan(θ)
                ) - C * b_φ * (
                    deriv_b_r * (1-β) - deriv_b_θ * tan(θ) -
                    b_θ * sec(θ) ** 2
                ) - b_φ * tan(θ)
            ) + deriv_η_H * (
                b_φ * (1 - β) - C * (b_r * (1 - β) - b_θ * tan(θ))
            ) + η_A * (
                sec(θ) ** 2 * (1 - b_r ** 2) + 2 * tan(θ) * b_r * deriv_b_r +
                (β - 1) * deriv_b_r * b_θ + (β - 1) * b_r * deriv_b_θ +
                A * b_φ * (b_θ * (1 - β) - b_r * tan(θ)) +
                C * deriv_b_φ * (b_θ * (1 - β) - b_r * tan(θ)) + C * b_φ * (
                    deriv_b_θ * (1 - β) - deriv_b_r * tan(θ) -
                    b_r * sec(θ) ** 2
                ) + (1 - β) * (1 - b_θ ** 2) - tan(θ) * b_r * b_θ
            ) + deriv_η_A * (
                tan(θ) * (1 - b_r ** 2) - (1 - β) * b_r * b_θ + C * b_φ * (
                    b_θ * (1 - β) - b_r * tan(θ)
                )
            )
        )
    )


def Y_1_func(v_r, v_φ, deriv_B_r, deriv_B_φ, β, a_0):
    """
    Compute Y_1
    """
    return v_r ** 2 * (β - 1) * (4 * β - 5) / 2 + v_φ ** 2 + a_0 * (
        deriv_B_r * (β - 1 + deriv_B_r) + deriv_B_φ ** 2
    )


def Y_2_func(
    v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H, η_P,
    η_perp_sq, dderiv_η_P, dderiv_η_H,
):
    """
    Compute Y_2
    """
    return (
        deriv_B_φ * (
            v_r * η_P * (6 * β - 11) / (2 * β - 1) - (β - 2) * η_P ** 2 +
            η_perp_sq * (1 - dderiv_η_P / η_P)
        ) - v_r * (
            dderiv_η_H - dderiv_η_P * η_H / η_P
        ) + deriv_B_r * deriv_B_φ * (
            (β - 1 + 2 * deriv_B_r) * η_A * η_P + η_H ** 2 * (
                β - 1 - 2 * deriv_B_r
            ) - 2 * η_A * v_r
        ) - η_H * deriv_B_φ ** 2 / η_P * (
            (4 * β - 5) * η_P ** 2 + η_O * (
                2 * deriv_B_φ * η_H - v_r - 2 * (β - 1) * η_P
            )
        ) + deriv_B_r * (
            (β - 1) * η_H * η_P + v_φ * η_P - v_r * (4 * β - 5) * η_H
        ) + η_H * deriv_B_r ** 2 * (
            η_P + v_r
        ) + dderiv_B_θ * (
            v_r * η_H - 2 * η_P * v_φ / (2 * β - 1)
        )
    ) / η_perp_sq


def Y_3_func(
    v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_P, η_A, η_H, dderiv_η_P,
    dderiv_η_H,
):
    """
    Compute Y_3
    """
    return - (
        v_r * (
            dderiv_B_θ - (4 * β - 5) * deriv_B_r - dderiv_η_P / η_P
        ) - dderiv_η_H * deriv_B_φ + η_H * deriv_B_φ * (
            2 + dderiv_η_P / η_P + deriv_B_r * (
                2 * (β-1) + deriv_B_r
            ) + deriv_B_φ ** 2 * (
                1 - 2 * η_A / η_P
            )
        ) + 2 * η_A * deriv_B_φ ** 2 * (
            1 - β + deriv_B_r + v_r / η_P
        )
    ) / η_P - dderiv_B_θ * (β - 1)


def Y_4_func(a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, β, Y_2, Y_1):
    """
    Compute Y_4
    """
    return 2 * a_0 * (
        Y_2 - deriv_B_φ * (deriv_B_r * β - Y_1 + 1)
    ) + 2 * v_r * v_φ * (4 * β - 5)


def Y_5_func(a_0, v_r, β, η_perp_sq):
    """
    Compute Y_5
    """
    return v_r * (8 * β - 9) + 4 * a_0 / (η_perp_sq * (2 * β - 1))


def Y_6_func(a_0, v_r, v_φ, β, η_perp_sq, η_P, η_H, Y_5):
    """
    Compute Y_6
    """
    return (2 * β - 3) * v_r + v_φ ** 2 / Y_5 - a_0 / η_perp_sq * (
        v_φ * η_H * (4 * β - 1) / ((2 * β - 1) * Y_5) - 1 / (2 * η_P) * (
            η_perp_sq - η_H * η_P + 4 * η_H ** 2 * η_P * a_0 / (
                (2 * β - 1) * Y_5 * η_perp_sq
            )
        )
    )


def dderiv_v_r_midplane(
    a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_perp_sq, η_H, η_P,
    Y_6, Y_5, Y_4, Y_3, Y_2, Y_1
):
    """
    Compute v_r'' around the midplane
    """
    return (
        v_φ * Y_4 / Y_5 + ((2 * β - 5 / 2) * v_r) ** 2 + a_0 / 2 * (
            dderiv_B_θ * (deriv_B_r + 2 * (β - 1)) +
            2 * deriv_B_φ ** 2 * (β - 1) + Y_3 + Y_2 * η_H / η_P -
            2 * η_H * Y_4 / ((2 * β - 1) * η_perp_sq * Y_5) +
            Y_1 * (deriv_B_r + β - 1)
        )
    ) / Y_6


def dderiv_v_φ_midplane(a_0, v_φ, dderiv_v_r, η_perp_sq, η_H, Y_5, Y_4):
    """
    Compute v_φ'' around the midplane
    """
    return dderiv_v_r / Y_5 * (a_0 * η_H / η_perp_sq - v_φ) + Y_4 / Y_5


def taylor_series(β, a_0, init_con, η_derivs):
    """
    Compute taylor series of v_r'', ρ'' and v_φ''
    """
    v_r = init_con[ODEIndex.v_r]
    v_φ = init_con[ODEIndex.v_φ]
    deriv_B_φ = init_con[ODEIndex.B_φ_prime]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]

    η_P = η_O + η_A
    η_perp_sq = η_P ** 2 + η_H ** 2

    deriv_B_r = - (
        1 - β + (v_r + deriv_B_φ * η_H) / η_P
    )

    dderiv_B_θ = (β - 2) * deriv_B_r

    Y_1 = Y_1_func(v_r, v_φ, deriv_B_r, deriv_B_φ, β, a_0)

    dderiv_ρ = - Y_1

    if η_derivs:
        dderiv_η_scale = dderiv_ρ / 2
    else:
        dderiv_η_scale = 0

    dderiv_η_O = dderiv_η_scale * η_O
    dderiv_η_A = dderiv_η_scale * η_A
    dderiv_η_H = dderiv_η_scale * η_H
    dderiv_η_P = dderiv_η_O + dderiv_η_A

    Y_2 = Y_2_func(
        v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H, η_P,
        η_perp_sq, dderiv_η_P, dderiv_η_H,
    )
    Y_3 = Y_3_func(
        v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_P, η_A, η_H, dderiv_η_P,
        dderiv_η_H
    )
    Y_4 = Y_4_func(a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, β, Y_2, Y_1)
    Y_5 = Y_5_func(a_0, v_r, β, η_perp_sq)
    Y_6 = Y_6_func(a_0, v_r, v_φ, β, η_perp_sq, η_P, η_H, Y_5)

    dderiv_v_r = dderiv_v_r_midplane(
        a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_perp_sq, η_H,
        η_P, Y_6, Y_5, Y_4, Y_3, Y_2, Y_1
    )

    dderiv_v_φ = dderiv_v_φ_midplane(
        a_0, v_φ, dderiv_v_r, η_perp_sq, η_H, Y_5, Y_4
    )

    log.info("Y_1: {}".format(Y_1))
    log.info("Y_2: {}".format(Y_2))
    log.info("Y_3: {}".format(Y_3))
    log.info("Y_4: {}".format(Y_4))
    log.info("Y_5: {}".format(Y_5))
    log.info("Y_6: {}".format(Y_6))
    log.info("B_θ'': {}".format(dderiv_B_θ))
    log.info("v_r'': {}".format(dderiv_v_r))
    log.info("v_φ'': {}".format(dderiv_v_φ))
    log.info("ρ'': {}".format(dderiv_ρ))

    return dderiv_ρ, dderiv_v_r, dderiv_v_φ
