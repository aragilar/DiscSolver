# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

from math import pi, sqrt, tan

import logbook

from ..utils import sec

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


def dderiv_v_r_midplane(
    ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ,
    β, η_O, η_H, η_A, Y6, Y5, Y4, Y3, Y2, Y1
):
    """
    Compute v_r'' around the midplane
    """
    return Y6 / (v_r * (4 * β - 6)) * (
        2 * v_φ * Y5 * Y4 + (
            v_r**2 * (4 * β - 5)**2
        ) / 2 + 1 / (4 * pi * ρ) * (
            dderiv_B_θ * deriv_B_r +
            B_θ * Y3 + (
                Y2 * η_H * B_θ
            ) / (
                η_O + η_A
            ) - (
                2 * η_H * B_θ**2 * Y5 * Y4
            ) / (
                (2 * β - 1) * (
                    (η_O + η_A)**2 + η_H**2
                )
            ) + 2 * (β - 1) * (
                B_θ * dderiv_B_θ + deriv_B_φ**2
            ) - Y1 * B_θ * (
                deriv_B_r + (β - 1) * B_θ
            )
        )
    )


def Y1_func(
    ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, β, c_s
):
    """
    Compute Y1
    """
    return - (
        v_r**2 * (β-1) * (4*β-5) / 2 +
        v_φ**2 + 1 / (4 * pi * ρ) * (
            (β-1) * B_θ * deriv_B_r + deriv_B_r**2 + deriv_B_φ**2
        )
    ) / (c_s**2)


def Y2_func(
    B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H,
    dderiv_η_O, dderiv_η_A, dderiv_η_H,
):
    """
    Compute Y2
    """
    return 1 / (η_O + η_A + (η_H ** 2) / (η_A + η_O)) * (
        deriv_B_φ * (
            v_r * (6 * β - 11) / (2 * β - 1) -
            (β - 3) * (η_O + η_A) +
            (η_H ** 2) / (η_A + η_O)
        ) - v_r * dderiv_η_H * B_θ / (η_A + η_O) +
        deriv_B_r * deriv_B_φ / (B_θ) * (
            η_A * (β - 1 + 2 * deriv_B_r / B_θ) +
            1 / (η_A + η_O) * (
                η_H ** 2 * (β - 1 - 2 * deriv_B_r / B_θ) -
                2 * η_A * v_r
            )
        ) - η_H * deriv_B_φ ** 2 / B_θ * (
            (2 * β - 3) +
            1 / (η_O + η_A) * (
                2 * (β - 1) * η_A + (
                    2 * deriv_B_φ * η_H / B_θ -
                    v_r
                ) * (
                    1 - η_A / (η_A + η_O)
                )
            )
        ) - (dderiv_η_A + dderiv_η_O) * (
            deriv_B_φ +
            η_H ** 2 * deriv_B_φ / ((η_A + η_O) ** 2) -
            η_H * v_r * B_θ / (η_A + η_O)
        ) + deriv_B_r * (
            (β - 1) * η_H +
            v_φ -
            v_r * (4 * β - 5) * η_H / (η_A + η_O)
        ) + η_H * deriv_B_r ** 2 / B_θ * (
            1 + v_r / (η_A + η_O)
        ) + dderiv_B_θ * (
            v_r * η_H / (η_A + η_O) -
            2 * v_φ / (2 * β - 1)
        )
    )


def Y3_func(
    B_θ, v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H, dderiv_η_O,
    dderiv_η_A,
):
    """
    Compute Y3
    """
    return - 1 / (η_A + η_O) * (
        v_r * (
            dderiv_B_θ - (4 * β - 5) * deriv_B_r - B_θ * (
                dderiv_η_O + dderiv_η_A
            )
        ) + (
            2 * η_A * deriv_B_φ**2
        ) / B_θ * (
            1 - β +
            deriv_B_r / B_θ +
            v_r / (η_O + η_A)
        ) + η_H * deriv_B_φ * (
            2 + dderiv_η_O + dderiv_η_A + deriv_B_r / B_θ * (
                2 * (β - 1) +
                deriv_B_r / B_θ
            ) + (
                deriv_B_φ**2
            ) / (
                B_θ**2
            ) * (
                1 - (2 * η_A) / (η_O + η_A)
            )
        )
    ) - B_θ * (β - 1)


def Y4_func(ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, Y2, Y1):
    """
    Compute Y4
    """
    return (
        1 / (2 * pi * ρ) * (
            Y2 * B_θ + deriv_B_φ * (
                dderiv_B_θ +
                2 * deriv_B_r * (1-β) -
                B_θ * (Y1 + 1)
            )
        ) - 2 * v_r * v_φ * (4 * β - 5)
    )


def Y5_func(ρ, B_θ, v_r, β, η_O, η_A, η_H):
    """
    Compute Y5
    """
    return (
        pi * ρ * (2 * β - 1) * (
            η_O + η_A + (η_H**2) / (η_O + η_A)
        )
    ) / (
        B_θ**2 + v_r * pi * ρ * (8 * β - 11) * (2 * β - 1) * (
            η_O + η_A + (η_H**2) / (η_O + η_A)
        )
    )


def Y6_func(ρ, B_θ, v_r, v_φ, β, η_O, η_A, η_H, Y5):
    """
    Compute Y6
    """
    return (
        1 - 1 / (
            v_r * (4 * β - 6)
        ) * (
            2 * Y5 * (
                (B_θ**2 * η_H) / (
                    2 * pi * ρ * (
                        η_O + η_A + (η_H**2) / (η_O + η_A)
                    )
                ) - v_φ
            ) * (
                v_φ - (
                    η_H * B_θ**2
                ) / (
                    4 * pi * ρ * (2 * β - 1) * (
                        (η_O + η_A)**2 + η_H**2
                    )
                )
            ) + (B_θ**2) / (4 * pi * ρ) * (
                1 / (η_O + η_A) -
                η_H / (
                    (η_O + η_A)**2 + η_H**2
                )
            )
        )
    ) ** -1


def dderiv_v_φ_midplane(ρ, B_θ, v_φ, dderiv_v_rM, η_O, η_H, η_A, Y5, Y4):
    """
    Compute v_φ'' around the midplane
    """
    return (
        Y5 * dderiv_v_rM * (
            (B_θ**2 * η_H) / (
                2 * pi * ρ * (
                    η_O + η_A +
                    (η_H**2) / (η_O + η_A)
                )
            ) - v_φ
        ) + Y5 * Y4
    )


def taylor_series(β, c_s, init_con, η_derivs=True):
    """
    Compute taylor series of v_r'', ρ'' and v_φ''
    """
    B_θ = init_con[2]
    v_r = init_con[3]
    v_φ = init_con[4]
    ρ = init_con[6]
    deriv_B_φ = init_con[7]
    η_O = init_con[8]
    η_A = init_con[9]
    η_H = init_con[10]
    deriv_B_r = - (
        B_θ * (1 - β) + (v_r * B_θ + deriv_B_φ * η_H) / (η_O + η_A)
    )

    dderiv_B_θ = (β - 2) * deriv_B_r + B_θ

    Y1 = Y1_func(ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, β, c_s)

    dderiv_ρ = ρ * Y1

    if η_derivs:
        dderiv_η_scale = dderiv_ρ / ρ / 2
    else:
        dderiv_η_scale = 0

    dderiv_η_O = dderiv_η_scale * η_O
    dderiv_η_A = dderiv_η_scale * η_A
    dderiv_η_H = dderiv_η_scale * η_H

    Y2 = Y2_func(
        B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A,
        η_H, dderiv_η_O, dderiv_η_A, dderiv_η_H,
    )
    Y3 = Y3_func(
        B_θ, v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H,
        dderiv_η_O, dderiv_η_A,
    )
    Y4 = Y4_func(ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, Y2, Y1)
    Y5 = Y5_func(ρ, B_θ, v_r, β, η_O, η_A, η_H)
    Y6 = Y6_func(ρ, B_θ, v_r, v_φ, β, η_O, η_A, η_H, Y5)

    dderiv_v_r = dderiv_v_r_midplane(
        ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ,
        β, η_O, η_H, η_A, Y6, Y5, Y4, Y3, Y2, Y1
    )

    dderiv_v_φ = dderiv_v_φ_midplane(
        ρ, B_θ, v_φ, dderiv_v_r, η_O, η_H, η_A, Y5, Y4
    )

    log.info("B_θ'': {}".format(dderiv_B_θ))
    log.info("v_r'': {}".format(dderiv_v_r))
    log.info("v_φ'': {}".format(dderiv_v_φ))
    log.info("ρ'': {}".format(dderiv_ρ))

    return dderiv_ρ, dderiv_v_r, dderiv_v_φ
