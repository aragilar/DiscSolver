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
            η_H * norm_B_r - v_θ * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) - η_A * norm_B_θ * norm_B_φ
        ) + B_r * (
            v_φ - deriv_v_θ * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) - v_θ * magic_deriv
        ) - B_θ * (
            deriv_v_φ * 2 / (2 * β - 1) - deriv_v_r * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) - v_r * magic_deriv + (1-β) * (
                η_H * norm_B_r -
                η_A * norm_B_θ * norm_B_φ
            )
        ) + deriv_B_θ * (
            v_r * (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) + 2 / (2 * β - 1) * v_φ
        ) + deriv_B_φ * (
            v_θ * 2 / (2 * β - 1) + η_O * tan(θ) +
            η_H * norm_B_φ * (2-β) + η_A * (
                tan(θ) * (1 - norm_B_r**2) +
                β * norm_B_r * norm_B_θ
            ) - (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) * (
                η_H * (
                    norm_B_r * (1 - β) - norm_B_θ * tan(θ)
                ) - η_A * norm_B_φ * (
                    norm_B_θ * (1 - β) - norm_B_r * tan(θ)
                )
            )
        ) + B_φ * (
            η_O * (1 - β) +
            η_H * norm_B_φ * tan(θ) -
            η_A * (
                (1 - β) * (1 - norm_B_θ**2) -
                tan(θ) * norm_B_r * norm_B_θ
            ) + v_r - 2 / (2 * β - 1) * deriv_v_θ -
            η_O * sec(θ)**2 -
            η_H * deriv_norm_B_φ * (1 - β) -
            η_A * (
                sec(θ)**2 * (1 - norm_B_r**2) +
                2 * tan(θ) * norm_B_r * deriv_norm_B_r -
                (1 - β) * (
                    norm_B_r * deriv_norm_B_θ +
                    norm_B_θ * deriv_norm_B_r
                )
            ) + magic_deriv * (
                η_H * (
                    norm_B_r * (1 - β) - norm_B_θ * tan(θ)
                ) - η_A * norm_B_φ * (
                    norm_B_θ * (1 - β) - norm_B_r * tan(θ)
                )
            ) + (
                η_H * norm_B_θ + η_A * norm_B_r * norm_B_φ
            ) / (
                η_O + η_A * (1-norm_B_φ**2)
            ) * (
                η_H * (
                    deriv_norm_B_r * (1-β) -
                    deriv_norm_B_θ * tan(θ) -
                    sec(θ)**2 * norm_B_θ
                ) - η_A * (
                    deriv_norm_B_φ * (
                        norm_B_θ * (1 - β) - norm_B_r * tan(θ)
                    ) + norm_B_φ * (
                        deriv_norm_B_θ * (1 - β) -
                        deriv_norm_B_r * tan(θ) -
                        sec(θ)**2 * norm_B_r
                    )
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
):
    """
    Compute Y2
    """
    return 1 / (η_O + η_A + (η_H**2) / (η_A + η_O)) * (
        dderiv_B_θ * ((η_H * v_r) / (η_O + η_A) - (2 * v_φ) / (2*β - 1)) +
        deriv_B_φ * (
            η_O * (3 - β) + η_A * (2 - β) - v_r + (2 * v_r * (4*β - 5)) /
            (2*β - 1)
        ) + deriv_B_r * (
            v_φ - η_H * (1 - β + (v_r * (4*β - 5)) / (η_O + η_A))
        ) + (
            2 * deriv_B_φ * deriv_B_r * η_A * v_r
        ) / (B_θ * (η_O + η_A)) + (
            η_H * deriv_B_φ**2
        ) / (B_θ) * (1 - (v_r) / (η_O + η_A) * (1 + (η_A) / (η_O + η_A))) + (
            deriv_B_r**2 * η_H
        ) / (B_θ**2) * (
            1 - 2 * η_A / (η_O + η_A)
        )
    )


def Y3_func(B_θ, v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H):
    """
    Compute Y3
    """
    return - 1 / (η_A + η_O) * (
        v_r * (
            dderiv_B_θ - (4 * β - 5) * deriv_B_r
        ) + (
            2 * η_A * deriv_B_φ**2
        ) / B_θ * (
            1 - β +
            deriv_B_r / B_θ +
            v_r / (η_O + η_A)
        ) + η_H * deriv_B_φ * (
            2 + deriv_B_r / B_θ * (
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


def taylor_series(β, c_s, η_O, η_A, η_H, init_con):
    """
    Compute taylor series of v_r'', ρ'' and v_φ''
    """
    B_θ = init_con[2]
    v_r = init_con[3]
    v_φ = init_con[4]
    ρ = init_con[6]
    deriv_B_φ = init_con[7]
    deriv_B_r = - (
        B_θ * (1 - β) + (v_r * B_θ + deriv_B_φ * η_H) / (η_O + η_A)
    )

    dderiv_B_θ = (β - 2) * deriv_B_r + B_θ

    Y1 = Y1_func(ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, β, c_s)
    Y2 = Y2_func(
        B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H,
    )
    Y3 = Y3_func(B_θ, v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, η_O, η_A, η_H)
    Y4 = Y4_func(ρ, B_θ, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, β, Y2, Y1)
    Y5 = Y5_func(ρ, B_θ, v_r, β, η_O, η_A, η_H)
    Y6 = Y6_func(ρ, B_θ, v_r, v_φ, β, η_O, η_A, η_H, Y5)

    dderiv_ρ = ρ * Y1

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
