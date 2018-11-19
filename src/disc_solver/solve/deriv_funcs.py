# -*- coding: utf-8 -*-
"""
Computing derivatives
"""

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT

from numpy import errstate, zeros, sqrt, tan
import logbook

from ..utils import sec, ODEIndex

from .j_e_funcs import E_θ_func, J_func

log = logbook.Logger(__name__)


def Z_5_func(*, η_O, η_A, η_H, b_r, b_θ, b_φ, C):
    """
    Compute the value of the variable Z_5
    """
    return η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)


def deriv_B_φ_func(*, θ, γ, B_r, B_θ, B_φ, v_r, v_φ, v_θ, η_O, η_A, η_H, E_r):
    """
    Compute the derivative of B_φ, assuming E_r is being used
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    Z_5 = Z_5_func(η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C)

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


def deriv_B_r_func(*, θ, γ, B_r, B_θ, B_φ, v_r, v_θ, deriv_B_φ, η_O, η_A, η_H):
    """
    Compute the derivative of B_r
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    norm_B_r, norm_B_φ, norm_B_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    return (
        (
            v_θ * B_r - v_r * B_θ - deriv_B_φ * (
                η_H * norm_B_θ +
                η_A * norm_B_r * norm_B_φ
            ) + B_φ * (
                η_A * norm_B_φ * (
                    norm_B_r * tan(θ) -
                    norm_B_θ * (1/4 - γ)
                ) + η_H * (
                    norm_B_r * (1/4 - γ) +
                    norm_B_θ * tan(θ)
                )
            )
        ) / (
            η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
        ) - B_θ * (1/4 - γ)
    )


def deriv_E_r_func(
    *, γ, θ, v_r, v_φ, B_r, B_θ, B_φ, η_O, η_A, η_H, b_r, b_θ, b_φ, deriv_B_r,
    deriv_B_φ
):
    """
    Compute the derivative of E_r
    """
    J_r, J_θ, J_φ = J_func(
        γ=γ, θ=θ, B_θ=B_θ, B_φ=B_φ, deriv_B_φ=deriv_B_φ, deriv_B_r=deriv_B_r,
    )

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


def B_unit_derivs(*, B_r, B_φ, B_θ, deriv_B_r, deriv_B_φ, deriv_B_θ):
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
    deriv_B_φ, γ, deriv_η_O, deriv_η_A, deriv_η_H
):
    """
    Compute the derivative of B_φ
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        deriv_B_θ=deriv_B_θ
    )

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    A = A_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

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
                η_H * b_θ + η_A * b_r * b_φ
            ) + η_O + η_A * (
                1 - b_r ** 2 + b_r * b_θ * (1 / 4 - γ)
            ) + (1 / 4 - γ) * η_H * b_φ
        ) - B_φ * (
            deriv_η_O + deriv_η_A * (
                1 - b_r ** 2 + b_r * b_θ * (1 / 4 - γ)
            ) + (1 / 4 - γ) * deriv_η_H * b_φ - η_A * (
                2 * b_r * deriv_b_r - deriv_b_r * b_θ * (1 / 4 - γ) -
                b_r * deriv_b_θ * (1 / 4 - γ)
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
            )
        )
    ) / (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
    )


def Y_1_func(v_r, v_φ, deriv_B_r, deriv_B_φ, γ, a_0):
    """
    Compute Y_1
    """
    return v_φ ** 2 + a_0 * (
        deriv_B_r * (1/4 - γ + deriv_B_r) + deriv_B_φ ** 2
    ) - v_r ** 2 * γ * (1 - 4 * γ) / 2


def Y_2_func(
    v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_A, η_H, η_P,
    η_perp_sq, dderiv_η_P, dderiv_η_H,
):
    """
    Compute Y_2
    """
    return - η_P / η_perp_sq * (
        (3 / 4 - γ) * (
            (1 / 4 - γ) * η_P * deriv_B_φ + (1 / 4 - γ) * η_H * deriv_B_r +
            η_H * (deriv_B_φ ** 2 + deriv_B_r ** 2)
        ) - (v_r + η_H * deriv_B_φ) / η_P * (
            η_H * (
                deriv_B_r ** 2 + deriv_B_φ ** 2
            ) - dderiv_η_H + 2 * η_A * deriv_B_r * deriv_B_φ + η_H / η_P * (
                dderiv_η_P - 2 * η_A * deriv_B_φ ** 2
            )
        ) + deriv_B_φ * (
            η_A * (1 / 2 - 2 * γ) * deriv_B_r * (
                1 + η_H / η_P
            ) + η_H * (1 / 2 - 2 * γ) * deriv_B_φ * (
                1 - η_H / η_P
            ) + dderiv_η_P - 2 * η_A * deriv_B_r ** 2 + 4 * γ * v_r -
            2 * η_H ** 2 / η_P
        ) + v_φ * dderiv_B_θ + η_H * v_r * dderiv_B_θ / η_P +
        4 * γ * v_r * deriv_B_r * η_H / η_P
    )


def Y_3_func(
    v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_P, η_A, η_H, dderiv_η_P,
    dderiv_η_H,
):
    """
    Compute Y_3
    """
    return (
        dderiv_η_P * v_r + dderiv_η_P * η_H * deriv_B_φ -
        2 * v_r * η_A * deriv_B_φ ** 2 - 2 * η_A * η_H * deriv_B_φ ** 3
    ) / (η_P ** 2) - (
        4 * γ * v_r * deriv_B_r + (1 / 2 - 2 * γ) * η_A * deriv_B_φ ** 2 -
        (1 / 2 - 2 * γ) * η_H * deriv_B_r * deriv_B_φ - 2 * η_H * deriv_B_φ -
        η_H * deriv_B_r ** 2 * deriv_B_φ - η_H * deriv_B_φ ** 3 +
        dderiv_η_H * deriv_B_φ + 2 * η_A * deriv_B_r * deriv_B_φ ** 2 +
        v_r * dderiv_B_θ
    ) / η_P - dderiv_B_θ * (1 / 4 - γ)


def Y_4_func(a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, γ, Y_2, Y_1):
    """
    Compute Y_4
    """
    return 2 * a_0 * (
        Y_2 - deriv_B_φ * (deriv_B_r * (5/4 - γ) - Y_1)
    ) - 4 * γ * v_r * v_φ


def Y_5_func(a_0, v_r, γ, η_perp_sq):
    """
    Compute Y_5
    """
    return v_r * (1 - 8 * γ) + 2 * a_0 / η_perp_sq


def Y_6_func(a_0, v_r, v_φ, γ, η_perp_sq, η_H, η_P, Y_5):
    """
    Compute Y_6
    """
    return 1 / Y_5 * (
        2 * a_0 * η_H / η_perp_sq + v_φ
    ) * (
        1 + η_H * a_0 / (2 * η_perp_sq)
    ) - η_P * a_0 / (2 * η_perp_sq) - (2 * γ + 1 / 2) * v_r


def dderiv_v_r_midplane(
    a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_perp_sq, η_H, η_P,
    Y_6, Y_5, Y_4, Y_3, Y_2, Y_1
):
    """
    Compute v_r'' around the midplane
    """
    return (
        v_φ * Y_4 / Y_5 + 4 * γ ** 2 * v_r ** 2 + a_0 / 2 * (
            dderiv_B_θ * deriv_B_r + 2 * deriv_B_φ ** 2 * (1 / 4 - γ) + Y_3 -
            Y_2 * η_H / η_P + η_H * Y_4 / (η_perp_sq * Y_5) +
            2 * dderiv_B_θ * (1 / 4 - γ) + Y_1 * (deriv_B_r + 1 / 4 - γ)
        )
    ) / Y_6


def dderiv_v_φ_midplane(a_0, v_φ, dderiv_v_r, η_perp_sq, η_H, Y_5, Y_4):
    """
    Compute v_φ'' around the midplane
    """
    return Y_4 / Y_5 - dderiv_v_r / Y_5 * (2 * a_0 * η_H / η_perp_sq + v_φ)


def ddderiv_B_φ_midplane(
    η_H, dderiv_v_r, η_perp_sq, dderiv_v_φ, η_P, Y_2
):
    """
    Compute B_φ''' around the midplane
    """
    return Y_2 - (η_H * dderiv_v_r / η_perp_sq) - (
        dderiv_v_φ * η_P / η_perp_sq
    )


def ddderiv_B_r_midplane(Y_3, η_H, η_P, dderiv_v_r, dderiv_B_φ_prime):
    """
    Compute B_r''' around the midplane
    """
    return Y_3 - (η_H * dderiv_B_φ_prime + dderiv_v_r) / η_P


def ddderiv_v_θ_midplane(dderiv_ρ, dderiv_v_r, v_r, γ):
    """
    Compute v_θ''' around the midplane
    """
    return 4 * γ * v_r * (dderiv_ρ - 1 - dderiv_v_r / (2 * v_r))


def dderiv_η_skw_midplane(*, dderiv_ρ, deriv_B_r, deriv_B_φ, dderiv_B_θ):
    """
    Compute the derivative of η assuming the same form as in SKW
    """
    return 2 * (deriv_B_r ** 2 + deriv_B_φ ** 2 + dderiv_B_θ) - dderiv_ρ


def deriv_B_r_midplane_func(*, γ, deriv_B_φ, η_H, η_P, v_r):
    """
    Compute B_r' at the midplane
    """
    return γ - 1/4 - (deriv_B_φ * η_H + v_r) / η_P


def taylor_series(*, γ, a_0, init_con, η_derivs):
    """
    Compute taylor series for second and third order components.
    """
    # pylint: disable=too-many-statements
    v_r = init_con[ODEIndex.v_r]
    v_φ = init_con[ODEIndex.v_φ]
    deriv_B_φ = init_con[ODEIndex.B_φ_prime]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]

    η_P = η_O + η_A
    η_perp_sq = η_P ** 2 + η_H ** 2

    deriv_B_r = deriv_B_r_midplane_func(
        γ=γ, deriv_B_φ=deriv_B_φ, η_H=η_H, η_P=η_P, v_r=v_r
    )

    dderiv_B_θ = 1 - (γ + 3/4) * deriv_B_r

    Y_1 = Y_1_func(v_r, v_φ, deriv_B_r, deriv_B_φ, γ, a_0)

    dderiv_ρ = - Y_1

    if η_derivs:
        dderiv_η_scale = dderiv_η_skw_midplane(
            dderiv_ρ=dderiv_ρ, dderiv_B_θ=dderiv_B_θ, deriv_B_r=deriv_B_r,
            deriv_B_φ=deriv_B_φ,
        )
    else:
        dderiv_η_scale = 0

    dderiv_η_O = dderiv_η_scale * η_O
    dderiv_η_A = dderiv_η_scale * η_A
    dderiv_η_H = dderiv_η_scale * η_H
    dderiv_η_P = dderiv_η_O + dderiv_η_A

    Y_2 = Y_2_func(
        v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_A, η_H, η_P,
        η_perp_sq, dderiv_η_P, dderiv_η_H,
    )
    Y_3 = Y_3_func(
        v_r, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_P, η_A, η_H, dderiv_η_P,
        dderiv_η_H
    )
    Y_4 = Y_4_func(a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, γ, Y_2, Y_1)
    Y_5 = Y_5_func(a_0, v_r, γ, η_perp_sq)
    Y_6 = Y_6_func(a_0, v_r, v_φ, γ, η_perp_sq, η_H, η_P, Y_5)

    dderiv_v_r = dderiv_v_r_midplane(
        a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, dderiv_B_θ, γ, η_perp_sq, η_H,
        η_P, Y_6, Y_5, Y_4, Y_3, Y_2, Y_1
    )

    dderiv_v_φ = dderiv_v_φ_midplane(
        a_0, v_φ, dderiv_v_r, η_perp_sq, η_H, Y_5, Y_4
    )

    dderiv_B_φ_prime = ddderiv_B_φ_midplane(
        η_H, dderiv_v_r, η_perp_sq, dderiv_v_φ, η_P, Y_2
    )

    ddderiv_B_r = ddderiv_B_r_midplane(
        Y_3, η_H, η_P, dderiv_v_r, dderiv_B_φ_prime
    )

    ddderiv_v_θ = ddderiv_v_θ_midplane(dderiv_ρ, dderiv_v_r, v_r, γ)

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

    derivs = zeros(len(ODEIndex))

    derivs[ODEIndex.B_θ] = dderiv_B_θ
    derivs[ODEIndex.B_r] = ddderiv_B_r
    derivs[ODEIndex.ρ] = dderiv_ρ
    derivs[ODEIndex.v_r] = dderiv_v_r
    derivs[ODEIndex.v_φ] = dderiv_v_φ
    derivs[ODEIndex.v_θ] = ddderiv_v_θ
    derivs[ODEIndex.B_φ_prime] = dderiv_B_φ_prime

    derivs[ODEIndex.η_O] = dderiv_η_O
    derivs[ODEIndex.η_A] = dderiv_η_A
    derivs[ODEIndex.η_H] = dderiv_η_H

    return derivs


def deriv_v_θ_sonic(
    *, a_0, ρ, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_θ, v_φ, deriv_v_r,
    deriv_v_φ, deriv_B_r, deriv_B_θ, B_φ_prime, γ, deriv_η_O, deriv_η_A,
    deriv_η_H
):
    """
    Compute v_θ' at the sonic point
    """
    # THIS IS CURRENTLY INCORRECT!!!
    deriv_B_φ = B_φ_prime

    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        deriv_B_θ=deriv_B_θ
    )

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    A = A_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    dderiv_B_φ_mod = (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
    ) ** -1 * (
        v_φ * B_r - 4 / (3 - 4 * γ) * (
            deriv_v_φ * B_θ + v_φ * deriv_B_θ
        ) + (B_θ * (1/4 - γ) + deriv_B_r) * (
            η_H * b_r - η_A * b_θ * b_φ
        ) - A * v_θ * B_r - C * v_θ * deriv_B_r + A * v_r * B_θ +
        C * deriv_v_r * B_θ + C * v_r * deriv_B_θ + deriv_B_φ * (
            v_θ * 4 / (3 - 4 * γ) + η_O * tan(θ) - deriv_η_O -
            deriv_η_H * C * b_θ - deriv_η_A * (
                1 - b_r ** 2 - C * b_r * b_φ
            ) + η_H * (
                b_φ * (γ + 3/4) + C * (
                    b_r * (1/4 - γ) + b_θ * tan(θ) + deriv_b_θ
                ) - A * b_θ
            ) + η_A * (
                tan(θ) * (1 - b_r ** 2) + (1/4 - γ) * b_r * b_φ - C * b_φ * (
                    b_θ * (1/4 - γ) + b_r * tan(θ)
                ) + 2 * b_r * deriv_b_r - A * b_r * b_φ -
                C * deriv_b_r * b_φ - C * b_r * deriv_b_φ + b_r * b_θ
            )
        ) + B_φ * (
            deriv_η_O * tan(θ) + η_O * (
                sec(θ) ** 2 + γ - 1/4
            ) - v_r + η_H * (
                A * (
                    b_r * (1/4 - γ) + b_θ * tan(θ)
                ) + C * b_φ * (
                    deriv_b_r * (1/4 - γ) + deriv_b_θ * tan(θ) +
                    b_θ * sec(θ) ** 2
                ) - b_φ * tan(θ) - deriv_b_φ * (1/4 - γ)
            ) + deriv_η_H * (
                C * (b_r * (1/4 - γ) + b_θ * tan(θ)) - b_φ * (1/4 - γ)
            ) + η_A * (
                sec(θ) ** 2 * (1 - b_r ** 2) + 2 * tan(θ) * b_r * deriv_b_r +
                (1/4 - γ) * deriv_b_r * b_θ + (1/4 - γ) * b_r * deriv_b_θ - (
                    A * b_φ + C * deriv_b_φ
                ) * (
                    b_θ * (1/4 - γ) + b_r * tan(θ)
                ) - C * b_φ * (
                    deriv_b_θ * (1/4 - γ) + deriv_b_r * tan(θ) +
                    b_r * sec(θ) ** 2
                ) - (1/4 - γ) * (1 - b_θ ** 2) - tan(θ) * b_r * b_θ
            ) + deriv_η_A * (
                tan(θ) * (1 - b_r ** 2) + (1/4 - γ) * b_r * b_θ - C * b_φ * (
                    b_θ * (1/4 - γ) + b_r * tan(θ)
                )
            )
        )
    )

    dderiv_B_r_mod = (
        deriv_η_O + deriv_η_A * (1 - b_φ ** 2) - 2 * η_A * b_φ * deriv_b_φ
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    ) * (
        v_θ * B_r - v_r * B_θ + B_φ_prime * (
            η_H * b_θ - η_A * b_r * b_φ
        ) + B_φ * (
            η_A * b_φ * (b_θ * (1 / 4 - γ) + b_r * tan(θ)) -
            η_H * (b_r * (1 / 4 - γ) + b_θ * tan(θ))
        )
    ) + 1 / (η_O + η_A * (1 - b_φ ** 2)) * (
        v_θ * deriv_B_r - deriv_v_r * B_θ - v_r * deriv_B_θ +
        dderiv_B_φ_mod * (
            η_H * b_θ - η_A * b_r * b_φ
        ) + B_φ_prime * (
            η_A * b_φ * (
                b_θ * (1 / 4 - γ) - b_r * (1 - tan(θ))
            ) + η_H * (
                b_θ * (1 - tan(θ)) - b_r * (1 / 4 - γ)
            )
        ) + B_φ * (
            (deriv_η_A * b_φ + η_A * deriv_b_φ) * (
                b_θ * (1 / 4 - γ) + b_r * tan(θ)
            ) + η_A * b_φ * (
                deriv_b_θ * (1 / 4 - γ) + deriv_b_r * tan(θ) +
                b_r * sec(θ) ** 2
            ) - deriv_η_H * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            ) - η_H * (
                deriv_b_r * (1 / 4 - γ) + deriv_b_θ * tan(θ) +
                b_θ * sec(θ) ** 2
            )
        )
    ) - deriv_B_θ * (1 / 4 - γ)

    a = - 2 * v_θ
    b = v_r * v_θ + tan(θ) * (v_φ ** 2 + 1) + a_0 / ρ * (
        (1 / 4 - γ) * B_θ * B_r + B_r * deriv_B_r + B_φ * B_φ_prime -
        B_φ ** 2 * tan(θ)
    ) * (v_θ + 1) / v_θ + (
        v_θ * a_0 * B_φ * (4 / (3 - 4 * γ) - C * B_r) * (
            1 + (η_H * b_θ - η_A * b_r * b_φ) / (η_O + η_A * (1 - b_φ ** 2))
        )
    ) / (
        ρ * (
            η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
        )
    ) + (
        B_r ** 2 * v_θ * a_0
    ) / (
        ρ * (η_O + η_A * (1 - b_φ ** 2))
    )

    c = deriv_v_r * (v_θ ** 2 - 4 * γ) / 2 + v_θ * (
        sec(θ) ** 2 * (v_φ ** 2 + 1) + tan(θ) * v_φ * deriv_v_φ + a_0 / ρ * (
            (1 / 4 - γ) * (B_θ * deriv_B_r + deriv_B_θ * B_r) +
            deriv_B_r ** 2 + B_r * dderiv_B_r_mod + B_φ_prime ** 2 +
            B_φ * dderiv_B_φ_mod - B_φ ** 2 * sec(θ) ** 2 -
            2 * B_φ * B_φ_prime * tan(θ) + (2 * γ * v_r / v_θ - tan(θ)) * (
                (1 / 4 - γ) * B_θ * B_r + B_r * deriv_B_r +
                B_φ * B_φ_prime * - B_φ ** 2 * tan(θ)
            )
        )
    )

    return (-b - sqrt(b**2 - 4 * a * c)) / (2 * a)


def get_taylor_first_order(*, init_con, γ):
    """
    Compute first order taylor series at θ.

    Note that η' is assumed to be proportional to ρ' (i.e. 0)
    """
    v_r = init_con[ODEIndex.v_r]
    B_φ_prime = init_con[ODEIndex.B_φ_prime]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]
    η_P = η_O + η_A

    first_order = zeros(len(ODEIndex))

    first_order[ODEIndex.B_r] = deriv_B_r_midplane_func(
        γ=γ, deriv_B_φ=B_φ_prime, η_H=η_H, η_P=η_P, v_r=v_r
    )
    first_order[ODEIndex.B_φ] = B_φ_prime
    first_order[ODEIndex.v_θ] = - 2 * γ * v_r
    log.debug("First order taylor {}".format(first_order))

    return first_order


def get_taylor_second_order(
    *, init_con, γ, a_0, η_derivs
):
    """
    Return the second order constants of a taylor series off the midplane.
    """
    second_order = zeros(len(ODEIndex))

    derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs
    )

    second_order[ODEIndex.B_θ] = derivs[ODEIndex.B_θ]
    second_order[ODEIndex.ρ] = derivs[ODEIndex.ρ]
    second_order[ODEIndex.v_r] = derivs[ODEIndex.v_r]
    second_order[ODEIndex.v_φ] = derivs[ODEIndex.v_φ]
    second_order[ODEIndex.B_φ_prime] = derivs[ODEIndex.B_φ_prime]

    second_order[ODEIndex.η_O] = derivs[ODEIndex.η_O]
    second_order[ODEIndex.η_A] = derivs[ODEIndex.η_A]
    second_order[ODEIndex.η_H] = derivs[ODEIndex.η_H]
    log.debug("Second order taylor {}".format(second_order))

    return second_order


def get_taylor_third_order(
    *, init_con, γ, a_0, η_derivs
):
    """
    Return the third order constants of a taylor series off the midplane.

    Note that η' is assumed to be proportional to ρ' (i.e. 0)
    """
    third_order = zeros(len(ODEIndex))

    derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs
    )

    third_order[ODEIndex.B_r] = derivs[ODEIndex.B_r]
    third_order[ODEIndex.B_φ] = derivs[ODEIndex.B_φ_prime]
    third_order[ODEIndex.v_θ] = derivs[ODEIndex.v_φ]
    log.debug("Third order taylor {}".format(third_order))

    return third_order
