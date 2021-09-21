# -*- coding: utf-8 -*-
"""
Computing derivatives at the midplane
"""

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT

from numpy import zeros
import logbook

from ..utils import ODEIndex

from .config import B_φ_prime_boundary_func

log = logbook.Logger(__name__)


def dderiv_b_θ_midplane(*, deriv_B_r, deriv_B_φ):
    """
    Compute b_θ'' at the midplane
    """
    return - (deriv_B_r ** 2 + deriv_B_φ ** 2)


def midplane_C_func(*, η_H, η_P):
    """
    Compute C at the midplane
    """
    return η_H / η_P


def midplane_deriv_A_func(
    *, dderiv_η_H, dderiv_b_θ, C, η_A, deriv_b_φ, deriv_b_r, η_P, η_H,
    dderiv_η_P,
):
    """
    Compute A' at the midplane
    """
    return (
        dderiv_η_H + η_H * dderiv_b_θ - 2 * η_A * deriv_b_r * deriv_b_φ - C * (
            dderiv_η_P - 2 * η_A * deriv_b_φ ** 2
        )
    ) / η_P


def midplane_Z_5_func(*, η_perp_sq, η_P):
    """
    Compute Z_5 at the midplane
    """
    return η_perp_sq / η_P


def dderiv_Z_5_midplane(
    *, dderiv_η_P, η_A, deriv_b_φ, deriv_A, η_H, C, dderiv_η_H, dderiv_b_θ,
    deriv_b_r,
):
    """
    Compute Z_5'' at the midplane
    """
    return dderiv_η_P - 2 * η_A * deriv_b_φ ** 2 + deriv_A * η_H + C * (
        dderiv_η_H + η_H * dderiv_b_θ - 2 * η_A * deriv_b_r * deriv_b_φ
    )


def Y_1_func(*, v_r, v_φ, deriv_B_r, deriv_B_φ, γ, a_0):
    """
    Compute Y_1
    """
    return v_φ ** 2 + a_0 * (
        deriv_B_r * (1/4 - γ + deriv_B_r) + deriv_B_φ ** 2
    ) - v_r ** 2 * γ * (1 - 4 * γ) / 2


def Y_2_func(
    *, deriv_B_r, deriv_v_θ, v_r, dderiv_B_θ, deriv_B_φ, dderiv_η_H, η_H,
    deriv_b_r, γ, dderiv_b_θ, dderiv_η_P, η_A, η_P,
):
    """
    Compute Y_2
    """
    return (
        deriv_v_θ * deriv_B_r - v_r * dderiv_B_θ - deriv_B_φ * (
            dderiv_η_H + η_H * (deriv_b_r * (1/4 - γ) + 1 + dderiv_b_θ)
        ) - (dderiv_η_P - 2 * η_A * deriv_b_r ** 2) * (deriv_B_r + 1/4 - γ)
    ) / η_P - dderiv_B_θ * (1/4 - γ)


def Y_3_func(
    *, deriv_A, v_r, C, deriv_v_θ, deriv_B_r, dderiv_B_θ, dderiv_E_r, v_φ,
    deriv_B_φ, η_P, γ, η_H, deriv_b_φ, η_A, deriv_b_r, dderiv_Z_5, Z_5,
):
    """
    Compute Y_3
    """
    return - (
        deriv_A * v_r - C * (
            deriv_v_θ * deriv_B_r - v_r * dderiv_B_θ
        ) + dderiv_E_r + v_φ * dderiv_B_θ - deriv_B_φ * (
            deriv_v_θ + η_P + (1/4 - γ) * (
                η_H * deriv_b_φ + η_A * deriv_b_r
            ) + C * (
                η_A * deriv_b_φ * (1/4 - γ) - η_H * (deriv_b_r * (1/4 - γ) + 1)
            )
        ) + deriv_B_φ * dderiv_Z_5
    ) / Z_5


def Y_4_func(*, a_0, v_r, v_φ, deriv_B_r, deriv_B_φ, γ, Y_2, Y_1):
    """
    Compute Y_4
    """
    return 2 * a_0 * (
        Y_2 - deriv_B_φ * (deriv_B_r * (5/4 - γ) - Y_1)
    ) - 4 * γ * v_r * v_φ


def Y_5_func(*, a_0, v_r, γ, Z_5):
    """
    Compute Y_5
    """
    return v_r * (1 - 8 * γ) + 2 * a_0 / Z_5


def Y_6_func(*, v_φ, Y_5, a_0, C, Z_5, γ, v_r, η_P, η_H):
    """
    Compute Y_6
    """
    return v_φ / Y_5 * (
        2 * a_0 * C / Z_5 + v_φ
    ) - (4 * γ + 1) * v_r / 2 - a_0 / (2 * η_P) * (
        C * η_H / Z_5 - 1 - η_H * (2 * a_0 * C / Z_5 + v_φ) / (Z_5 * Y_5)
    )


def dderiv_E_r_midplane(*, v_r, v_φ, deriv_B_r, deriv_B_φ, γ, η_H, η_P):
    """
    Compute E_r'' around the midplane
    """
    return - (3 / 4 - γ) * (
        deriv_B_r * (v_φ - η_H * (1 / 4 - γ)) - deriv_B_φ * (
            v_r + η_P * (1 / 4 - γ)
        ) - η_H * (deriv_B_φ ** 2 + deriv_B_r ** 2)
    )


def dderiv_v_r_midplane(
    *, a_0, Y_3, Y_2, Y_1, Y_6, Y_5, Y_4, η_H, η_P, Z_5, v_φ, v_r, γ,
    dderiv_B_θ, deriv_B_r, deriv_B_φ,
):
    """
    Compute v_r'' around the midplane
    """
    return (
        a_0 / 2 * (
            Y_3 - Y_2 * η_H / η_P + η_H * Y_4 / (η_P * Y_5 * Z_5)
        ) + v_φ * Y_4 / Y_5 + 4 * γ ** 2 * v_r ** 2 + a_0 / 2 * (
            dderiv_B_θ * deriv_B_r + 2 * deriv_B_φ ** 2 * (1 / 4 - γ) +
            2 * dderiv_B_θ * (1 / 4 - γ) + Y_1 * (deriv_B_r + (1 / 4 - γ))
        )
    ) / Y_6


def dderiv_v_φ_midplane(*, a_0, v_φ, dderiv_v_r, C, Z_5, Y_5, Y_4):
    """
    Compute v_φ'' around the midplane
    """
    return Y_4 / Y_5 - dderiv_v_r / Y_5 * (2 * a_0 * C / Z_5 + v_φ)


def ddderiv_B_φ_midplane(*, dderiv_v_r, dderiv_v_φ, Y_2, C, Z_5):
    """
    Compute B_φ''' around the midplane
    """
    return Y_2 - (C * dderiv_v_r + dderiv_v_φ) / Z_5


def ddderiv_B_r_midplane(*, Y_3, η_H, η_P, dderiv_v_r, dderiv_B_φ_prime):
    """
    Compute B_r''' around the midplane
    """
    return Y_3 - (η_H * dderiv_B_φ_prime + dderiv_v_r) / η_P


def ddderiv_v_θ_midplane(*, dderiv_ρ, dderiv_v_r, v_r, γ):
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


def taylor_series(*, γ, a_0, init_con, η_derivs, use_E_r):
    """
    Compute taylor series for second and third order components.
    """
    # pylint: disable=too-many-statements
    v_r = init_con[ODEIndex.v_r]
    v_φ = init_con[ODEIndex.v_φ]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]

    deriv_B_φ = B_φ_prime_boundary_func(v_r=v_r, v_φ=v_φ, a_0=a_0)

    η_P = η_O + η_A
    η_perp_sq = η_P ** 2 + η_H ** 2

    deriv_v_θ = - 2 * γ * v_r
    deriv_B_r = deriv_B_r_midplane_func(
        γ=γ, deriv_B_φ=deriv_B_φ, η_H=η_H, η_P=η_P, v_r=v_r
    )

    dderiv_B_θ = 1 - (γ + 3/4) * deriv_B_r
    dderiv_E_r = dderiv_E_r_midplane(
        v_r=v_r, v_φ=v_φ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ, γ=γ,
        η_H=η_H, η_P=η_P
    )

    Y_1 = Y_1_func(
        v_r=v_r, v_φ=v_φ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ, γ=γ,
        a_0=a_0,
    )

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

    dderiv_b_θ = dderiv_b_θ_midplane(deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ)
    # At the midplane, these simplify to each other
    deriv_b_r = deriv_B_r
    deriv_b_φ = deriv_B_φ

    C = midplane_C_func(η_H=η_H, η_P=η_P)
    deriv_A = midplane_deriv_A_func(
        dderiv_η_H=dderiv_η_H, dderiv_b_θ=dderiv_b_θ, C=C, η_A=η_A,
        deriv_b_φ=deriv_b_φ, deriv_b_r=deriv_b_r, η_P=η_P, η_H=η_H,
        dderiv_η_P=dderiv_η_P,
    )
    Z_5 = midplane_Z_5_func(η_perp_sq=η_perp_sq, η_P=η_P)
    dderiv_Z_5 = dderiv_Z_5_midplane(
        dderiv_η_P=dderiv_η_P, η_A=η_A, deriv_b_φ=deriv_b_φ, deriv_A=deriv_A,
        η_H=η_H, C=C, dderiv_η_H=dderiv_η_H, dderiv_b_θ=dderiv_b_θ,
        deriv_b_r=deriv_b_r,
    )

    Y_2 = Y_2_func(
        deriv_B_r=deriv_B_r, deriv_v_θ=deriv_v_θ, v_r=v_r,
        dderiv_B_θ=dderiv_B_θ, deriv_B_φ=deriv_B_φ, dderiv_η_H=dderiv_η_H,
        η_H=η_H, deriv_b_r=deriv_b_r, γ=γ, dderiv_b_θ=dderiv_b_θ,
        dderiv_η_P=dderiv_η_P, η_A=η_A, η_P=η_P,
    )
    Y_3 = Y_3_func(
        deriv_A=deriv_A, v_r=v_r, C=C, deriv_v_θ=deriv_v_θ,
        deriv_B_r=deriv_B_r, dderiv_B_θ=dderiv_B_θ, dderiv_E_r=dderiv_E_r,
        v_φ=v_φ, deriv_B_φ=deriv_B_φ, η_P=η_P, γ=γ, η_H=η_H,
        deriv_b_φ=deriv_b_φ, η_A=η_A, deriv_b_r=deriv_b_r,
        dderiv_Z_5=dderiv_Z_5, Z_5=Z_5,
    )
    Y_4 = Y_4_func(
        a_0=a_0, v_r=v_r, v_φ=v_φ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        γ=γ, Y_2=Y_2, Y_1=Y_1,
    )
    Y_5 = Y_5_func(a_0=a_0, v_r=v_r, γ=γ, Z_5=Z_5)
    Y_6 = Y_6_func(
        v_φ=v_φ, Y_5=Y_5, a_0=a_0, C=C, Z_5=Z_5, γ=γ, v_r=v_r, η_P=η_P,
        η_H=η_H,
    )

    dderiv_v_r = dderiv_v_r_midplane(
        a_0=a_0, Y_3=Y_3, Y_2=Y_2, Y_1=Y_1, Y_6=Y_6, Y_5=Y_5, Y_4=Y_4, η_H=η_H,
        η_P=η_P, Z_5=Z_5, v_φ=v_φ, v_r=v_r, γ=γ, dderiv_B_θ=dderiv_B_θ,
        deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
    )

    dderiv_v_φ = dderiv_v_φ_midplane(
        a_0=a_0, v_φ=v_φ, dderiv_v_r=dderiv_v_r, C=C, Z_5=Z_5, Y_5=Y_5,
        Y_4=Y_4,
    )

    dderiv_B_φ_prime = ddderiv_B_φ_midplane(
        dderiv_v_r=dderiv_v_r, dderiv_v_φ=dderiv_v_φ, Y_2=Y_2, C=C, Z_5=Z_5,
    )

    ddderiv_B_r = ddderiv_B_r_midplane(
        Y_3=Y_3, η_H=η_H, η_P=η_P, dderiv_v_r=dderiv_v_r,
        dderiv_B_φ_prime=dderiv_B_φ_prime,
    )

    ddderiv_v_θ = ddderiv_v_θ_midplane(
        dderiv_ρ=dderiv_ρ, dderiv_v_r=dderiv_v_r, v_r=v_r, γ=γ,
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

    derivs = zeros(len(ODEIndex))

    derivs[ODEIndex.B_θ] = dderiv_B_θ
    derivs[ODEIndex.B_r] = ddderiv_B_r
    derivs[ODEIndex.ρ] = dderiv_ρ
    derivs[ODEIndex.v_r] = dderiv_v_r
    derivs[ODEIndex.v_φ] = dderiv_v_φ
    derivs[ODEIndex.v_θ] = ddderiv_v_θ
    derivs[ODEIndex.B_φ] = dderiv_B_φ_prime
    if use_E_r:
        derivs[ODEIndex.E_r] = dderiv_E_r
    else:
        derivs[ODEIndex.B_φ_prime] = dderiv_B_φ_prime

    derivs[ODEIndex.η_O] = dderiv_η_O
    derivs[ODEIndex.η_A] = dderiv_η_A
    derivs[ODEIndex.η_H] = dderiv_η_H

    return derivs


def get_taylor_first_order(*, init_con, γ, a_0):
    """
    Compute first order taylor series at θ.

    Note that η' is assumed to be proportional to ρ' (i.e. 0)
    """
    v_r = init_con[ODEIndex.v_r]
    v_φ = init_con[ODEIndex.v_φ]
    η_O = init_con[ODEIndex.η_O]
    η_A = init_con[ODEIndex.η_A]
    η_H = init_con[ODEIndex.η_H]
    η_P = η_O + η_A

    B_φ_prime = B_φ_prime_boundary_func(v_r=v_r, v_φ=v_φ, a_0=a_0)

    first_order = zeros(len(ODEIndex))

    first_order[ODEIndex.B_r] = deriv_B_r_midplane_func(
        γ=γ, deriv_B_φ=B_φ_prime, η_H=η_H, η_P=η_P, v_r=v_r
    )
    first_order[ODEIndex.B_φ] = B_φ_prime
    first_order[ODEIndex.v_θ] = - 2 * γ * v_r
    log.debug("First order taylor {}".format(first_order))

    return first_order


def get_taylor_second_order(
    *, init_con, γ, a_0, η_derivs, use_E_r=False
):
    """
    Return the second order constants of a taylor series off the midplane.
    """
    second_order = zeros(len(ODEIndex))

    derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs, use_E_r=use_E_r,
    )

    second_order[ODEIndex.B_θ] = derivs[ODEIndex.B_θ]
    second_order[ODEIndex.ρ] = derivs[ODEIndex.ρ]
    second_order[ODEIndex.v_r] = derivs[ODEIndex.v_r]
    second_order[ODEIndex.v_φ] = derivs[ODEIndex.v_φ]

    second_order[ODEIndex.η_O] = derivs[ODEIndex.η_O]
    second_order[ODEIndex.η_A] = derivs[ODEIndex.η_A]
    second_order[ODEIndex.η_H] = derivs[ODEIndex.η_H]

    if use_E_r:
        second_order[ODEIndex.E_r] = derivs[ODEIndex.E_r]
    else:
        second_order[ODEIndex.B_φ_prime] = derivs[ODEIndex.B_φ_prime]

    log.debug("Second order taylor {}".format(second_order))

    return second_order


def get_taylor_third_order(
    *, init_con, γ, a_0, η_derivs, use_E_r=False
):
    """
    Return the third order constants of a taylor series off the midplane.

    Note that η' is assumed to be proportional to ρ' (i.e. 0)
    """
    third_order = zeros(len(ODEIndex))

    derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs, use_E_r=use_E_r,
    )

    third_order[ODEIndex.B_r] = derivs[ODEIndex.B_r]
    third_order[ODEIndex.B_φ] = derivs[ODEIndex.B_φ]
    third_order[ODEIndex.v_θ] = derivs[ODEIndex.v_φ]
    log.debug("Third order taylor {}".format(third_order))

    return third_order
