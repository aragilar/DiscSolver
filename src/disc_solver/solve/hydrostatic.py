# -*- coding: utf-8 -*-
"""
The system of odes
"""

import logbook

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT
from numpy import sqrt, tan, copy, errstate, degrees, isreal, float64

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from .config import define_conditions
from .deriv_funcs import B_unit_derivs, A_func, C_func
from .utils import (
    error_handler, rad_to_scaled, scaled_to_rad,
    SolverError,
)

from ..float_handling import float_type
from ..file_format import InternalData, Solution
from ..utils import ODEIndex, sec, solve_quartic, VELOCITY_INDEXES

INTEGRATOR = "cvode"
LINSOLVER = "dense"
COORDS = "spherical midplane 0"
SOLUTION_TYPE = "hydrostatic"
SONIC_POINT_TOLERANCE = float_type(0.01)

log = logbook.Logger(__name__)


def X_dash_func(
    *, η_O, η_A, η_H, b_θ, b_r, b_φ, deriv_η_O, deriv_η_A, deriv_η_H,
    deriv_b_θ, deriv_b_r, deriv_b_φ
):
    """
    Compute the value of the variable X'
    """
    return (
        (
            deriv_η_O + deriv_η_A * (1 - b_φ ** 2) - 2 * η_A * b_φ * deriv_b_φ
        ) * (η_H * b_θ - η_A * b_r * b_φ)
    ) / (
        (η_O + η_A * (1 - b_φ ** 2)) ** 2
    ) - (
        deriv_η_H * b_θ + η_H * deriv_b_θ - deriv_η_A * b_r * b_φ -
        η_A * deriv_b_r * b_φ - η_A * b_r * deriv_b_φ
    ) / (
        η_O + η_A * (1 - b_φ ** 2)
    )


def X_func(*, η_O, η_A, η_H, b_θ, b_r, b_φ):
    """
    Compute the value of the variable X
    """
    return (η_H * b_θ - η_A * b_r * b_φ) / (η_O + η_A * (1 - b_φ ** 2))


def b_func(*, X, v_r):
    """
    Compute the value of the variable b
    """
    return X * v_r / 2


def c_func(
    *, θ, norm_kepler_sq, a_0, X, v_r, B_φ, B_θ, B_r, ρ, η_A, η_O, η_H, b_φ,
    b_r, b_θ
):
    """
    Compute the value of the variable c
    """
    return v_r ** 2 / 2 + 5 / 2 - norm_kepler_sq + a_0 / ρ * (
        B_φ ** 2 / 4 + X * B_φ * (
            1 / 4 * B_r + B_θ * tan(θ)
        ) - B_θ / (η_O + η_A * (1 - b_φ**2)) * (
            v_r * B_θ - B_φ * (
                η_A * b_φ * (b_r * tan(θ) - b_θ / 4) -
                η_H * (b_r / 4 + b_θ * tan(θ))
            )
        )
    )


def א_1_func(
    *, a_0, θ, v_φ, ρ, B_φ, B_r, B_θ, deriv_B_r, deriv_B_θ, deriv_B_φ, deriv_ρ
):
    """
    Compute the value of the variable א_1
    """
    return 2 * a_0 / (v_φ * ρ) * (
        deriv_B_θ * (deriv_B_φ - B_φ * tan(θ)) - (
            deriv_B_r * B_φ + B_r * deriv_B_φ
        ) / 4 - B_θ * deriv_B_φ * tan(θ) - B_θ * B_φ * sec(θ) ** 2 -
        deriv_ρ / ρ * (
            B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
        )
    )


def א_2_func(
    *, a_0, θ, v_r, ρ, B_φ, B_r, B_θ, deriv_B_r, deriv_B_θ, deriv_B_φ,
    deriv_ρ, b_r, b_θ, b_φ, deriv_b_r, deriv_b_θ, deriv_b_φ, X, X_dash, b, c,
    η_O, η_A, η_H, deriv_η_O, deriv_η_A, deriv_η_H
):
    """
    Compute the value of the variable א_2
    """
    return - v_r * X_dash / 4 + (
        v_r ** 2 * X * X_dash / 16 + a_0 * deriv_ρ / (ρ ** 2) * (
            B_φ ** 2 / 4 + X * B_φ * (
                1 / 4 * B_r + B_θ * tan(θ)
            ) - B_θ / (η_O + η_A * (1 - b_φ**2)) * (
                v_r * B_θ - B_φ * (
                    η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) -
                    η_H * (b_r / 4 + b_θ * tan(θ))
                )
            )
        ) - a_0 / ρ * (
            1 / 2 * deriv_B_φ * B_φ +
            (X_dash * B_φ + X * deriv_B_φ) * (B_r / 4 + B_θ * tan(θ)) +
            X * B_φ * (
                deriv_B_r / 4 + deriv_B_θ * tan(θ) + B_θ * sec(θ)**2
            ) - (
                deriv_B_θ * (η_O + η_A * (1 - b_φ**2)) - B_θ * (
                    deriv_η_O + deriv_η_A * (1 - b_φ**2) -
                    2 * η_A * b_φ * deriv_b_φ
                )
            ) / (
                (η_O + η_A * (1 - b_φ**2)) ** 2
            ) * (
                v_r * B_θ - B_φ * (
                    η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) -
                    η_H * (b_r / 4 + b_θ * tan(θ))
                )
            ) - B_θ / (η_O + η_A * (1 - b_φ**2)) * (
                deriv_B_θ * v_r - deriv_B_φ * (
                    η_A * b_φ * (b_r * tan(θ) - b_θ / 4) -
                    η_H * (b_r / 4 + b_θ * tan(θ))
                ) - B_φ * (
                    (deriv_η_A * b_φ + η_A * deriv_b_φ) * (
                        b_r * tan(θ) - b_θ / 4
                    ) + η_A * b_φ * (
                        deriv_b_r * tan(θ) + b_r * sec(θ)**2 - deriv_b_θ / 4
                    ) - deriv_η_H * (b_r / 4 + b_θ * tan(θ)) - η_H * (
                        deriv_b_r / 4 + deriv_b_θ * tan(θ) + b_θ * sec(θ)**2
                    )
                )
            )
        )
    ) / sqrt(b ** 2 - 4 * c)


def א_3_func(
    *, a_0, v_r, ρ, B_θ, b_φ, X, b, c, η_O, η_A
):
    """
    Compute the value of the variable א_3
    """
    return (
        X / 4 - 1 / sqrt(b ** 2 - 4 * c) * (
            v_r * (X ** 2 / 16 - 1) + 2 * a_0 * B_θ ** 2 / (
                ρ * (η_O + η_A * (1 - b_φ ** 2))
            )
        )
    )


def Z_func(
    *, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_φ, deriv_B_φ, C, b_r, b_θ, b_φ, ℵ_3
):
    """
    Compute the value of the variable Z
    """
    return (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
    ) - B_θ ** 2 * v_φ * (
        C / ℵ_3 + 1
    ) / (
        B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
    )


def dderiv_B_φ_func(
    *, a_0, norm_kepler_sq, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_φ, ρ,
    deriv_B_r, deriv_B_θ, deriv_B_φ, deriv_η_O, deriv_η_A, deriv_η_H,
    deriv_ρ
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

    X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    X_dash = X_dash_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    b = b_func(X=X, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ,
        B_θ=B_θ, B_r=B_r, ρ=ρ, η_A=η_A, η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r,
        b_θ=b_θ
    )
    if (b**2 - 4 * c) < 0:
        log.warning("b = {}".format(b))
        log.warning("c = {}".format(c))
        log.error("Discriminant less than 0, = {}; θ = {}".format(
            b**2 - 4 * c, degrees(θ)
        ))
        return None
    else:
        log.debug("b = {}".format(b))
        log.debug("c = {}".format(c))
        log.debug("Discriminant not less than 0, = {}".format(b**2 - 4 * c))

    א_1 = א_1_func(
        a_0=a_0, θ=θ, v_φ=v_φ, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ
    )

    א_2 = א_2_func(
        a_0=a_0, θ=θ, v_r=v_r, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, deriv_b_r=deriv_b_r,
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, X=X, X_dash=X_dash, b=b, c=c,
        η_O=η_O, η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )

    א_3 = א_3_func(
        a_0=a_0, v_r=v_r, ρ=ρ, B_θ=B_θ, b_φ=b_φ, X=X, b=b, c=c, η_O=η_O,
        η_A=η_A
    )

    Z = Z_func(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_φ=v_φ,
        deriv_B_φ=deriv_B_φ, C=C, b_r=b_r, b_θ=b_θ, b_φ=b_φ, ℵ_3=ℵ_3
    )

    log.info("א_1 = {}".format(א_1))
    log.info("א_2 = {}".format(א_2))
    log.info("א_3 = {}".format(א_3))
    log.info("Z = {}".format(Z))

    return (
        3 * v_φ * B_r / 4 - B_θ * ℵ_2 - v_φ * deriv_B_θ +
        3 / 4 * (B_θ / 4 + deriv_B_r) * (
            η_H * b_r - η_A * b_θ * b_φ
        ) + A * v_r * B_θ + C * v_r * deriv_B_θ + deriv_B_φ * (
            η_O * tan(θ) - deriv_η_O + η_A * (
                tan(θ) * (1 - b_r ** 2) + b_r / 4 * (
                    b_φ + 3 * b_θ / 4
                ) - C * b_φ * (
                    b_r * tan(θ) - b_θ / 4
                ) + 2 * b_r * deriv_b_r - A * b_r * b_φ -
                C * deriv_b_r * b_φ - C * b_r * deriv_b_φ
            ) - deriv_η_H * C * b_θ - deriv_η_A * (
                1 - b_r ** 2 - C * b_r * b_φ
            ) + η_H * (
                C * (b_r / 4 + b_θ * tan(θ) - deriv_b_θ) + b_φ / 2 - A * b_θ
            )
        ) + B_φ * (
            deriv_η_O * tan(θ) + η_O * (sec(θ) ** 2 - 3 / 16) - 3 * v_r / 4 +
            deriv_η_H * (
                C * (b_r / 4 + b_θ * tan(θ)) - b_φ / 4
            ) + η_H * (
                A * (
                    b_r / 4 + b_θ * tan(θ)
                ) - deriv_b_φ / 4 + C * b_φ * (
                    deriv_b_r / 4 + deriv_b_θ * tan(θ) + b_θ * sec(θ) ** 2
                ) - 3 * b_φ * tan(θ) / 4
            ) + η_A * (
                sec(θ) ** 2 * (1 - b_r ** 2) + 2 * tan(θ) * b_r * deriv_b_r +
                deriv_b_r * b_θ / 4 + b_r * deriv_b_θ / 4 - A * b_φ * (
                    b_r * tan(θ) - b_θ / 4
                ) - C * deriv_b_φ * (b_r * tan(θ) - b_θ / 4) - C * b_φ * (
                    deriv_b_r * tan(θ) + b_r * sec(θ) ** 2 - deriv_b_θ / 4
                ) - 3 / 4 * (
                    (1 - b_θ ** 2) / 4 + tan(θ) * b_r * b_θ
                )
            ) + deriv_η_A * (
                tan(θ) * (1 - b_r ** 2) + b_r * b_θ / 4 - C * b_φ * (
                    b_r * tan(θ) - b_θ / 4
                )
            )
        ) + B_θ * (C / ℵ_3 + 1) * (
            ℵ_1 / (
                2 * a_0 / (v_φ ** 2 * ρ) * (
                    B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
                )
            ) - ℵ_2
        )
    ) / Z


def deriv_v_φ_func(
    *, a_0, norm_kepler_sq, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, ρ,
    deriv_B_r, deriv_B_θ, deriv_B_φ, deriv_η_O, deriv_η_A, deriv_η_H,
    deriv_ρ, deriv_v_r
):
    """
    Compute the derivative of v_φ
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        deriv_B_θ=deriv_B_θ
    )

    X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    X_dash = X_dash_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    b = b_func(X=X, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ,
        B_θ=B_θ, B_r=B_r, ρ=ρ, η_A=η_A, η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r,
        b_θ=b_θ
    )

    א_2 = א_2_func(
        a_0=a_0, θ=θ, v_r=v_r, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, deriv_b_r=deriv_b_r,
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, X=X, X_dash=X_dash, b=b, c=c,
        η_O=η_O, η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )

    א_3 = א_3_func(
        a_0=a_0, v_r=v_r, ρ=ρ, B_θ=B_θ, b_φ=b_φ, X=X, b=b, c=c, η_O=η_O,
        η_A=η_A
    )

    return ℵ_2 - deriv_v_r * ℵ_3


def deriv_v_r_func(
    *, a_0, norm_kepler_sq, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_φ, ρ,
    deriv_B_r, deriv_B_θ, deriv_B_φ, deriv_η_O, deriv_η_A, deriv_η_H,
    dderiv_B_φ, deriv_ρ
):
    """
    Compute the derivative of v_r
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
        B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
        deriv_B_θ=deriv_B_θ
    )

    X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    X_dash = X_dash_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    b = b_func(X=X, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ,
        B_θ=B_θ, B_r=B_r, ρ=ρ, η_A=η_A, η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r,
        b_θ=b_θ
    )

    א_1 = א_1_func(
        a_0=a_0, θ=θ, v_φ=v_φ, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ
    )

    א_2 = א_2_func(
        a_0=a_0, θ=θ, v_r=v_r, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, deriv_b_r=deriv_b_r,
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, X=X, X_dash=X_dash, b=b, c=c,
        η_O=η_O, η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )

    א_3 = א_3_func(
        a_0=a_0, v_r=v_r, ρ=ρ, B_θ=B_θ, b_φ=b_φ, X=X, b=b, c=c, η_O=η_O,
        η_A=η_A
    )

    return v_φ / ℵ_3 * (
        B_θ * dderiv_B_φ + ℵ_1 * v_φ * ρ / (2 * a_0)
    ) / (
        B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
    ) - ℵ_2 / ℵ_3


def v_r_func(
    *, a_0, norm_kepler_sq, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, ρ, deriv_B_φ
):
    """
    Compute v_r
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    a = 1/2

    b = - B_θ ** 2 * a_0 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))

    c = 5 / 2 - norm_kepler_sq + a_0 / ρ * (
        B_φ ** 2 / 4 + B_θ * deriv_B_φ * X + B_θ * B_φ / (
            η_O + η_A * (1 - b_φ**2)
        ) * (
            η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) - η_H * (
                b_r / 4 + b_θ * tan(θ)
            )
        )
    )

    e = (
        2 * a_0 / ρ * (B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ))
    ) ** 2
    v_r_list = solve_quartic(a=a, b=b, c=c, d=0, e=e, use_np_roots=True)

    return v_r_list[isreal(v_r_list)][1]


def v_φ_func(
    *, a_0, norm_kepler_sq, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, ρ,
):
    """
    Compute v_φ
    """
    B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
    with errstate(invalid="ignore"):
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

    X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    b = b_func(X=X, v_r=v_r)

    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ,
        B_θ=B_θ, B_r=B_r, ρ=ρ, η_A=η_A, η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r,
        b_θ=b_θ
    )

    return - b + sqrt(b**2 - 4 * c)


def ode_system(
    *, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1),
    η_derivs=True, η_derivs_func=None, store_internal=True, no_v_deriv=True
):
    """
    Set up the system we are solving for.
    """
    # pylint: disable=too-many-statements
    if η_derivs:
        η_O_scale = init_con[ODEIndex.η_O] / sqrt(init_con[ODEIndex.ρ])
        η_A_scale = init_con[ODEIndex.η_A] / sqrt(init_con[ODEIndex.ρ])
        η_H_scale = init_con[ODEIndex.η_H] / sqrt(init_con[ODEIndex.ρ])
    else:
        η_O_scale = 0
        η_A_scale = 0
        η_H_scale = 0
        η_derivs_func = None

    if store_internal:
        internal_data = InternalData()
        derivs_list = internal_data.derivs
        params_list = internal_data.params
        angles_list = internal_data.angles
        problems = internal_data.problems
    else:
        internal_data = None

    def rhs_equation(x, params, derivs):
        """
        Compute the ODEs
        """
        # pylint: disable=too-many-statements
        θ = scaled_to_rad(x, θ_scale)
        B_r = params[ODEIndex.B_r]
        B_φ = params[ODEIndex.B_φ]
        B_θ = params[ODEIndex.B_θ]
        v_r = params[ODEIndex.v_r]
        v_φ = params[ODEIndex.v_φ]
        ρ = params[ODEIndex.ρ]
        B_φ_prime = params[ODEIndex.B_φ_prime]
        η_O = params[ODEIndex.η_O]
        η_A = params[ODEIndex.η_A]
        η_H = params[ODEIndex.η_H]

        # check sanity of input values
        if ρ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative density")
            return 1

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

        with errstate(invalid="ignore"):
            norm_B_r, norm_B_φ, norm_B_θ = (
                B_r/B_mag, B_φ/B_mag, B_θ/B_mag
            )

        deriv_B_φ = B_φ_prime
        deriv_B_θ = B_θ * tan(θ) - 3/4 * B_r

        if no_v_deriv:
            v_r = v_r_func(
                a_0=a_0, norm_kepler_sq=norm_kepler_sq, B_r=B_r, B_φ=B_φ,
                B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, ρ=ρ,
                deriv_B_φ=deriv_B_φ
            )
            v_φ = v_φ_func(
                a_0=a_0, norm_kepler_sq=norm_kepler_sq, B_r=B_r, B_φ=B_φ,
                B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r, ρ=ρ,
            )
            params[ODEIndex.v_r] = v_r
            params[ODEIndex.v_φ] = v_φ

        deriv_B_r = (
            (
                deriv_B_φ * (
                    η_H * norm_B_θ -
                    η_A * norm_B_r * norm_B_φ
                ) + B_φ * (
                    η_A * norm_B_φ * (
                        norm_B_r * tan(θ) -
                        norm_B_θ / 4
                    ) - η_H * (
                        norm_B_r / 4 +
                        norm_B_θ * tan(θ)
                    )
                ) - v_r * B_θ
            ) / (
                η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
            ) - B_θ / 4
        )

        deriv_ρ = - ρ * v_φ ** 2 * tan(θ) - a_0 * (
            B_θ * B_r / 4 + B_r * deriv_B_r + B_φ * deriv_B_φ -
            B_φ ** 2 * tan(θ)
        )

        if η_derivs_func is not None:
            deriv_η_O, deriv_η_A, deriv_η_H = η_derivs_func()
        elif η_derivs_func is None and η_derivs:
            deriv_ρ_scale = deriv_ρ / sqrt(ρ) / 2
            deriv_η_O = deriv_ρ_scale * η_O_scale
            deriv_η_A = deriv_ρ_scale * η_A_scale
            deriv_η_H = deriv_ρ_scale * η_H_scale
        else:
            deriv_η_O = 0
            deriv_η_A = 0
            deriv_η_H = 0

        dderiv_B_φ = dderiv_B_φ_func(
            a_0=a_0, norm_kepler_sq=norm_kepler_sq, B_r=B_r, B_φ=B_φ, B_θ=B_θ,
            η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r, v_φ=v_φ, ρ=ρ,
            deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
            deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
            deriv_ρ=deriv_ρ,
        )
        if dderiv_B_φ is None:
            # Discriminant is less than 0
            return 1
        deriv_v_r = deriv_v_r_func(
            a_0=a_0, norm_kepler_sq=norm_kepler_sq, B_r=B_r, B_φ=B_φ, B_θ=B_θ,
            η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r, v_φ=v_φ, ρ=ρ,
            deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
            deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
            dderiv_B_φ=dderiv_B_φ, deriv_ρ=deriv_ρ,
        )
        deriv_v_φ = deriv_v_φ_func(
            a_0=a_0, norm_kepler_sq=norm_kepler_sq, B_r=B_r, B_φ=B_φ, B_θ=B_θ,
            η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r, ρ=ρ, deriv_B_r=deriv_B_r,
            deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ, deriv_η_O=deriv_η_O,
            deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H, deriv_ρ=deriv_ρ,
            deriv_v_r=deriv_v_r
        )

        derivs[ODEIndex.B_r] = deriv_B_r
        derivs[ODEIndex.B_φ] = deriv_B_φ
        derivs[ODEIndex.B_θ] = deriv_B_θ
        derivs[ODEIndex.v_r] = deriv_v_r
        derivs[ODEIndex.v_φ] = deriv_v_φ
        derivs[ODEIndex.ρ] = deriv_ρ
        derivs[ODEIndex.B_φ_prime] = dderiv_B_φ
        derivs[ODEIndex.η_O] = deriv_η_O
        derivs[ODEIndex.η_A] = deriv_η_A
        derivs[ODEIndex.η_H] = deriv_η_H

        derivs[ODEIndex.v_θ] = 0
        if no_v_deriv:
            derivs[VELOCITY_INDEXES] = 0
        if __debug__:
            log.debug("θ: {}, {}", θ, degrees(θ))

        if store_internal:
            params_list.append(copy(params))
            derivs_list.append(copy(derivs))
            angles_list.append(θ)

            if len(params_list) != len(angles_list):
                log.error(
                    "Internal data not consistent, "
                    "params is {}, angles is {}".format(
                        len(params_list), len(angles_list)
                    )
                )

        return 0
    return rhs_equation, internal_data


def hydrostatic_solution(
    *, angles, initial_conditions, a_0, norm_kepler_sq,
    relative_tolerance=float_type(1e-6), absolute_tolerance=float_type(1e-10),
    max_steps=500, onroot_func=None, tstop=None, ontstop_func=None,
    η_derivs=True, store_internal=True, root_func=None, root_func_args=None,
    θ_scale=float_type(1), no_v_deriv=True
):
    """
    Find solution
    """
    extra_args = {}
    if root_func is not None:
        extra_args["rootfn"] = root_func
        if root_func_args is not None:
            extra_args["nr_rootfns"] = root_func_args
        else:
            raise SolverError("Need to specify size of root array")

    system, internal_data = ode_system(
        a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=initial_conditions, η_derivs=η_derivs,
        store_internal=store_internal, θ_scale=θ_scale, no_v_deriv=no_v_deriv,
    )

    ode_solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=error_handler,
        onroot=onroot_func,
        tstop=rad_to_scaled(tstop, θ_scale),
        ontstop=ontstop_func,
        bdf_stability_detection=True,
        **extra_args
    )

    try:
        soln = ode_solver.solve(angles, initial_conditions)
    except CVODESolveFailed as e:
        soln = e.soln
        log.warn(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                degrees(scaled_to_rad(soln.errors.t, θ_scale)),
                soln.flag, soln.message
            )
        )
        if soln.flag == StatusEnum.TOO_CLOSE:
            raise e
    except CVODESolveFoundRoot as e:
        soln = e.soln
        log.notice("Found root at {}".format(
            degrees(scaled_to_rad(soln.roots.t, θ_scale))
        ))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        for tstop_scaled in soln.tstop.t:
            log.notice("Stopped at {}".format(
                degrees(scaled_to_rad(tstop_scaled, θ_scale))
            ))

    return soln, internal_data


def solution(
    soln_input, initial_conditions, *, onroot_func=None, tstop=None,
    ontstop_func=None, store_internal=True, root_func=None,
    root_func_args=None, θ_scale=float_type(1), no_v_deriv=False
):
    """
    Find solution
    """
    angles = initial_conditions.angles
    init_con = initial_conditions.init_con
    a_0 = initial_conditions.a_0
    norm_kepler_sq = initial_conditions.norm_kepler_sq
    absolute_tolerance = soln_input.absolute_tolerance
    relative_tolerance = soln_input.relative_tolerance
    max_steps = soln_input.max_steps
    η_derivs = soln_input.η_derivs
    if no_v_deriv:
        if float_type not in (float, float64):
            log.warn(
                "Float type used unlikely to work with root solver, setting "
                "no_v_deriv to False"
            )
            no_v_deriv = False
        else:
            init_con[VELOCITY_INDEXES] = 0

    soln, internal_data = hydrostatic_solution(
        angles=angles, initial_conditions=init_con, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal, root_func=root_func,
        root_func_args=root_func_args, θ_scale=θ_scale, no_v_deriv=no_v_deriv
    )

    return Solution(
        solution_input=soln_input, initial_conditions=initial_conditions,
        flag=soln.flag, coordinate_system=COORDS, internal_data=internal_data,
        angles=scaled_to_rad(soln.values.t, θ_scale), solution=soln.values.y,
        t_roots=(
            scaled_to_rad(soln.roots.t, θ_scale)
            if soln.roots.t is not None else None
        ), y_roots=soln.roots.y, sonic_point=None,
        sonic_point_values=None,
    )


def solver(inp, run, *, store_internal=True):
    """
    hydrostatic solver
    """
    cons = define_conditions(inp)
    cons.γ = 0
    hydro_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal
    )
    run.solutions["0"] = hydro_solution
    run.final_solution = run.solutions["0"]
