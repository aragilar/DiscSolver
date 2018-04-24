# -*- coding: utf-8 -*-
"""
The system of odes
"""

from collections import namedtuple

import logbook

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT
from numpy import sqrt, tan, copy, errstate, degrees

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
from ..utils import ODEIndex, sec

INTEGRATOR = "cvode"
LINSOLVER = "dense"
COORDS = "spherical midplane 0"
SOLUTION_TYPE = "hydrostatic"
SONIC_POINT_TOLERANCE = float_type(0.01)

log = logbook.Logger(__name__)

TaylorSolution = namedtuple(
    "TaylorSolution", [
        "angles", "params", "new_angles", "internal_data",
        "new_initial_conditions", "angle_stopped",
    ]
)


def b_func(*, C, v_r):
    """
    Compute the value of the variable b
    """
    return C * v_r / 2


def c_func(
    *, θ, norm_kepler_sq, a_0, C, v_r, B_φ, B_θ, B_r, ρ, η_A, η_O, η_H, b_φ,
    b_r, b_θ
):
    """
    Compute the value of the variable c
    """
    return v_r ** 2 / 2 + 5 / 2 - norm_kepler_sq + a_0 / ρ * (
        B_φ ** 2 / 4 + C * B_φ * (
            1 / 4 * B_r + B_θ * tan(θ)
        ) - B_θ / (η_O + η_A * (1 - b_φ**2)) * (
            v_r * B_θ - B_φ * (
                η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) -
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
    deriv_ρ, b_r, b_θ, b_φ, deriv_b_r, deriv_b_θ, deriv_b_φ, C, A, b, c, η_O,
    η_A, η_H, deriv_η_O, deriv_η_A, deriv_η_H
):
    """
    Compute the value of the variable א_2
    """
    return - v_r * A / 4 + (
        v_r ** 2 * C * A / 16 + a_0 * deriv_ρ / (ρ ** 2) * (
            B_φ ** 2 / 4 + C * B_φ * (
                1 / 4 * B_r + B_θ * tan(θ)
            ) - B_θ / (η_O + η_A * (1 - b_φ**2)) * (
                v_r * B_θ - B_φ * (
                    η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) -
                    η_H * (b_r / 4 + b_θ * tan(θ))
                )
            )
        ) - a_0 / ρ * (
            1 / 2 * deriv_B_φ * B_φ +
            (A * B_φ + C * deriv_B_φ) * (B_r / 4 + B_θ * tan(θ)) +
            C * B_φ * (
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
                    η_A * b_φ * (b_θ / 4 + b_r * tan(θ)) -
                    η_H * (b_r / 4 + b_θ * tan(θ))
                ) - B_φ * (
                    (deriv_η_A * b_φ + η_A * deriv_b_φ) * (
                        b_θ / 4 + b_r * tan(θ)
                    ) + η_A * b_φ * (
                        deriv_b_θ / 4 + deriv_b_r * tan(θ) + b_r * sec(θ)**2
                    ) - deriv_η_H * (b_r / 4 + b_θ * tan(θ)) - η_H * (
                        deriv_b_r / 4 + deriv_b_θ * tan(θ) + b_θ * sec(θ)**2
                    )
                )
            )
        )
    ) / sqrt(b ** 2 - 4 * c)


def Z_func(
    *, a_0, B_r, B_φ, B_θ, η_O, η_H, η_A, θ, v_r, v_φ, ρ,
    deriv_B_φ, C, b, c, b_r, b_θ, b_φ
):
    """
    Compute the value of the variable Z
    """
    return (
        η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
    ) + 8 * B_θ ** 2 * a_0 / (3 * v_φ * ρ) * (
        C - 1 / sqrt(b ** 2 - 4 * c) * (
            v_r * (C ** 2 / 16 - 1) +
            2 * a_0 * B_θ ** 2 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))
        )
    ) / (
        2 * a_0 / (v_φ ** 2 * ρ) * (
            B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
        ) * (
            C / 4 - 1 / sqrt(b ** 2 - 4 * c) * (
                v_r * (C ** 2 / 16 - 1) +
                2 * a_0 * B_θ ** 2 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))
            )
        )
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

    b = b_func(C=C, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, C=C, v_r=v_r, B_φ=B_φ,
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

    Z = Z_func(
        a_0=a_0, B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ,
        v_r=v_r, v_φ=v_φ, ρ=ρ, deriv_B_φ=deriv_B_φ, C=C, b=b, c=c, b_r=b_r,
        b_θ=b_θ, b_φ=b_φ
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
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, C=C, A=A, b=b, c=c, η_O=η_O,
        η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )
    log.info("Z = {}".format(Z))
    log.info("א_1 = {}".format(א_1))
    log.info("א_2 = {}".format(א_2))

    return (
        v_φ * B_r - 4 / 3 * (
            ℵ_2 * B_θ + v_φ * deriv_B_θ
        ) + (B_θ / 4 + deriv_B_r) * (
            η_H * b_r - η_A * b_θ * b_φ
        ) + deriv_B_φ * (
            η_O * tan(θ) - deriv_η_O - deriv_η_H * C * b_θ -
            deriv_η_A * (
                1 - b_r ** 2 - C * b_r * b_φ
            ) + η_H * (
                3 * b_φ / 4 + C * (
                    b_r / 4 + b_θ * tan(θ) + deriv_b_θ
                ) - A * b_θ
            ) + η_A * (
                tan(θ) * (1 - b_r ** 2) + b_r * b_φ / 4 - C * b_φ * (
                    b_θ / 4 + b_r * tan(θ)
                ) + 2 * b_r * deriv_b_r - A * b_r * b_φ -
                C * deriv_b_r * b_φ - C * b_r * deriv_b_φ + b_r * b_θ
            )
        ) + A * v_r * B_θ + C * v_r * deriv_B_θ + B_φ * (
            deriv_η_O * tan(θ) + η_O * (sec(θ)**2 - 1 / 4) -
            v_r + η_H * (
                A * (
                    b_r / 4 + b_θ * tan(θ)
                ) + C * b_φ * (
                    deriv_b_r / 4 + deriv_b_θ * tan(θ) + b_θ * sec(θ)**2
                ) - b_φ * tan(θ) - deriv_b_φ / 4
            ) + deriv_η_H * (
                C * (b_r / 4 + b_θ * tan(θ)) - b_φ / 4
            ) + η_A * (
                sec(θ)**2 * (1 - b_r ** 2) + 2 * tan(θ) * b_r * deriv_b_r +
                deriv_b_r * b_θ / 4 + b_r * deriv_b_θ / 4 -
                A * b_φ * (b_θ / 4 + b_r * tan(θ)) - C * deriv_b_φ * (
                    b_θ / 4 + b_r * tan(θ)
                ) - C * b_φ * (
                    deriv_b_θ / 4 + deriv_b_r * tan(θ) + b_r * sec(θ)**2
                ) - (1 - b_θ ** 2) / 4 - tan(θ) * b_r * b_θ
            ) + deriv_η_A * (
                tan(θ) * (1 - b_r ** 2) + b_r * b_θ / 4 - C * b_φ * (
                    b_θ / 4 + b_r * tan(θ)
                )
            )
        ) + 4 * B_θ / 3 * (
            C - 1 / sqrt(b ** 2 - 4 * c) * (
                v_r * (C ** 2 / 16 - 1) +
                2 * a_0 * B_θ ** 2 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))
            )
        ) * (
            ℵ_1 - 2 * a_0 * ℵ_2 / (v_φ ** 2 * ρ) * (
                B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
            )
        ) / (
            2 * a_0 / (v_φ ** 2 * ρ) * (
                B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
            ) * (
                C / 4 - 1 / sqrt(b ** 2 - 4 * c) * (
                    v_r * (C ** 2 / 16 - 1) +
                    2 * a_0 * B_θ ** 2 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))
                )
            )
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

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    A = A_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    b = b_func(C=C, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, C=C, v_r=v_r, B_φ=B_φ,
        B_θ=B_θ, B_r=B_r, ρ=ρ, η_A=η_A, η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r,
        b_θ=b_θ
    )

    א_2 = א_2_func(
        a_0=a_0, θ=θ, v_r=v_r, ρ=ρ, B_φ=B_φ, B_r=B_r, B_θ=B_θ,
        deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
        deriv_ρ=deriv_ρ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, deriv_b_r=deriv_b_r,
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, C=C, A=A, b=b, c=c, η_O=η_O,
        η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )

    return ℵ_2 - deriv_v_r * (
        C / 4 - 1 / sqrt(b ** 2 - 4 * c) * (
            v_r * (C ** 2 / 16 - 1) + 2 * a_0 * B_θ ** 2 / (
                ρ * (η_O + η_A * (1 - b_φ ** 2))
            )
        )
    )


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

    C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

    A = A_func(
        η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
        deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
        deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
    )

    b = b_func(C=C, v_r=v_r)
    c = c_func(
        θ=θ, norm_kepler_sq=norm_kepler_sq, a_0=a_0, C=C, v_r=v_r, B_φ=B_φ,
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
        deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, C=C, A=A, b=b, c=c, η_O=η_O,
        η_A=η_A, η_H=η_H, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
        deriv_η_H=deriv_η_H
    )

    return (
        2 * B_θ * dderiv_B_φ * a_0 / (v_φ * ρ) -
        2 * a_0 * ℵ_2 / (v_φ ** 2 * ρ) * (
            B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
        ) + ℵ_1
    ) / (
        2 * a_0 / (v_φ ** 2 * ρ) * (
            B_θ * deriv_B_φ - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
        ) * (
            C / 4 - 1 / sqrt(b ** 2 - 4 * c) * (
                v_r * (C ** 2 / 16 - 1) +
                2 * a_0 * B_θ ** 2 / (ρ * (η_O + η_A * (1 - b_φ ** 2)))
            )
        )
    )


def ode_system(
    *, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1),
    η_derivs=True, η_derivs_func=None, store_internal=True
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
        deriv_B_r = (
            (
                deriv_B_φ * (
                    η_H * norm_B_θ -
                    η_A * norm_B_r * norm_B_φ
                ) + B_φ * (
                    η_A * norm_B_φ * (
                        norm_B_θ / 4 +
                        norm_B_r * tan(θ)
                    ) - η_H * (
                        norm_B_r / 4 +
                        norm_B_θ * tan(θ)
                    )
                ) - v_r * B_θ
            ) / (
                η_O + η_A * (1 - norm_B_φ) * (1 + norm_B_φ)
            ) - B_θ / 4
        )

        deriv_B_θ = B_θ * tan(θ) - 3/4 * B_r

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
    θ_scale=float_type(1)
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
        store_internal=store_internal, θ_scale=θ_scale,
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
    root_func_args=None, θ_scale=float_type(1)
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

    soln, internal_data = hydrostatic_solution(
        angles=angles, initial_conditions=init_con, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal, root_func=root_func,
        root_func_args=root_func_args, θ_scale=θ_scale,
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
    hydro_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal
    )
    run.solutions["0"] = hydro_solution
    run.final_solution = run.solutions["0"]
