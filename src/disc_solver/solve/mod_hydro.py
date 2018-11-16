# -*- coding: utf-8 -*-
"""
The system of odes
"""

import logbook

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT
import numpy as np
from numpy import sqrt, tan, copy, errstate, degrees

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from root_solver import CubicTracker

from .config import (
    B_φ_prime_boundary_func, E_r_boundary_func,
)
from .deriv_funcs import (
    B_unit_derivs, A_func, C_func, deriv_η_skw_func, deriv_E_r_func,
)
from .hydrostatic import (
    X_dash_func, X_func, Z_1_func, Z_4_func, Z_5_func, dderiv_B_φ_func
)
from .utils import (
    error_handler, rad_to_scaled, scaled_to_rad,
    SolverError,
)

from ..float_handling import float_type, FLOAT_TYPE
from ..file_format import InternalData, Solution, InitialConditions
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "dense"
COORDS = "spherical midplane 0"
SOLUTION_TYPE = "mod_hydro"
SONIC_POINT_TOLERANCE = float_type(0.01)

log = logbook.Logger(__name__)


def mod_hydro_define_conditions(inp, run):
    """
    Compute initial conditions based on input
    """
    ρ = float_type(1)
    B_θ = float_type(1)
    v_θ = float_type(0)
    B_r = float_type(0)
    B_φ = float_type(0)

    v_r = - inp.v_rin_on_c_s
    a_0 = inp.v_a_on_c_s ** 2
    norm_kepler_sq = 1 / inp.c_s_on_v_k ** 2

    η_O = inp.η_O
    η_A = inp.η_A
    η_H = inp.η_H

    η_P = η_O + η_A
    η_perp_sq = η_P ** 2 + η_H ** 2

    v_φ = (2 * norm_kepler_sq - 5 / 2 + v_r * (a_0 / η_P - v_r / 2)) / (
        2 * sqrt(norm_kepler_sq) - v_r * η_H / (2 * η_P)
    )

    init_con = np.zeros(11, dtype=FLOAT_TYPE)

    init_con[ODEIndex.B_r] = B_r
    init_con[ODEIndex.B_φ] = B_φ
    init_con[ODEIndex.B_θ] = B_θ
    init_con[ODEIndex.v_r] = v_r
    init_con[ODEIndex.v_φ] = v_φ
    init_con[ODEIndex.v_θ] = v_θ
    init_con[ODEIndex.ρ] = ρ
    init_con[ODEIndex.η_O] = η_O
    init_con[ODEIndex.η_A] = η_A
    init_con[ODEIndex.η_H] = η_H

    if run.use_E_r:
        E_r = E_r_boundary_func(
            v_r=v_r, v_φ=v_φ, η_P=η_P, η_H=η_H, η_perp_sq=η_perp_sq, a_0=a_0
        )
        init_con[ODEIndex.E_r] = E_r
    else:
        B_φ_prime = B_φ_prime_boundary_func(v_r=v_r, v_φ=v_φ, a_0=a_0)
        init_con[ODEIndex.B_φ_prime] = B_φ_prime

    angles = np.radians(np.linspace(inp.start, inp.stop, inp.num_angles))
    if np.any(np.isnan(init_con)):
        raise SolverError("Input implies NaN")

    return InitialConditions(
        norm_kepler_sq=norm_kepler_sq, a_0=a_0, init_con=init_con,
        angles=angles, γ=0
    )


def Z_2_func(
    *, θ, a_0, X, v_r, B_φ, B_θ, ρ, η_A, η_O, η_H, b_φ, b_r, b_θ, X_dash,
    deriv_B_φ, deriv_B_θ, deriv_b_θ, deriv_b_φ, deriv_b_r, deriv_η_O,
    deriv_η_A, deriv_η_H, deriv_ρ, norm_kepler
):
    """
    Compute the value of the variable Z_2
    """
    return - a_0 / ρ * (
        deriv_B_φ * B_φ / 2 - deriv_B_θ * deriv_B_φ * X -
        B_θ * deriv_B_φ * X_dash - deriv_ρ / ρ * (
            B_φ ** 2 / 4 - X * deriv_B_φ * B_θ -
            B_θ / (
                η_O + η_A * (1 - b_φ ** 2)
            ) * (
                v_r * B_θ - B_φ * (
                    η_A * b_φ * (
                        b_r * tan(θ) - b_θ / 4
                    ) + η_H * (
                        b_r / 4 + b_θ * tan(θ)
                    )
                )
            )
        ) - B_θ / (
            η_O + η_A * (1 - b_φ ** 2)
        ) * (
            v_r * deriv_B_θ - deriv_B_φ * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ / 4
                ) + η_H * (b_r / 4 + b_θ * tan(θ))
            ) - B_φ * (
                deriv_η_A * b_φ * (b_r * tan(θ) - b_θ / 4) +
                η_A * deriv_b_φ * (b_r * tan(θ) - b_θ / 4) + η_A * b_φ * (
                    deriv_b_r * tan(θ) + b_r * (tan(θ) ** 2 + 1) -
                    deriv_b_θ / 4
                ) + deriv_η_H * (b_r / 4 + b_θ * tan(θ)) + η_H * (
                    deriv_b_r / 4 + deriv_b_θ * tan(θ) + b_θ * (
                        tan(θ) ** 2 + 1
                    )
                )
            )
        ) - (
            deriv_B_θ / (
                η_O + η_A * (1 - b_φ ** 2)
            ) - (
                B_θ * (
                    deriv_η_O + deriv_η_A * (1 - b_φ ** 2) -
                    2 * η_A * deriv_b_φ * b_φ
                )
            ) / (
                (η_O + η_A * (1 - b_φ ** 2)) ** 2
            )
        ) * (
            v_r * B_θ - B_φ * (
                η_A * b_φ * (b_r * tan(θ) - b_θ / 4) +
                η_H * (b_r / 4 + b_θ * tan(θ))
            )
        )
    ) / (2 * norm_kepler)


def Z_3_func(*, v_r, B_θ, norm_kepler, η_O, η_A, b_φ):
    """
    Compute the value of the variable Z_3
    """
    return B_θ ** 2 / (η_O + η_A * (1 - b_φ ** 2)) - v_r / (2 * norm_kepler)


def Z_6_func(*, a_0, X, v_r, B_θ, ρ, Z_3, Z_5, norm_kepler, C, v_φ, Z_4):
    """
    Compute the value of the variable Z_6
    """
    return Z_5 + B_θ ** 2 * (
        X / (2 * norm_kepler) + a_0 * (C + Z_3) * (
            2 * v_φ - Z_4 * X / norm_kepler
        ) / (v_φ ** 2 * ρ - Z_3 * Z_4 * a_0 * v_r)
    )


def Z_7_func(*, a_0, X, v_r, B_θ, ρ, Z_5, norm_kepler, C, v_φ, Z_4):
    """
    Compute the value of the variable Z_7
    """
    return Z_5 + B_θ ** 2 * (
        X / norm_kepler + a_0 * (C - v_r / (2 * norm_kepler)) * (
            2 * norm_kepler * v_φ - Z_4 * X
        ) / (v_φ ** 2 * ρ * norm_kepler - Z_4 * a_0 * v_r)
    )


def Z_8_func(*, θ, b_r, b_θ, b_φ, B_φ, η_O, η_A, η_H, E_r, Z_5, C):
    """
    Compute the value of the variable Z_8
    """
    return (
        - E_r + B_φ * (
            tan(θ) * (η_O + η_A * (1 - b_r ** 2)) + (
                η_H * b_φ + η_A * b_r * b_θ
            ) / 4 - C * (
                η_A * b_φ * (
                    b_r * tan(θ) - b_θ / 4
                ) + η_H * (
                    b_r / 4 + b_θ * tan(θ)
                )
            )
        )
    ) / Z_5


def deriv_B_φ_func(*, Z_5, Z_8, v_r, v_φ, B_θ, C):
    """
    Compute the derivative of B_φ
    """
    return - B_θ / Z_5 * (v_φ + C * v_r) + Z_8


def v_φ_func(*, θ, a_0, ρ, Z_5, Z_8, v_r, B_r, B_θ, B_φ, C):
    """
    Compute the value of v_φ
    """
    return (
        B_θ * Z_8 - B_θ ** 2 * C * v_r / Z_5 - B_r * B_φ / 4 -
        B_θ * B_φ * tan(θ)
    ) / (
        B_θ ** 2 / Z_5 + v_r * ρ / (2 * a_0)
    )


def v_r_func_generator(initial_v_r):
    """
    Compute the value of v_r
    """
    cubic_solver = CubicTracker(
        start_root=initial_v_r, is_real=True, dtype=FLOAT_TYPE,
    )

    def v_r_func(
        *, θ, a_0, norm_kepler, ρ, Z_5, Z_8, B_r, B_θ, B_φ, b_r, b_θ, b_φ, η_O,
        η_A, η_H, C, X
    ):
        a_3 = ρ / (8 * norm_kepler * a_0)

        a_2 = B_θ ** 2 / (4 * norm_kepler) * (
            (1 + C * X) / Z_5 - 1 / (η_O + η_A * (1 - b_φ ** 2))
        )

        a_1 = - B_θ ** 2 * C / Z_5 * (
            1 + a_0 * B_θ ** 2 * X / (2 * norm_kepler * ρ * Z_5)
        ) - ρ / (2 * a_0) * (
            norm_kepler - (
                5 / 2 + a_0 / ρ * (
                    B_φ ** 2 / 4 - Z_8 * B_θ * X +
                    B_θ * B_φ / (η_O + η_A * (1 - b_φ ** 2)) * (
                        η_A * b_φ * (b_r * tan(θ) - b_θ / 4) +
                        η_H * (b_r / 4 + b_θ * tan(θ))
                    )
                )
            ) / (2 * norm_kepler)
        ) + B_θ ** 4 * a_0 / (2 * norm_kepler * Z_5 * ρ) * (
            C * X / Z_5 - 1 / (η_O + η_A * (1 - b_φ ** 2))
        )

        a_0 = (
            B_θ * Z_8 - B_r * B_φ / 4 - B_θ * B_φ * tan(θ)
        ) * (
            1 + (a_0 * B_θ ** 2 * X) / (2 * norm_kepler * ρ * Z_5)
        ) - B_θ ** 2 / Z_5 * (
            norm_kepler - (
                5 / 2 + a_0 / ρ * (
                    B_φ ** 2 / 4 - Z_8 * B_θ * X +
                    B_θ * B_φ / (η_O + η_A * (1 - b_φ ** 2)) * (
                        η_A * b_φ * (b_r * tan(θ) - b_θ / 4) +
                        η_H * (b_r / 4 + b_θ * tan(θ))
                    )
                )
            ) / (2 * norm_kepler)
        )

        return cubic_solver.solve_root([a_3, a_2, a_1, a_0])
    return v_r_func


def deriv_v_φ_func(*, Z_2, deriv_v_r, dderiv_B_φ, norm_kepler, v_r, X, B_θ):
    """
    Compute the derivative of v_φ
    """
    return Z_2 - deriv_v_r * v_r / (2 * norm_kepler) + dderiv_B_φ * B_θ * X / (
        2 * norm_kepler
    )


def deriv_v_r_func(
    *, Z_1, Z_2, Z_4, v_φ, ρ, a_0, dderiv_B_φ, B_θ, norm_kepler, X, v_r
):
    """
    Compute the derivative of v_r
    """
    return (
        Z_1 * v_φ ** 2 * ρ * norm_kepler - 2 * Z_2 * Z_4 * a_0 * norm_kepler +
        dderiv_B_φ * a_0 * B_θ * (2 * v_φ * norm_kepler - Z_4 * X)
    ) / (
        v_φ ** 2 * ρ * norm_kepler - Z_4 * a_0 * v_r
    )


def fix_solution(
    soln, *, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1),
    use_E_r=False
):
    """
    Fix solution due to not being able to store non-derivatives
    """
    if not use_E_r:
        return soln

    norm_kepler = sqrt(norm_kepler_sq)
    vals = soln.values.y
    angles = scaled_to_rad(soln.values.t, θ_scale)
    v_r_func = v_r_func_generator(init_con[ODEIndex.v_r])

    def single_solve(params, θ):
        B_r = params[ODEIndex.B_r]
        B_φ = params[ODEIndex.B_φ]
        B_θ = params[ODEIndex.B_θ]
        ρ = params[ODEIndex.ρ]
        η_O = params[ODEIndex.η_O]
        η_A = params[ODEIndex.η_A]
        η_H = params[ODEIndex.η_H]

        E_r = params[ODEIndex.E_r]

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

        with errstate(invalid="ignore"):
            b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

        C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)
        X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)
        Z_5 = Z_5_func(
            η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C
        )
        Z_8 = Z_8_func(
            θ=θ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, B_φ=B_φ, η_O=η_O, η_A=η_A,
            η_H=η_H, E_r=E_r, Z_5=Z_5, C=C,
        )

        v_r = v_r_func(
            θ=θ, a_0=a_0, norm_kepler=norm_kepler, ρ=ρ, Z_5=Z_5, Z_8=Z_8,
            B_r=B_r, B_θ=B_θ, B_φ=B_φ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, η_O=η_O,
            η_A=η_A, η_H=η_H, C=C, X=X,
        )

        v_φ = v_φ_func(
            θ=θ, a_0=a_0, ρ=ρ, Z_5=Z_5, Z_8=Z_8, v_r=v_r, B_r=B_r, B_θ=B_θ,
            B_φ=B_φ, C=C,
        )

        params[ODEIndex.v_r] = v_r
        params[ODEIndex.v_φ] = v_φ

    for θ, params in zip(angles, vals):
        single_solve(θ=θ, params=params)

    return soln


def ode_system(
    *, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1), η_derivs=True,
    store_internal=True, use_E_r=False
):
    """
    Set up the system we are solving for.
    """
    # pylint: disable=too-many-statements
    η_O_0 = init_con[ODEIndex.η_O]
    η_A_0 = init_con[ODEIndex.η_A]
    η_H_0 = init_con[ODEIndex.η_H]

    norm_kepler = sqrt(norm_kepler_sq)

    if store_internal:
        internal_data = InternalData()
        derivs_list = internal_data.derivs
        params_list = internal_data.params
        angles_list = internal_data.angles
        problems = internal_data.problems
    else:
        internal_data = None

    if use_E_r:
        v_r_func = v_r_func_generator(init_con[ODEIndex.v_r])

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
        η_O = params[ODEIndex.η_O]
        η_A = params[ODEIndex.η_A]
        η_H = params[ODEIndex.η_H]
        if use_E_r:
            E_r = params[ODEIndex.E_r]
        else:
            B_φ_prime = params[ODEIndex.B_φ_prime]

        # check sanity of input values
        if ρ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative density")
            return 1

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)

        with errstate(invalid="ignore"):
            b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

        C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)
        X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)
        Z_5 = Z_5_func(
            η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C
        )

        deriv_B_θ = B_θ * tan(θ) - 3/4 * B_r

        if use_E_r:
            Z_8 = Z_8_func(
                θ=θ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, B_φ=B_φ, η_O=η_O, η_A=η_A,
                η_H=η_H, E_r=E_r, Z_5=Z_5, C=C,
            )

            v_r = v_r_func(
                θ=θ, a_0=a_0, norm_kepler=norm_kepler, ρ=ρ, Z_5=Z_5, Z_8=Z_8,
                B_r=B_r, B_θ=B_θ, B_φ=B_φ, b_r=b_r, b_θ=b_θ, b_φ=b_φ, η_O=η_O,
                η_A=η_A, η_H=η_H, C=C, X=X,
            )

            v_φ = v_φ_func(
                θ=θ, a_0=a_0, ρ=ρ, Z_5=Z_5, Z_8=Z_8, v_r=v_r, B_r=B_r, B_θ=B_θ,
                B_φ=B_φ, C=C,
            )

            params[ODEIndex.v_r] = v_r
            params[ODEIndex.v_φ] = v_φ

            deriv_B_φ = deriv_B_φ_func(
                Z_5=Z_5, Z_8=Z_8, v_r=v_r, v_φ=v_φ, B_θ=B_θ, C=C,
            )
        else:
            deriv_B_φ = B_φ_prime

        deriv_B_r = (
            B_φ * (
                η_A * b_φ * (
                    b_r * tan(θ) -
                    b_θ / 4
                ) + η_H * (
                    b_r / 4 +
                    b_θ * tan(θ)
                )
            ) - deriv_B_φ * (
                η_H * b_θ +
                η_A * b_r * b_φ
            ) - v_r * B_θ
        ) / (
            η_O + η_A * (1 - b_φ) * (1 + b_φ)
        ) - B_θ / 4

        deriv_ρ = - ρ * v_φ ** 2 * tan(θ) - a_0 * (
            B_θ * B_r / 4 + B_r * deriv_B_r + B_φ * deriv_B_φ -
            B_φ ** 2 * tan(θ)
        )

        if η_derivs:
            deriv_η_scale = deriv_η_skw_func(
                deriv_ρ=deriv_ρ, deriv_B_θ=deriv_B_θ, ρ=ρ, B_r=B_r, B_φ=B_φ,
                B_θ=B_θ, deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ,
            )
            deriv_η_O = deriv_η_scale * η_O_0
            deriv_η_A = deriv_η_scale * η_A_0
            deriv_η_H = deriv_η_scale * η_H_0
        else:
            deriv_η_O = 0
            deriv_η_A = 0
            deriv_η_H = 0

        if use_E_r:
            deriv_E_r = deriv_E_r_func(
                γ=0, θ=θ, v_r=v_r, v_φ=v_φ, B_r=B_r, B_θ=B_θ, B_φ=B_φ, η_O=η_O,
                η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ,
                deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ
            )
        else:
            deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
                B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r,
                deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ
            )

            A = A_func(
                η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
                deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
                deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
            )
            X_dash = X_dash_func(
                η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
                deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
                deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ
            )

            Z_1 = Z_1_func(
                θ=θ, a_0=a_0, B_φ=B_φ, B_θ=B_θ, B_r=B_r, ρ=ρ,
                deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ,
                v_φ=v_φ, deriv_ρ=deriv_ρ
            )
            Z_2 = Z_2_func(
                θ=θ, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ, B_θ=B_θ, ρ=ρ, η_A=η_A,
                η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r, b_θ=b_θ, X_dash=X_dash,
                deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ, deriv_b_θ=deriv_b_θ,
                deriv_b_φ=deriv_b_φ, deriv_b_r=deriv_b_r, deriv_η_O=deriv_η_O,
                deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H, deriv_ρ=deriv_ρ,
                norm_kepler=norm_kepler,
            )
            Z_3 = Z_3_func(
                v_r=v_r, B_θ=B_θ, norm_kepler=norm_kepler, η_O=η_O, η_A=η_A,
                b_φ=b_φ,
            )
            Z_4 = Z_4_func(B_θ=B_θ, B_r=B_r, B_φ=B_φ, deriv_B_φ=deriv_B_φ, θ=θ)
            Z_6 = Z_6_func(
                a_0=a_0, X=X, v_r=v_r, B_θ=B_θ, ρ=ρ, Z_5=Z_5,
                norm_kepler=norm_kepler, C=C, v_φ=v_φ, Z_4=Z_4, Z_3=Z_3,
            )
            Z_7 = Z_7_func(
                a_0=a_0, X=X, v_r=v_r, B_θ=B_θ, ρ=ρ, Z_5=Z_5,
                norm_kepler=norm_kepler, C=C, v_φ=v_φ, Z_4=Z_4,
            )

            dderiv_B_φ = dderiv_B_φ_func(
                B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r,
                v_φ=v_φ, deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ,
                deriv_B_φ=deriv_B_φ, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
                deriv_η_H=deriv_η_H, A=A, C=C, b_r=b_r, b_θ=b_θ, b_φ=b_φ,
                Z_6=Z_6, Z_7=Z_7, deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ,
                deriv_b_r=deriv_b_r,
            )

            deriv_v_r = deriv_v_r_func(
                a_0=a_0, B_θ=B_θ, v_φ=v_φ, ρ=ρ, dderiv_B_φ=dderiv_B_φ, Z_1=Z_1,
                Z_2=Z_2, Z_4=Z_4, norm_kepler=norm_kepler, X=X, v_r=v_r,
            )

            deriv_v_φ = deriv_v_φ_func(
                Z_2=Z_2, deriv_v_r=deriv_v_r, dderiv_B_φ=dderiv_B_φ,
                norm_kepler=norm_kepler, v_r=v_r, X=X, B_θ=B_θ,
            )

        derivs[ODEIndex.B_r] = deriv_B_r
        derivs[ODEIndex.B_φ] = deriv_B_φ
        derivs[ODEIndex.B_θ] = deriv_B_θ
        derivs[ODEIndex.ρ] = deriv_ρ
        derivs[ODEIndex.η_O] = deriv_η_O
        derivs[ODEIndex.η_A] = deriv_η_A
        derivs[ODEIndex.η_H] = deriv_η_H

        derivs[ODEIndex.v_θ] = 0
        if use_E_r:
            derivs[ODEIndex.v_r] = 0
            derivs[ODEIndex.v_φ] = 0
            derivs[ODEIndex.E_r] = deriv_E_r
        else:
            derivs[ODEIndex.v_r] = deriv_v_r
            derivs[ODEIndex.v_φ] = deriv_v_φ
            derivs[ODEIndex.B_φ_prime] = dderiv_B_φ

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


def mod_hydro_solution(
    *, angles, initial_conditions, a_0, norm_kepler_sq,
    relative_tolerance=float_type(1e-6), absolute_tolerance=float_type(1e-10),
    max_steps=500, onroot_func=None, tstop=None, ontstop_func=None,
    η_derivs=True, store_internal=True, root_func=None, root_func_args=None,
    θ_scale=float_type(1), use_E_r=False
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
        store_internal=store_internal, θ_scale=θ_scale, use_E_r=use_E_r,
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

    soln = fix_solution(
        soln, a_0=a_0, norm_kepler_sq=norm_kepler_sq, θ_scale=θ_scale,
        use_E_r=use_E_r, init_con=initial_conditions,
    )

    return soln, internal_data


def solution(
    soln_input, initial_conditions, *, onroot_func=None, tstop=None,
    ontstop_func=None, store_internal=True, root_func=None,
    root_func_args=None, θ_scale=float_type(1), use_E_r=False
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

    soln, internal_data = mod_hydro_solution(
        angles=angles, initial_conditions=init_con, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal, root_func=root_func,
        root_func_args=root_func_args, θ_scale=θ_scale, use_E_r=use_E_r,
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
    mod_hydro solver
    """
    hydro_solution = solution(
        inp, mod_hydro_define_conditions(inp, run),
        store_internal=store_internal, use_E_r=run.use_E_r,
    )
    run.solutions["0"] = hydro_solution
    run.final_solution = run.solutions["0"]
