# -*- coding: utf-8 -*-
"""
The system of odes
"""

import logbook

# DO NOT IMPORT MATH, BREAKS FLOAT SUPPORT
from numpy import sqrt, tan, copy, errstate, degrees

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from .config import define_conditions
from .deriv_funcs import B_unit_derivs, A_func, C_func, deriv_η_skw_func
from .hydrostatic import (
    X_dash_func, X_func, Z_1_func, Z_4_func, Z_5_func, dderiv_B_φ_func
)
from .utils import (
    error_handler, rad_to_scaled, scaled_to_rad,
    SolverError,
)

from ..float_handling import float_type
from ..file_format import InternalData, Solution
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "dense"
COORDS = "spherical midplane 0"
SOLUTION_TYPE = "mod_hydro"
SONIC_POINT_TOLERANCE = float_type(0.01)

log = logbook.Logger(__name__)


def Z_2_func(
    *, θ, a_0, X, v_r, B_φ, B_θ, ρ, η_A, η_O, η_H, b_φ, b_r, b_θ, X_dash,
    deriv_B_φ, deriv_B_θ, deriv_b_θ, deriv_b_φ, deriv_b_r, deriv_η_O,
    deriv_η_A, deriv_η_H, norm_kepler
):
    """
    Compute the value of the variable Z_2
    """
    return - (
        5 / 2 + a_0 / ρ * (
            deriv_B_φ * B_φ / 2 - deriv_B_θ * deriv_B_φ * X -
            B_θ * deriv_B_φ * X_dash - B_θ / (
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


def ode_system(
    *, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1), η_derivs=True,
    store_internal=True
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
            b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

        X = X_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

        deriv_B_φ = B_φ_prime
        deriv_B_θ = B_θ * tan(θ) - 3/4 * B_r

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

        deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
            B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r,
            deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ
        )

        C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)

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
            θ=θ, a_0=a_0, B_φ=B_φ, B_θ=B_θ, B_r=B_r, ρ=ρ, deriv_B_r=deriv_B_r,
            deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ, v_φ=v_φ, deriv_ρ=deriv_ρ
        )
        Z_2 = Z_2_func(
            θ=θ, a_0=a_0, X=X, v_r=v_r, B_φ=B_φ, B_θ=B_θ, ρ=ρ, η_A=η_A,
            η_O=η_O, η_H=η_H, b_φ=b_φ, b_r=b_r, b_θ=b_θ, X_dash=X_dash,
            deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ, deriv_b_θ=deriv_b_θ,
            deriv_b_φ=deriv_b_φ, deriv_b_r=deriv_b_r, deriv_η_O=deriv_η_O,
            deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H, norm_kepler=norm_kepler,
        )
        Z_3 = Z_3_func(
            v_r=v_r, B_θ=B_θ, norm_kepler=norm_kepler, η_O=η_O, η_A=η_A,
            b_φ=b_φ,
        )
        Z_4 = Z_4_func(B_θ=B_θ, B_r=B_r, B_φ=B_φ, deriv_B_φ=deriv_B_φ, θ=θ)
        Z_5 = Z_5_func(
            η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C
        )
        Z_6 = Z_6_func(
            a_0=a_0, X=X, v_r=v_r, B_θ=B_θ, ρ=ρ, Z_5=Z_5,
            norm_kepler=norm_kepler, C=C, v_φ=v_φ, Z_4=Z_4, Z_3=Z_3,
        )
        Z_7 = Z_7_func(
            a_0=a_0, X=X, v_r=v_r, B_θ=B_θ, ρ=ρ, Z_5=Z_5,
            norm_kepler=norm_kepler, C=C, v_φ=v_φ, Z_4=Z_4,
        )

        dderiv_B_φ = dderiv_B_φ_func(
            B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r, v_φ=v_φ,
            deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ,
            deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H, A=A,
            C=C, b_r=b_r, b_θ=b_θ, b_φ=b_φ, Z_6=Z_6, Z_7=Z_7,
            deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, deriv_b_r=deriv_b_r,
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


def mod_hydro_solution(
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

    soln, internal_data = mod_hydro_solution(
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
    mod_hydro solver
    """
    cons = define_conditions(inp)
    cons.γ = 0
    hydro_solution = solution(
        inp, define_conditions(inp), store_internal=store_internal
    )
    run.solutions["0"] = hydro_solution
    run.final_solution = run.solutions["0"]
