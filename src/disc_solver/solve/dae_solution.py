# -*- coding: utf-8 -*-
"""
The system of daes
"""

from math import sqrt, tan, degrees

import logbook

from numpy import copy

from scikits.odes import dae
from scikits.odes.sundials import (
    IDASolveFailed, IDASolveFoundRoot, IDASolveReachedTSTOP
)

from .deriv_funcs import B_unit_derivs, C_func, A_func
from .utils import gen_sonic_point_rootfn, ode_error_handler

from ..file_format import DAEInternalData, Solution
from ..utils import ODEIndex, sec


INTEGRATOR = "ida"
LINSOLVER = "lapackdense"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)


def dae_system(
    *, γ, a_0, norm_kepler_sq, init_con, η_derivs=True, store_internal=True
):
    """
    Set up the system we are solving for.
    """
    if η_derivs:
        η_O_scale = init_con[ODEIndex.η_O] / sqrt(init_con[ODEIndex.ρ])
        η_A_scale = init_con[ODEIndex.η_A] / sqrt(init_con[ODEIndex.ρ])
        η_H_scale = init_con[ODEIndex.η_H] / sqrt(init_con[ODEIndex.ρ])
    else:
        η_O_scale = 0
        η_A_scale = 0
        η_H_scale = 0

    if store_internal:
        internal_data = DAEInternalData()
        derivs_list = internal_data.derivs
        params_list = internal_data.params
        angles_list = internal_data.angles
        problems = internal_data.problems
    else:
        internal_data = None

    def rhs_equation(θ, params, derivs, residual):
        """
        Compute the ODEs
        """
        B_r = params[ODEIndex.B_r]
        B_φ = params[ODEIndex.B_φ]
        B_θ = params[ODEIndex.B_θ]
        v_r = params[ODEIndex.v_r]
        v_φ = params[ODEIndex.v_φ]
        v_θ = params[ODEIndex.v_θ]
        ρ = params[ODEIndex.ρ]
        B_φ_prime = params[ODEIndex.B_φ_prime]
        η_O = params[ODEIndex.η_O]
        η_A = params[ODEIndex.η_A]
        η_H = params[ODEIndex.η_H]

        deriv_B_r = derivs[ODEIndex.B_r]
        deriv_B_φ = derivs[ODEIndex.B_φ]
        deriv_B_θ = derivs[ODEIndex.B_θ]
        deriv_v_r = derivs[ODEIndex.v_r]
        deriv_v_φ = derivs[ODEIndex.v_φ]
        deriv_v_θ = derivs[ODEIndex.v_θ]
        deriv_ρ = derivs[ODEIndex.ρ]
        deriv_B_φ_prime = derivs[ODEIndex.B_φ_prime]
        deriv_η_O = derivs[ODEIndex.η_O]
        deriv_η_A = derivs[ODEIndex.η_A]
        deriv_η_H = derivs[ODEIndex.η_H]

        # check sanity of input values
        if ρ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative density")
            return 1
        if v_θ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative velocity")
            return -1

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag
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

        residual[ODEIndex.B_φ] = B_φ_prime - deriv_B_φ

        residual[ODEIndex.v_r] = v_θ * deriv_v_r + (
            sqrt(norm_kepler_sq) - v_φ
        ) * (
            sqrt(norm_kepler_sq) + v_φ
        ) - 5 / 2 + 2 * γ - a_0 / ρ * (
            B_θ * deriv_B_r + (1 / 4 - γ) * (B_θ ** 2 + B_φ ** 2)
        ) - v_r ** 2 / 2 - v_θ ** 2

        residual[ODEIndex.v_φ] = v_θ * deriv_v_φ - a_0 / ρ * (
            B_θ * B_φ_prime - (1 / 4 - γ) * B_r * B_φ - B_θ * B_φ * tan(θ)
        ) - v_θ * v_φ * tan(θ) + v_φ * v_r / 2

        residual[ODEIndex.ρ] = deriv_ρ * v_θ + ρ * (
            2 * γ * v_r + deriv_v_θ - v_θ * tan(θ)
        )

        residual[ODEIndex.v_θ] = v_θ * deriv_v_θ + a_0 / ρ * (
            (1 / 4 - γ) * B_θ * B_r + B_r * deriv_B_r +
            B_φ * B_φ_prime - B_φ ** 2 * tan(θ)
        ) + v_r * v_θ / 2 + v_φ ** 2 * tan(θ) + deriv_ρ / ρ

        residual[ODEIndex.B_r] = v_θ * B_r - v_r * B_θ + B_φ * (
            η_A * b_φ * (
                b_θ * (1 / 4 - γ) + b_r * tan(θ)
            ) - η_H * (
                b_r * (1 / 4 - γ) + b_θ * tan(θ)
            )
        ) - (
            η_O + η_A * (1 - b_φ) * (1 + b_φ)
        ) * (
            B_θ * (1 / 4 - γ) + deriv_B_r
        ) + B_φ_prime * (
            η_H * b_θ - η_A * b_r * b_φ
        )

        residual[ODEIndex.B_φ_prime] = (
            v_φ * B_r - 4 / (3 - 4 * γ) * (
                deriv_v_φ * B_θ + v_φ * deriv_B_θ
            ) + (B_θ * (1 / 4 - γ) + deriv_B_r) * (
                η_H * b_r - η_A * b_θ * b_φ
            ) - A * v_θ * B_r - C * deriv_v_θ * B_r - C * v_θ * deriv_B_r +
            A * v_r * B_θ + C * deriv_v_r * B_θ + C * v_r * deriv_B_θ +
            B_φ_prime * (
                v_θ * 4 / (3 - 4 * γ) + η_O * tan(θ) - deriv_η_O -
                deriv_η_H * C * b_θ - deriv_η_A * (
                    1 - b_r ** 2 - C * b_r * b_φ
                ) + η_H * (
                    b_φ * (γ + 3 / 4) + C * (
                        b_r * (1 / 4 - γ) + b_θ * tan(θ) + deriv_b_θ
                    ) - A * b_θ
                ) + η_A * (
                    tan(θ) * (1 - b_r ** 2) + (1 / 4 - γ) * b_r * b_φ -
                    C * b_φ * (
                        b_θ * (1 / 4 - γ) + b_r * tan(θ)
                    ) + 2 * b_r * deriv_b_r - A * b_r * b_φ -
                    C * deriv_b_r * b_φ - C * b_r * deriv_b_φ + b_r * b_θ
                )
            ) + B_φ * (
                deriv_η_O * tan(θ) + η_O * (
                    sec(θ) ** 2 - 1 / 4 + γ
                ) - v_r + 4 / (3 - 4 * γ) * deriv_v_θ + η_H * (
                    A * (
                        b_r * (1 / 4 - γ) + b_θ * tan(θ)
                    ) + C * b_φ * (
                        deriv_b_r * (1 / 4 - γ) + deriv_b_θ * tan(θ) +
                        b_θ * sec(θ) ** 2
                    ) - b_φ * tan(θ) - deriv_b_φ * (1 / 4 - γ)
                ) + deriv_η_H * (
                    C * (b_r * (1 / 4 - γ) + b_θ * tan(θ)) - b_φ * (1 / 4 - γ)
                ) + η_A * (
                    sec(θ) ** 2 * (1 - b_r ** 2) +
                    2 * tan(θ) * b_r * deriv_b_r +
                    (1 / 4 - γ) * deriv_b_r * b_θ +
                    (1 / 4 - γ) * b_r * deriv_b_θ -
                    (
                        A * b_φ + C * deriv_b_φ
                    ) * (
                        b_θ * (1 / 4 - γ) + b_r * tan(θ)
                    ) - C * b_φ * (
                        deriv_b_θ * (1 / 4 - γ) + deriv_b_r * tan(θ) +
                        b_r * sec(θ) ** 2
                    ) - (1 / 4 - γ) * (1 - b_θ ** 2) - tan(θ) * b_r * b_θ
                ) + deriv_η_A * (
                    tan(θ) * (1 - b_r ** 2) + (1 / 4 - γ) * b_r * b_θ -
                    C * b_φ * (
                        b_θ * (1 / 4 - γ) + b_r * tan(θ)
                    )
                )
            ) - deriv_B_φ_prime * (
                η_O + η_A * (1 - b_r ** 2) + C * (η_H * b_θ - η_A * b_r * b_φ)
            )
        )

        deriv_ρ_scale = deriv_ρ / sqrt(ρ) / 2
        residual[ODEIndex.η_O] = deriv_η_O - deriv_ρ_scale * η_O_scale
        residual[ODEIndex.η_A] = deriv_η_A - deriv_ρ_scale * η_A_scale
        residual[ODEIndex.η_H] = deriv_η_H - deriv_ρ_scale * η_H_scale

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


def solution(
    input, initial_conditions, *,
    onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, store_internal=True
):
    """
    Find solution
    """
    angles = initial_conditions.angles
    init_con = initial_conditions.init_con
    deriv_init_con = initial_conditions.deriv_init_con
    γ = initial_conditions.γ
    a_0 = initial_conditions.a_0
    norm_kepler_sq = initial_conditions.norm_kepler_sq
    absolute_tolerance = input.absolute_tolerance
    relative_tolerance = input.relative_tolerance
    max_steps = input.max_steps
    η_derivs = input.η_derivs

    extra_args = {}
    if find_sonic_point:
        extra_args["rootfn"] = gen_sonic_point_rootfn(1)
        extra_args["nr_rootfns"] = 1

    system, internal_data = dae_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        η_derivs=η_derivs, store_internal=store_internal
    )
    solver = dae(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=ode_error_handler,
        onroot=onroot_func,
        tstop=tstop,
        ontstop=ontstop_func,
        **extra_args
    )

    try:
        soln = solver.solve(angles, init_con, deriv_init_con)
    except IDASolveFailed as e:
        soln = e.soln
        log.warn(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    except IDASolveFoundRoot as e:
        soln = e.soln
        for root in soln.roots.t:
            log.notice("Found sonic point at {}".format(degrees(root)))
    except IDASolveReachedTSTOP as e:
        soln = e.soln
        for tstop in soln.tstop.t:
            log.notice("Stopped at {}".format(degrees(tstop)))

    return Solution(
        solution_input=input, initial_conditions=initial_conditions,
        flag=soln.flag, coordinate_system=COORDS, internal_data=internal_data,
        angles=soln.values.t, solution=soln.values.y,
        derivatives=soln.values.ydot, t_roots=soln.roots.t,
        y_roots=soln.roots.y
    )
