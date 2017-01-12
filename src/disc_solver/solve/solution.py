# -*- coding: utf-8 -*-
"""
The system of odes
"""

from math import sqrt, tan, degrees, radians
from collections import namedtuple

import logbook

from numpy import concatenate, copy, insert

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)

from .deriv_funcs import dderiv_B_φ_soln, taylor_series
from .utils import gen_sonic_point_rootfn, error_handler

from ..file_format import InternalData, Solution
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "lapackdense"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)

TaylorSolution = namedtuple(
    "TaylorSolution", [
        "angles", "params", "new_angles", "internal_data",
        "new_initial_conditions", "angle_stopped",
    ]
)


def ode_system(
    *, γ, a_0, norm_kepler_sq, init_con, with_taylor=False, η_derivs=True,
    store_internal=True
):
    """
    Set up the system we are solving for.
    """
    # pylint: disable=too-many-statements
    dderiv_ρ_M, dderiv_v_rM, dderiv_v_φM = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs
    )
    if η_derivs:
        η_O_scale = init_con[ODEIndex.η_O] / sqrt(init_con[ODEIndex.ρ])
        η_A_scale = init_con[ODEIndex.η_A] / sqrt(init_con[ODEIndex.ρ])
        η_H_scale = init_con[ODEIndex.η_H] / sqrt(init_con[ODEIndex.ρ])
    else:
        η_O_scale = 0
        η_A_scale = 0
        η_H_scale = 0

    if store_internal:
        internal_data = InternalData()
        derivs_list = internal_data.derivs
        params_list = internal_data.params
        angles_list = internal_data.angles
        v_r_nlist = internal_data.v_r_normal
        v_φ_nlist = internal_data.v_φ_normal
        ρ_nlist = internal_data.ρ_normal
        v_r_taylist = internal_data.v_r_taylor
        v_φ_taylist = internal_data.v_φ_taylor
        ρ_taylist = internal_data.ρ_taylor
        problems = internal_data.problems
    else:
        internal_data = None

    def rhs_equation(θ, params, derivs):
        """
        Compute the ODEs
        """
        # pylint: disable=too-many-statements
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
        norm_B_r, norm_B_φ, norm_B_θ = (
            B_r/B_mag, B_φ/B_mag, B_θ/B_mag
        )

        deriv_B_φ = B_φ_prime
        deriv_B_r = (
            (
                v_θ * B_r - v_r * B_θ + deriv_B_φ * (
                    η_H * norm_B_θ -
                    η_A * norm_B_r * norm_B_φ
                ) + B_φ * (
                    η_A * norm_B_φ * (
                        norm_B_θ * (1/4 - γ) +
                        norm_B_r * tan(θ)
                    ) - η_H * (
                        norm_B_r * (1/4 - γ) +
                        norm_B_θ * tan(θ)
                    )
                )
            ) / (
                η_O + η_A * (1 - norm_B_φ**2)
            ) - B_θ * (1/4 - γ)
        )

        deriv_B_θ = B_θ * tan(θ) - (γ + 3/4) * B_r

        deriv_v_r_taylor = θ * dderiv_v_rM
        deriv_v_r_normal = (
            v_r ** 2 / 2 + v_θ ** 2 + v_φ ** 2 + 5/2 - 2 * γ - norm_kepler_sq +
            a_0 / ρ * (
                B_θ * deriv_B_r + (1/4 - γ) * (B_θ ** 2 + B_φ ** 2)
            )
        ) / v_θ

        deriv_v_r = (
            deriv_v_r_taylor if with_taylor else deriv_v_r_normal
        )

        deriv_v_φ_taylor = θ * dderiv_v_φM
        deriv_v_φ_normal = (
            v_φ * v_θ * tan(θ) - v_φ * v_r / 2 + a_0 / ρ * (
                B_θ * deriv_B_φ - (1/4 - γ) * B_r * B_φ - B_θ * B_φ * tan(θ)
            )
        ) / v_θ
        deriv_v_φ = (
            deriv_v_φ_taylor if with_taylor else deriv_v_φ_normal
        )

        deriv_v_θ = (
            v_r / 2 * (v_θ ** 2 - 4 * γ) + v_θ * (
                tan(θ) * (v_φ ** 2 + 1) + a_0 / ρ * (
                    (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r + B_φ * deriv_B_φ -
                    B_φ ** 2 * tan(θ)
                )
            )
        ) / (1 - v_θ ** 2)

        deriv_ρ_taylor = θ * dderiv_ρ_M
        deriv_ρ_normal = - ρ * (
            (
                2 * γ * v_r + deriv_v_θ
            ) / v_θ - tan(θ)
        )
        deriv_ρ = deriv_ρ_taylor if with_taylor else deriv_ρ_normal

        deriv_ρ_scale = deriv_ρ / sqrt(ρ) / 2
        deriv_η_O = deriv_ρ_scale * η_O_scale
        deriv_η_A = deriv_ρ_scale * η_A_scale
        deriv_η_H = deriv_ρ_scale * η_H_scale

        dderiv_B_φ = dderiv_B_φ_soln(
            B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r,
            v_θ=v_θ, v_φ=v_φ, deriv_v_r=deriv_v_r, deriv_v_θ=deriv_v_θ,
            deriv_v_φ=deriv_v_φ, deriv_B_r=deriv_B_r, deriv_B_θ=deriv_B_θ,
            deriv_B_φ=deriv_B_φ, γ=γ, deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A,
            deriv_η_H=deriv_η_H
        )

        derivs[ODEIndex.B_r] = deriv_B_r
        derivs[ODEIndex.B_φ] = deriv_B_φ
        derivs[ODEIndex.B_θ] = deriv_B_θ
        derivs[ODEIndex.v_r] = deriv_v_r
        derivs[ODEIndex.v_φ] = deriv_v_φ
        derivs[ODEIndex.v_θ] = deriv_v_θ
        derivs[ODEIndex.ρ] = deriv_ρ
        derivs[ODEIndex.B_φ_prime] = dderiv_B_φ
        derivs[ODEIndex.η_O] = deriv_η_O
        derivs[ODEIndex.η_A] = deriv_η_A
        derivs[ODEIndex.η_H] = deriv_η_H
        if __debug__:
            log.debug("θ: {}, {}", θ, degrees(θ))

        if store_internal:
            params_list.append(copy(params))
            derivs_list.append(copy(derivs))
            angles_list.append(θ)
            v_r_nlist.append(deriv_v_r_normal)
            v_φ_nlist.append(deriv_v_φ_normal)
            ρ_nlist.append(deriv_ρ_normal)
            v_r_taylist.append(deriv_v_r_taylor)
            v_φ_taylist.append(deriv_v_φ_taylor)
            ρ_taylist.append(deriv_ρ_taylor)

            if len(params_list) != len(angles_list):
                log.error(
                    "Internal data not consistent, "
                    "params is {}, angles is {}".format(
                        len(params_list), len(angles_list)
                    )
                )

        return 0
    return rhs_equation, internal_data


def jacobian_viewer_generator(internal_data):
    """
    Generator for jacobian viewer
    """
    if internal_data is not None:
        jaclist = internal_data.jacobian_data.jacobians
        anglelist = internal_data.jacobian_data.angles
        paramlist = internal_data.jacobian_data.params
        derivlist = internal_data.jacobian_data.derivs

        def jacobian_viewer(
            method, jac_flag, θ, params, derivs, jac_tmp, aux_data
        ):
            """
            Get jacobian generated by solver
            """
            # pylint: disable=unused-argument
            anglelist.append(θ)
            paramlist.append(copy(params))
            derivlist.append(copy(derivs))
            jaclist.append(copy(jac_tmp))

            return jac_flag

        return jacobian_viewer

    return lambda _1, jac_flag, _2, _3, _4, _5, _6: jac_flag


def taylor_solution(
    *, angles, init_con, γ, a_0, norm_kepler_sq, taylor_stop_angle,
    relative_tolerance=1e-6, absolute_tolerance=1e-10, max_steps=500,
    η_derivs=True, store_internal=True
):
    """
    Compute solution using taylor series
    """
    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=init_con, with_taylor=True, η_derivs=η_derivs,
        store_internal=store_internal,
    )

    jacobian_viewer = jacobian_viewer_generator(internal_data)

    solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=error_handler,
        tstop=taylor_stop_angle,
        jac_viewer=jacobian_viewer,
        bdf_stability_detection=True,
    )

    try:
        soln = solver.solve(angles, init_con)
    except CVODESolveFailed as e:
        soln = e.soln
        raise RuntimeError(
            "Taylor solver stopped in at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
    else:
        raise RuntimeError("Taylor solver did not find taylor_stop_angle")

    new_angles = angles[angles > taylor_stop_angle]
    insert(new_angles, 0, taylor_stop_angle)

    return TaylorSolution(
        angles=soln.values.t, params=soln.values.y, new_angles=new_angles,
        internal_data=internal_data, new_initial_conditions=soln.tstop.y[0],
        angle_stopped=soln.tstop.t[0],
    )


def main_solution(
    *, angles, system_initial_conditions, ode_initial_conditions, γ, a_0,
    norm_kepler_sq, relative_tolerance=1e-6, absolute_tolerance=1e-10,
    max_steps=500, onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, η_derivs=True, store_internal=True, root_func=None,
    root_func_args=None
):
    """
    Find solution
    """
    extra_args = {}
    if find_sonic_point and root_func is not None:
        raise RuntimeError("Cannot use both sonic point finder and root_func")
    elif find_sonic_point:
        extra_args["rootfn"] = gen_sonic_point_rootfn(1)
        extra_args["nr_rootfns"] = 1
    elif root_func is not None:
        extra_args["rootfn"] = root_func
        if root_func_args is not None:
            extra_args["nr_rootfns"] = root_func_args
        else:
            raise RuntimeError("Need to specify size of root array")

    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=system_initial_conditions, η_derivs=η_derivs,
        store_internal=store_internal, with_taylor=False,
    )

    jacobian_viewer = jacobian_viewer_generator(internal_data)

    solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=error_handler,
        onroot=onroot_func,
        tstop=tstop,
        ontstop=ontstop_func,
        jac_viewer=jacobian_viewer,
        bdf_stability_detection=True,
        **extra_args
    )

    try:
        soln = solver.solve(angles, ode_initial_conditions)
    except CVODESolveFailed as e:
        soln = e.soln
        log.warn(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                degrees(soln.errors.t), soln.flag, soln.message
            )
        )
    except CVODESolveFoundRoot as e:
        soln = e.soln
        for root in soln.roots.t:
            log.notice("Found sonic point at {}".format(degrees(root)))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        for tstop in soln.tstop.t:
            log.notice("Stopped at {}".format(degrees(tstop)))

    return soln, internal_data


def solution(
    soln_input, initial_conditions, *,
    onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, store_internal=True, root_func=None,
    root_func_args=None
):
    """
    Find solution
    """
    angles = initial_conditions.angles
    init_con = initial_conditions.init_con
    γ = initial_conditions.γ
    a_0 = initial_conditions.a_0
    norm_kepler_sq = initial_conditions.norm_kepler_sq
    absolute_tolerance = soln_input.absolute_tolerance
    relative_tolerance = soln_input.relative_tolerance
    max_steps = soln_input.max_steps
    taylor_stop_angle = radians(soln_input.taylor_stop_angle)
    η_derivs = soln_input.η_derivs

    if taylor_stop_angle is None:
        post_taylor_angles = angles
        post_taylor_initial_conditions = init_con
    else:
        taylor_soln = taylor_solution(
            angles=angles, init_con=init_con, γ=γ, a_0=a_0,
            norm_kepler_sq=norm_kepler_sq,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, max_steps=max_steps,
            taylor_stop_angle=taylor_stop_angle, η_derivs=η_derivs,
            store_internal=store_internal
        )
        post_taylor_angles = taylor_soln.new_angles
        post_taylor_initial_conditions = taylor_soln.new_initial_conditions
        taylor_internal = taylor_soln.internal_data

    soln, internal_data = main_solution(
        angles=post_taylor_angles, system_initial_conditions=init_con,
        ode_initial_conditions=post_taylor_initial_conditions, γ=γ, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal,
        find_sonic_point=find_sonic_point, root_func=root_func,
        root_func_args=root_func_args,
    )

    if store_internal and taylor_stop_angle is not None:
        internal_data = taylor_internal + internal_data

    if taylor_stop_angle is None:
        joined_angles = soln.values.t
        joined_solution = soln.values.y
    else:
        joined_angles = concatenate((taylor_soln.angles, soln.values.t))
        joined_solution = concatenate((taylor_soln.params, soln.values.y))

    return Solution(
        solution_input=soln_input, initial_conditions=initial_conditions,
        flag=soln.flag, coordinate_system=COORDS, internal_data=internal_data,
        angles=joined_angles, solution=joined_solution, t_roots=soln.roots.t,
        y_roots=soln.roots.y
    )
