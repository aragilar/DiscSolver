# -*- coding: utf-8 -*-
"""
The system of odes
"""

from collections import namedtuple

import logbook

from numpy import (
    array, concatenate, copy, insert, errstate, sqrt, tan, degrees, radians,
)

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from .deriv_funcs import (
    dderiv_B_φ_soln, taylor_series, get_taylor_first_order,
    get_taylor_second_order, get_taylor_third_order, deriv_B_r_func,
    deriv_η_skw_func, deriv_B_φ_func, deriv_E_r_func
)
from .utils import (
    gen_sonic_point_rootfn, error_handler, rad_to_scaled, scaled_to_rad,
    SolverError,
)

from ..float_handling import float_type
from ..file_format import InternalData, Solution
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "dense"
COORDS = "spherical midplane 0"

log = logbook.Logger(__name__)

TaylorSolution = namedtuple(
    "TaylorSolution", [
        "angles", "params", "new_angles", "internal_data",
        "new_initial_conditions", "angle_stopped",
    ]
)


def ode_system(
    *, γ, a_0, norm_kepler_sq, init_con, θ_scale=float_type(1),
    with_taylor=False, η_derivs=True, store_internal=True, use_E_r=False
):
    """
    Set up the system we are solving for.
    """
    # pylint: disable=too-many-statements
    taylor_derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs
    )
    dderiv_v_rM = taylor_derivs[ODEIndex.v_r]
    dderiv_v_φM = taylor_derivs[ODEIndex.v_φ]
    dderiv_ρ_M = taylor_derivs[ODEIndex.ρ]

    norm_kepler = sqrt(norm_kepler_sq)
    η_O_0 = init_con[ODEIndex.η_O]
    η_A_0 = init_con[ODEIndex.η_A]
    η_H_0 = init_con[ODEIndex.η_H]

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
        v_θ = params[ODEIndex.v_θ]
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
        if v_θ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative velocity")
            return -1

        if use_E_r:
            deriv_B_φ = deriv_B_φ_func(
                θ=θ, γ=γ, B_r=B_r, B_θ=B_θ, B_φ=B_φ, v_r=v_r, v_φ=v_φ, v_θ=v_θ,
                η_O=η_O, η_A=η_A, η_H=η_H, E_r=E_r
            )
        else:
            deriv_B_φ = B_φ_prime

        deriv_B_r = deriv_B_r_func(
            B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r,
            v_θ=v_θ, deriv_B_φ=deriv_B_φ, γ=γ,
        )

        deriv_B_θ = B_θ * tan(θ) - (γ + 3/4) * B_r

        deriv_v_r_taylor = θ * dderiv_v_rM
        with errstate(invalid="ignore", divide="ignore"):
            deriv_v_r_normal = (
                v_r ** 2 / 2 + v_θ ** 2 + 5/2 - 2 * γ +
                (v_φ - norm_kepler) * (v_φ + norm_kepler) + a_0 / ρ * (
                    B_θ * deriv_B_r + (1/4 - γ) * (B_θ ** 2 + B_φ ** 2)
                )
            ) / v_θ

        deriv_v_r = (
            deriv_v_r_taylor if with_taylor else deriv_v_r_normal
        )

        deriv_v_φ_taylor = θ * dderiv_v_φM
        with errstate(invalid="ignore", divide="ignore"):
            deriv_v_φ_normal = (
                v_φ * v_θ * tan(θ) - v_φ * v_r / 2 + a_0 / ρ * (
                    B_θ * deriv_B_φ - (1/4 - γ) * B_r * B_φ -
                    B_θ * B_φ * tan(θ)
                )
            ) / v_θ
        deriv_v_φ = (
            deriv_v_φ_taylor if with_taylor else deriv_v_φ_normal
        )

        deriv_v_θ = (
            v_r / 2 * (v_θ ** 2 - 4 * γ) + v_θ * (
                tan(θ) * (v_φ ** 2 + 1) + a_0 / ρ * (
                    (1/4 - γ) * B_θ * B_r + B_r * deriv_B_r +
                    B_φ * deriv_B_φ - B_φ ** 2 * tan(θ)
                )
            )
        ) / ((1 - v_θ) * (1 + v_θ))

        deriv_ρ_taylor = θ * dderiv_ρ_M
        with errstate(invalid="ignore", divide="ignore"):
            deriv_ρ_normal = - ρ * (
                (
                    2 * γ * v_r + deriv_v_θ
                ) / v_θ - tan(θ)
            )
        deriv_ρ = deriv_ρ_taylor if with_taylor else deriv_ρ_normal

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
            B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
            b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

            deriv_E_r = deriv_E_r_func(
                γ=γ, θ=θ, v_r=v_r, v_φ=v_φ, B_r=B_r, B_θ=B_θ, B_φ=B_φ, η_O=η_O,
                η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ,
                deriv_B_r=deriv_B_r, deriv_B_φ=deriv_B_φ
            )
        else:
            dderiv_B_φ = dderiv_B_φ_soln(
                B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ,
                v_r=v_r, v_θ=v_θ, v_φ=v_φ, deriv_v_r=deriv_v_r,
                deriv_v_θ=deriv_v_θ, deriv_v_φ=deriv_v_φ, deriv_B_r=deriv_B_r,
                deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ, γ=γ,
                deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H
            )

        derivs[ODEIndex.B_r] = deriv_B_r
        derivs[ODEIndex.B_φ] = deriv_B_φ
        derivs[ODEIndex.B_θ] = deriv_B_θ
        derivs[ODEIndex.v_r] = deriv_v_r
        derivs[ODEIndex.v_φ] = deriv_v_φ
        derivs[ODEIndex.v_θ] = deriv_v_θ
        derivs[ODEIndex.ρ] = deriv_ρ
        derivs[ODEIndex.η_O] = deriv_η_O
        derivs[ODEIndex.η_A] = deriv_η_A
        derivs[ODEIndex.η_H] = deriv_η_H

        if use_E_r:
            derivs[ODEIndex.E_r] = deriv_E_r
        else:
            derivs[ODEIndex.B_φ_prime] = dderiv_B_φ

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
    relative_tolerance=float_type(1e-6), absolute_tolerance=float_type(1e-10),
    max_steps=500, η_derivs=True, store_internal=True, θ_scale=float_type(1)
):
    """
    Compute solution using taylor series
    """
    taylor_stop_angle = rad_to_scaled(taylor_stop_angle, θ_scale)
    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=init_con, with_taylor=True, η_derivs=η_derivs,
        store_internal=store_internal, θ_scale=θ_scale,
    )

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
        bdf_stability_detection=True,
    )

    try:
        soln = solver.solve(angles, init_con)
    except CVODESolveFailed as e:
        soln = e.soln
        raise SolverError(
            "Taylor solver stopped in at {} with flag {!s}.\n{}".format(
                degrees(scaled_to_rad(soln.errors.t, θ_scale)),
                soln.flag, soln.message
            )
        )
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
    else:
        raise SolverError("Taylor solver did not find taylor_stop_angle")

    new_angles = angles[angles > taylor_stop_angle]
    insert(new_angles, 0, taylor_stop_angle)

    return TaylorSolution(
        angles=scaled_to_rad(soln.values.t, θ_scale), params=soln.values.y,
        new_angles=new_angles, internal_data=internal_data,
        new_initial_conditions=soln.tstop.y[0], angle_stopped=soln.tstop.t[0],
    )


def taylor_jump(
    *, angles, init_con, γ, a_0, taylor_stop_angle,
    η_derivs=True, θ_scale=float_type(1)
):
    """
    Perform a single large step off the midplane using a taylor series
    expansion.
    """
    taylor_stop_angle = rad_to_scaled(taylor_stop_angle, θ_scale)
    new_angles = angles[angles > taylor_stop_angle]
    insert(new_angles, 0, taylor_stop_angle)

    first_order_taylor_values = get_taylor_first_order(
        init_con=init_con, γ=γ
    )
    second_order_taylor_values = get_taylor_second_order(
        init_con=init_con, γ=γ, a_0=a_0, η_derivs=η_derivs
    )
    third_order_taylor_values = get_taylor_third_order(
        init_con=init_con, γ=γ, a_0=a_0, η_derivs=η_derivs
    )

    taylor_values = init_con + taylor_stop_angle * (
        first_order_taylor_values +
        taylor_stop_angle / 2 * (
            second_order_taylor_values +
            taylor_stop_angle / 3 * (
                third_order_taylor_values
            )
        )
    )

    return TaylorSolution(
        angles=scaled_to_rad(array([0, taylor_stop_angle]), θ_scale),
        params=array([init_con, taylor_values]),
        new_angles=new_angles, internal_data=None,
        new_initial_conditions=taylor_values, angle_stopped=taylor_stop_angle,
    )


def main_solution(
    *, angles, system_initial_conditions, ode_initial_conditions, γ, a_0,
    norm_kepler_sq, relative_tolerance=float_type(1e-6),
    absolute_tolerance=float_type(1e-10), max_steps=500, onroot_func=None,
    find_sonic_point=False, tstop=None, ontstop_func=None, η_derivs=True,
    store_internal=True, root_func=None, root_func_args=None,
    θ_scale=float_type(1), use_E_r=False
):
    """
    Find solution
    """
    extra_args = {}
    if find_sonic_point and root_func is not None:
        raise SolverError("Cannot use both sonic point finder and root_func")
    elif find_sonic_point:
        extra_args["rootfn"] = gen_sonic_point_rootfn(1)
        extra_args["nr_rootfns"] = 1
    elif root_func is not None:
        extra_args["rootfn"] = root_func
        if root_func_args is not None:
            extra_args["nr_rootfns"] = root_func_args
        else:
            raise SolverError("Need to specify size of root array")

    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=system_initial_conditions, η_derivs=η_derivs,
        store_internal=store_internal, with_taylor=False, θ_scale=θ_scale,
        use_E_r=use_E_r,
    )

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
        tstop=rad_to_scaled(tstop, θ_scale),
        ontstop=ontstop_func,
        bdf_stability_detection=True,
        **extra_args
    )

    try:
        soln = solver.solve(angles, ode_initial_conditions)
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
        if find_sonic_point:
            log.notice("Found sonic point at {}".format(
                degrees(scaled_to_rad(soln.roots.t, θ_scale))
            ))
        else:
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
    soln_input, initial_conditions, *,
    onroot_func=None, find_sonic_point=False, tstop=None,
    ontstop_func=None, store_internal=True, root_func=None,
    root_func_args=None, with_taylor=True, modified_initial_conditions=None,
    θ_scale=float_type(1), use_E_r=False
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
    η_derivs = soln_input.η_derivs
    use_taylor_jump = soln_input.use_taylor_jump

    if with_taylor and use_E_r:
        raise SolverError(
            "Cannot use taylor series and E_r at the same time currently"
        )
    elif with_taylor:
        taylor_stop_angle = radians(soln_input.taylor_stop_angle)
    else:
        taylor_stop_angle = None

    if taylor_stop_angle is None and modified_initial_conditions is None:
        post_taylor_angles = angles
        post_taylor_initial_conditions = init_con
    elif taylor_stop_angle is None:
        post_taylor_angles = modified_initial_conditions.angles
        post_taylor_initial_conditions = modified_initial_conditions.init_con
    else:
        if use_taylor_jump:
            taylor_soln = taylor_jump(
                angles=angles, init_con=init_con, γ=γ, a_0=a_0,
                taylor_stop_angle=taylor_stop_angle, η_derivs=η_derivs,
                θ_scale=θ_scale,
            )

        else:
            taylor_soln = taylor_solution(
                angles=angles, init_con=init_con, γ=γ, a_0=a_0,
                norm_kepler_sq=norm_kepler_sq,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance, max_steps=max_steps,
                taylor_stop_angle=taylor_stop_angle, η_derivs=η_derivs,
                store_internal=store_internal, θ_scale=θ_scale,
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
        root_func_args=root_func_args, θ_scale=θ_scale, use_E_r=use_E_r,
    )

    if store_internal and taylor_stop_angle is not None:
        if taylor_internal is not None:
            internal_data = taylor_internal + internal_data

    if taylor_stop_angle is None:
        joined_angles = scaled_to_rad(soln.values.t, θ_scale)
        joined_solution = soln.values.y
    else:
        joined_angles = concatenate(
            (taylor_soln.angles, scaled_to_rad(soln.values.t, θ_scale))
        )
        joined_solution = concatenate((taylor_soln.params, soln.values.y))

    if find_sonic_point and soln.roots.t is not None:
        sonic_point = scaled_to_rad(soln.roots.t[0], θ_scale)
        sonic_point_values = soln.roots.y[0]
    else:
        sonic_point = None
        sonic_point_values = None

    return Solution(
        solution_input=soln_input, initial_conditions=initial_conditions,
        flag=soln.flag, coordinate_system=COORDS, internal_data=internal_data,
        angles=joined_angles, solution=joined_solution,
        t_roots=(
            scaled_to_rad(soln.roots.t, θ_scale)
            if soln.roots.t is not None else None
        ), y_roots=soln.roots.y, sonic_point=sonic_point,
        sonic_point_values=sonic_point_values,
    )
