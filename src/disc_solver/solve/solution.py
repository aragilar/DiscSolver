# -*- coding: utf-8 -*-
"""
The system of odes
"""

from collections import namedtuple

import logbook

from numpy import (
    array, concatenate, copy, insert, errstate, sqrt, tan, degrees, radians,
    zeros, nan
)

from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from .deriv_funcs import (
    dderiv_B_φ_soln, deriv_B_r_func, deriv_η_skw_func, deriv_B_φ_func,
    deriv_E_r_func, C_func, A_func, Z_5_func, B_unit_derivs,
)
from .deriv_midplane import (
    taylor_series, get_taylor_first_order, get_taylor_second_order,
    get_taylor_third_order,
)
from .j_e_funcs import J_r_func, J_θ_func, J_φ_func
from .utils import (
    velocity_stop_generator, error_handler, rad_to_scaled, scaled_to_rad,
    SolverError,
)

from ..float_handling import float_type
from ..file_format import InternalData, Solution
from ..utils import ODEIndex, deduplicate_and_interpolate

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
    with_taylor=False, η_derivs=True, store_internal=True, use_E_r=False,
    v_θ_sonic_crit=None, after_sonic=None, deriv_v_θ_func=None,
    derivs_post_solution=False,
):
    """
    Set up the system we are solving for.
    """
    # pylint: disable=too-many-statements
    if v_θ_sonic_crit is not None and after_sonic is None:
        after_sonic = v_θ_sonic_crit

    taylor_derivs = taylor_series(
        γ=γ, a_0=a_0, init_con=init_con, η_derivs=η_derivs, use_E_r=use_E_r,
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
        if derivs_post_solution:
            B_r = params[..., ODEIndex.B_r]
            B_φ = params[..., ODEIndex.B_φ]
            B_θ = params[..., ODEIndex.B_θ]
            v_r = params[..., ODEIndex.v_r]
            v_φ = params[..., ODEIndex.v_φ]
            v_θ = params[..., ODEIndex.v_θ]
            ρ = params[..., ODEIndex.ρ]
            η_O = params[..., ODEIndex.η_O]
            η_A = params[..., ODEIndex.η_A]
            η_H = params[..., ODEIndex.η_H]
            if use_E_r:
                E_r = params[..., ODEIndex.E_r]
                B_φ_prime = None
            else:
                B_φ_prime = params[..., ODEIndex.B_φ_prime]
                E_r = None
        else:
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
                B_φ_prime = None
            else:
                B_φ_prime = params[ODEIndex.B_φ_prime]
                E_r = None

        # check sanity of input values
        if not derivs_post_solution and ρ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative density")
            return 1
        if not derivs_post_solution and v_θ < 0:
            if store_internal:
                # pylint: disable=unsubscriptable-object
                problems[θ].append("negative velocity")
            return -1

        B_mag = sqrt(B_r**2 + B_φ**2 + B_θ**2)
        b_r, b_φ, b_θ = B_r/B_mag, B_φ/B_mag, B_θ/B_mag

        C = C_func(η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ)
        Z_5 = Z_5_func(
            η_O=η_O, η_A=η_A, η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, C=C,
        )

        if use_E_r:
            deriv_B_φ = deriv_B_φ_func(
                θ=θ, γ=γ, B_r=B_r, B_θ=B_θ, B_φ=B_φ, v_r=v_r, v_φ=v_φ, v_θ=v_θ,
                η_O=η_O, η_A=η_A, η_H=η_H, E_r=E_r, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
                C=C, Z_5=Z_5,
            )
        else:
            deriv_B_φ = B_φ_prime

        deriv_B_r = deriv_B_r_func(
            B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ, v_r=v_r,
            v_θ=v_θ, deriv_B_φ=deriv_B_φ, γ=γ, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
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

        if deriv_v_θ_func is not None:
            deriv_v_θ = deriv_v_θ_func(θ)
        # There was code which used sonic_point.py, but the equations and logic
        # are likely wrong, using either jumping or interpolation are likely
        # wiser
        else:
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
            J_r = J_r_func(θ=θ, B_φ=B_φ, deriv_B_φ=deriv_B_φ)
            J_θ = J_θ_func(γ=γ, B_φ=B_φ)
            J_φ = J_φ_func(γ=γ, B_θ=B_θ, deriv_B_r=deriv_B_r)

            deriv_E_r = deriv_E_r_func(
                γ=γ, v_r=v_r, v_φ=v_φ, B_r=B_r, B_φ=B_φ, η_O=η_O, η_A=η_A,
                η_H=η_H, b_r=b_r, b_θ=b_θ, b_φ=b_φ, J_r=J_r, J_φ=J_φ, J_θ=J_θ,
            )
        else:
            deriv_b_r, deriv_b_φ, deriv_b_θ = B_unit_derivs(
                B_r=B_r, B_φ=B_φ, B_θ=B_θ, deriv_B_r=deriv_B_r,
                deriv_B_φ=deriv_B_φ, deriv_B_θ=deriv_B_θ, b_r=b_r, b_θ=b_θ,
                b_φ=b_φ,
            )

            A = A_func(
                η_O=η_O, η_A=η_A, η_H=η_H, b_θ=b_θ, b_r=b_r, b_φ=b_φ,
                deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
                deriv_b_θ=deriv_b_θ, deriv_b_r=deriv_b_r, deriv_b_φ=deriv_b_φ,
            )

            dderiv_B_φ = dderiv_B_φ_soln(
                B_r=B_r, B_φ=B_φ, B_θ=B_θ, η_O=η_O, η_H=η_H, η_A=η_A, θ=θ,
                v_r=v_r, v_θ=v_θ, v_φ=v_φ, deriv_v_r=deriv_v_r,
                deriv_v_θ=deriv_v_θ, deriv_v_φ=deriv_v_φ, deriv_B_r=deriv_B_r,
                deriv_B_θ=deriv_B_θ, deriv_B_φ=deriv_B_φ, γ=γ,
                deriv_η_O=deriv_η_O, deriv_η_A=deriv_η_A, deriv_η_H=deriv_η_H,
                b_r=b_r, b_θ=b_θ, b_φ=b_φ, deriv_b_r=deriv_b_r,
                deriv_b_θ=deriv_b_θ, deriv_b_φ=deriv_b_φ, C=C, A=A,
            )

        derivs[..., ODEIndex.B_r] = deriv_B_r
        derivs[..., ODEIndex.B_φ] = deriv_B_φ
        derivs[..., ODEIndex.B_θ] = deriv_B_θ
        derivs[..., ODEIndex.v_r] = deriv_v_r
        derivs[..., ODEIndex.v_φ] = deriv_v_φ
        derivs[..., ODEIndex.v_θ] = deriv_v_θ
        derivs[..., ODEIndex.ρ] = deriv_ρ
        derivs[..., ODEIndex.η_O] = deriv_η_O
        derivs[..., ODEIndex.η_A] = deriv_η_A
        derivs[..., ODEIndex.η_H] = deriv_η_H

        if use_E_r:
            derivs[..., ODEIndex.E_r] = deriv_E_r
        else:
            derivs[..., ODEIndex.B_φ_prime] = dderiv_B_φ

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


def jump_across_sonic(
    *, base_solution, angles, system_initial_conditions, γ, a_0,
    norm_kepler_sq, η_derivs=True, θ_scale=float_type(1), use_E_r=False,
    jump_before_sonic, jump_after_sonic
):
    """
    Cross sonic point via a jump
    """
    if jump_after_sonic is None:
        jump_after_sonic = jump_before_sonic
    initial_angle = base_solution.values.t[-1]
    initial_values = base_solution.values.y[-1]

    system, _ = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=system_initial_conditions, η_derivs=η_derivs,
        store_internal=False, with_taylor=False, θ_scale=θ_scale,
        use_E_r=use_E_r,
    )
    derivs = zeros(len(ODEIndex), dtype=float_type)
    system(initial_angle, initial_values, derivs)

    dθ = (jump_before_sonic + jump_after_sonic) / derivs[ODEIndex.v_θ]
    final_angle = initial_angle + dθ

    post_sonic_angles = concatenate(
        ([final_angle], angles[angles > rad_to_scaled(final_angle, θ_scale)])
    )

    post_sonic_initial_conditions = initial_values + derivs * dθ

    return post_sonic_angles, post_sonic_initial_conditions


def taylor_solution(
    *, angles, init_con, γ, a_0, norm_kepler_sq, taylor_stop_angle,
    relative_tolerance=float_type(1e-6), absolute_tolerance=float_type(1e-10),
    max_steps=500, η_derivs=True, store_internal=True, θ_scale=float_type(1),
    use_E_r=False
):
    """
    Compute solution using taylor series
    """
    taylor_stop_angle = rad_to_scaled(taylor_stop_angle, θ_scale)
    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=init_con, with_taylor=True, η_derivs=η_derivs,
        store_internal=store_internal, θ_scale=θ_scale, use_E_r=use_E_r,
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
        ) from e
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
    η_derivs=True, θ_scale=float_type(1), use_E_r=False
):
    """
    Perform a single large step off the midplane using a taylor series
    expansion.
    """
    taylor_stop_angle = rad_to_scaled(taylor_stop_angle, θ_scale)
    new_angles = angles[angles > taylor_stop_angle]
    insert(new_angles, 0, taylor_stop_angle)

    first_order_taylor_values = get_taylor_first_order(
        init_con=init_con, γ=γ, a_0=a_0,
    )
    second_order_taylor_values = get_taylor_second_order(
        init_con=init_con, γ=γ, a_0=a_0, η_derivs=η_derivs, use_E_r=use_E_r,
    )
    third_order_taylor_values = get_taylor_third_order(
        init_con=init_con, γ=γ, a_0=a_0, η_derivs=η_derivs, use_E_r=use_E_r,
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


def interp_cross_sonic(
    *, base_solution, angles, system_initial_conditions, after_sonic, γ, a_0,
    norm_kepler_sq, base_internal_data, relative_tolerance=float_type(1e-6),
    absolute_tolerance=float_type(1e-10), max_steps=500, η_derivs=True,
    store_internal=True, θ_scale=float_type(1), use_E_r=False,
    interp_slice=slice(-1000, -100)
):
    """
    Cross sonic point interpolating deriv_v_θ
    """
    θ_interp = base_internal_data.angles[interp_slice]
    deriv_v_θ_interp = base_internal_data.derivs[interp_slice, ODEIndex.v_θ]
    try:
        deriv_v_θ_func = deduplicate_and_interpolate(
            θ_interp, deriv_v_θ_interp, kind='cubic',
            fill_value="extrapolate", assume_sorted=False,
        )
    except ValueError:
        log.exception("Failed to initialise interpolation")
        return None, None, None, None

    starting_angle = base_solution.values.t[-1]
    soln, internal_data = main_solution(
        system_initial_conditions=system_initial_conditions,
        root_func=velocity_stop_generator(1 + after_sonic), root_func_args=1,
        angles=angles[angles >= starting_angle],
        ode_initial_conditions=base_solution.values.y[-1], γ=γ, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        η_derivs=η_derivs, store_internal=store_internal, θ_scale=θ_scale,
        use_E_r=use_E_r, deriv_v_θ_func=deriv_v_θ_func,
    )
    new_angles = concatenate(
        ([soln.values.t[-1]], angles[angles > soln.values.t[-1]])
    )
    new_init_con = soln.values.y[-1]

    return soln, internal_data, new_angles, new_init_con


def main_solution(
    *, angles, system_initial_conditions, ode_initial_conditions, γ, a_0,
    norm_kepler_sq, relative_tolerance=float_type(1e-6),
    absolute_tolerance=float_type(1e-10), max_steps=500, onroot_func=None,
    jump_before_sonic=None, tstop=None, ontstop_func=None, η_derivs=True,
    store_internal=True, root_func=None, root_func_args=None,
    θ_scale=float_type(1), use_E_r=False, v_θ_sonic_crit=None,
    after_sonic=None, deriv_v_θ_func=None, sonic_interp_size=None
):
    """
    Find solution
    """
    extra_args = {}
    if sonic_interp_size is not None and root_func is not None:
        raise SolverError("Cannot use both sonic point interp and root_func")
    if sonic_interp_size is not None and jump_before_sonic is not None:
        raise SolverError("Cannot use two sonic point methods")
    if sonic_interp_size is not None and not store_internal:
        raise SolverError("Interpolation requires internal storage")
    if jump_before_sonic is not None and root_func is not None:
        raise SolverError("Cannot use both sonic point jumper and root_func")
    elif jump_before_sonic is not None and onroot_func is not None:
        raise SolverError("Cannot use both sonic point jumper and onroot_func")
    elif sonic_interp_size is not None:
        extra_args["rootfn"] = velocity_stop_generator(1 - sonic_interp_size)
        extra_args["nr_rootfns"] = 1
    elif jump_before_sonic is not None:
        extra_args["rootfn"] = velocity_stop_generator(1 - jump_before_sonic)
        extra_args["nr_rootfns"] = 1
    elif root_func is not None:
        extra_args["rootfn"] = root_func
        if root_func_args is not None:
            extra_args["nr_rootfns"] = root_func_args
        else:
            raise SolverError("Need to specify size of root array")

    if len(angles) < 2:
        raise SolverError(
            f"Insufficient angles to solve at - angles: {angles}"
        )

    if extra_args.get("root_func") is not None:
        log.warn("Root function specified")
    else:
        log.warn("Root function NOT specified")

    system, internal_data = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq,
        init_con=system_initial_conditions, η_derivs=η_derivs,
        store_internal=store_internal, with_taylor=False, θ_scale=θ_scale,
        use_E_r=use_E_r, v_θ_sonic_crit=v_θ_sonic_crit,
        after_sonic=after_sonic, deriv_v_θ_func=deriv_v_θ_func,
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
        log.notice("Found root at {}".format(
            degrees(scaled_to_rad(soln.roots.t, θ_scale))
        ))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        for tstop_scaled in soln.tstop.t:
            log.notice("Stopped at {}".format(
                degrees(scaled_to_rad(tstop_scaled, θ_scale))
            ))

    if internal_data is not None:
        internal_data._finalise()  # pylint: disable=protected-access

    return soln, internal_data


def solution(
    soln_input, initial_conditions, *, root_func=None, root_func_args=None,
    onroot_func=None, tstop=None, ontstop_func=None, store_internal=True,
    with_taylor=True, modified_initial_conditions=None, θ_scale=float_type(1),
    is_post_shock_only=False,
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
    jump_before_sonic = soln_input.jump_before_sonic
    v_θ_sonic_crit = soln_input.v_θ_sonic_crit
    after_sonic = soln_input.after_sonic
    interp_slice = soln_input.interp_slice
    sonic_interp_size = soln_input.sonic_interp_size
    use_E_r = soln_input.use_E_r
    taylor_internal = None

    if with_taylor:
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
                θ_scale=θ_scale, use_E_r=use_E_r,
            )

        else:
            taylor_soln = taylor_solution(
                angles=angles, init_con=init_con, γ=γ, a_0=a_0,
                norm_kepler_sq=norm_kepler_sq,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance, max_steps=max_steps,
                taylor_stop_angle=taylor_stop_angle, η_derivs=η_derivs,
                store_internal=store_internal, θ_scale=θ_scale,
                use_E_r=use_E_r,
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
        jump_before_sonic=jump_before_sonic, root_func=root_func,
        root_func_args=root_func_args, θ_scale=θ_scale, use_E_r=use_E_r,
        v_θ_sonic_crit=v_θ_sonic_crit, after_sonic=after_sonic,
        sonic_interp_size=sonic_interp_size,
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

    if sonic_interp_size is not None:
        int_soln, int_internal, int_angles, int_init_con = interp_cross_sonic(
            base_solution=soln, angles=post_taylor_angles,
            system_initial_conditions=init_con, γ=γ, a_0=a_0,
            norm_kepler_sq=norm_kepler_sq, η_derivs=η_derivs,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, max_steps=max_steps,
            θ_scale=θ_scale, use_E_r=use_E_r, store_internal=store_internal,
            interp_slice=interp_slice, after_sonic=(
                sonic_interp_size if after_sonic is None else after_sonic
            ), base_internal_data=internal_data,
        )
        if int_soln is not None:
            if store_internal and int_internal is not None:
                internal_data = internal_data + int_internal

            joined_angles = concatenate(
                (joined_angles, scaled_to_rad(int_soln.values.t, θ_scale))
            )
            joined_solution = concatenate(
                (joined_solution, int_soln.values.y)
            )

            post_interp_soln, post_interp_internal_data = main_solution(
                angles=int_angles, system_initial_conditions=init_con,
                ode_initial_conditions=int_init_con, γ=γ, a_0=a_0,
                norm_kepler_sq=norm_kepler_sq,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance, max_steps=max_steps,
                onroot_func=onroot_func, tstop=tstop,
                ontstop_func=ontstop_func, η_derivs=η_derivs,
                store_internal=store_internal, root_func=root_func,
                root_func_args=root_func_args, θ_scale=θ_scale,
                use_E_r=use_E_r,
            )

            if store_internal and post_interp_internal_data is not None:
                internal_data = internal_data + post_interp_internal_data

            joined_angles = concatenate(
                (joined_angles, scaled_to_rad(
                    post_interp_soln.values.t, θ_scale
                ))
            )
            joined_solution = concatenate(
                (joined_solution, post_interp_soln.values.y)
            )

    elif jump_before_sonic is not None:
        log.info("jumping over sonic point with jump size {}".format(
            jump_before_sonic
        ))
        post_jump_angles, post_jump_initial_conditions = jump_across_sonic(
            base_solution=soln, angles=post_taylor_angles,
            system_initial_conditions=init_con, γ=γ, a_0=a_0,
            norm_kepler_sq=norm_kepler_sq, η_derivs=η_derivs,
            jump_before_sonic=jump_before_sonic, θ_scale=θ_scale,
            use_E_r=use_E_r, jump_after_sonic=after_sonic
        )

        post_jump_soln, post_jump_internal_data = main_solution(
            angles=post_jump_angles, system_initial_conditions=init_con,
            ode_initial_conditions=post_jump_initial_conditions, γ=γ, a_0=a_0,
            norm_kepler_sq=norm_kepler_sq,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance, max_steps=max_steps,
            onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
            η_derivs=η_derivs, store_internal=store_internal,
            root_func=root_func, root_func_args=root_func_args,
            θ_scale=θ_scale, use_E_r=use_E_r, v_θ_sonic_crit=v_θ_sonic_crit,
        )

        if store_internal and post_jump_internal_data is not None:
            internal_data = internal_data + post_jump_internal_data

        joined_angles = concatenate(
            (joined_angles, scaled_to_rad(post_jump_soln.values.t, θ_scale))
        )
        joined_solution = concatenate(
            (joined_solution, post_jump_soln.values.y)
        )

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
        is_post_shock_only=is_post_shock_only,
    )


def get_known_broken_solution(inp):
    """
    Return a solution which can be seen to be invalid
    """
    # Flag being None and initial_conditions being None should be enough to
    # signify that the solution is invalid
    flag = None
    initial_conditions = None
    angles = array([0])
    soln = array([nan] * len(ODEIndex))
    soln.shape = (1, len(ODEIndex))

    return Solution(
        solution_input=inp, initial_conditions=initial_conditions,
        flag=flag, coordinate_system=COORDS, internal_data=None,
        angles=angles, solution=soln,
        t_roots=None, y_roots=None, sonic_point=None,
        sonic_point_values=None,
    )
