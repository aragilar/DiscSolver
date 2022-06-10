# -*- coding: utf-8 -*-
"""
"""

from numpy import zeros

from ..float_handling import float_type
from ..solve.solution import ode_system
from ..utils import get_solutions
from .utils import (
    analyse_main_wrapper, analysis_func_wrapper,
)


def compute_jacobian(
    *, γ, a_0, norm_kepler_sq, init_con, θ_scale, η_derivs, use_E_r, θ, params,
    eps,
):
    rhs_eq, _ = ode_system(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, with_taylor=False, η_derivs=η_derivs,
        store_internal=False, use_E_r=use_E_r, v_θ_sonic_crit=None,
        after_sonic=None, deriv_v_θ_func=None, check_params=False
    )

    solution_length = params.shape[0]
    ode_size = params.shape[1]

    J = zeros([solution_length, ode_size, ode_size], dtype=float_type)

    # compute column for each param
    for i in range(ode_size):
        derivs_h = zeros([solution_length, ode_size], dtype=float_type)
        derivs_l = zeros([solution_length, ode_size], dtype=float_type)
        params_h = params.copy()
        params_l = params.copy()

        # offset only the value associated with this column
        params_h[:, i] += eps
        params_l[:, i] -= eps

        # we don't check the validity of inputs as we have these from the
        # solution
        rhs_eq(θ, params_h, derivs_h)
        rhs_eq(θ, params_l, derivs_l)

        J[:, :, i] = (derivs_h - derivs_l) / (2 * eps)
    return J


def compute_jacobian_from_solution(soln, *, eps, θ_scale=float_type(1)):
    solution = soln.solution
    angles = soln.angles
    cons = soln.initial_conditions
    soln_input = soln.solution_input

    init_con = cons.init_con
    γ = cons.γ
    a_0 = cons.a_0
    norm_kepler_sq = cons.norm_kepler_sq

    η_derivs = soln_input.η_derivs
    use_E_r = soln_input.use_E_r

    return compute_jacobian(
        γ=γ, a_0=a_0, norm_kepler_sq=norm_kepler_sq, init_con=init_con,
        θ_scale=θ_scale, η_derivs=η_derivs, use_E_r=use_E_r, θ=angles,
        params=solution, eps=eps,
    )


@analysis_func_wrapper
def compute_jacobian_from_file(
    soln_file, *, soln_range=None, eps, θ_scale=float_type(1), **kwargs
):

    soln = get_solutions(soln_file, soln_range)
    return compute_jacobian_from_solution(soln, eps=eps, θ_scale=θ_scale)
