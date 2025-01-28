import logbook

import matplotlib.pyplot as plt
from numpy import (
    array, concatenate, copy, insert, errstate, sqrt, tan, degrees, radians,
    zeros, nan
)

from .solution import main_solution

from ..analysis.plot import generate_plot
from ..float_handling import float_type
from ..utils import ODEIndex

log = logbook.Logger(__name__)


def get_v_θ_post(v_θ_pre):
    return 1 / v_θ_pre


def get_ρ_post(*, v_θ_pre, ρ_pre):
    return ρ_pre * v_θ_pre ** 2

def compute_shock(
    soln, *, min_v_θ_pre=1.1, max_v_θ_pre=10, min_angle=40,
    root_func=None, root_func_args=None, onroot_func=None, tstop=None,
    ontstop_func=None, store_internal=True, θ_scale=float_type(1),
):
    solution = soln.solution
    angles = soln.angles

    initial_conditions = soln.initial_conditions
    init_con = initial_conditions.init_con
    γ = initial_conditions.γ
    a_0 = initial_conditions.a_0
    norm_kepler_sq = initial_conditions.norm_kepler_sq

    soln_input = soln.solution_input
    absolute_tolerance = soln_input.absolute_tolerance
    relative_tolerance = soln_input.relative_tolerance
    max_steps = soln_input.max_steps
    η_derivs = soln_input.η_derivs
    use_E_r = soln_input.use_E_r

    v_θ = solution[:, ODEIndex.v_θ]
    jumpable_slice = (
        (v_θ > min_v_θ_pre) &
        (v_θ < max_v_θ_pre) &
        (angles > radians(min_angle))
    )
    jumpable_angles = angles[jumpable_slice]
    jumpable_values = solution[jumpable_slice]

    jump_θ = jumpable_angles[0]
    pre_jump_values = jumpable_values[0]
    v_θ_pre = pre_jump_values[ODEIndex.v_θ]
    ρ_pre = pre_jump_values[ODEIndex.ρ]
    v_θ_post = get_v_θ_post(v_θ_pre)
    ρ_post = get_ρ_post(*, v_θ_pre=v_θ_pre, ρ_pre=ρ_pre)
    post_jump_values = pre_jump_values.copy()
    post_jump_values[ODEIndex.v_θ] = v_θ_post
    post_jump_values[ODEIndex.ρ] = ρ_post

    post_jump_angles = angles[angles >= jump_θ]



    soln, internal_data = main_solution(
        angles=post_jump_angles, system_initial_conditions=init_con,
        ode_initial_conditions=post_jump_values, γ=γ, a_0=a_0,
        norm_kepler_sq=norm_kepler_sq, relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        η_derivs=η_derivs, store_internal=store_internal,
        root_func=root_func, root_func_args=root_func_args, θ_scale=θ_scale,
        use_E_r=use_E_r,
    )

    fig = plt.figure(constrained_layout=True)
    generate_plot.__wrapped__(fig, soln)
    plt.show()
