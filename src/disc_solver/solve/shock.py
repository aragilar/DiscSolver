import logbook

import matplotlib.pyplot as plt
from numpy import (
    array, concatenate, copy, insert, errstate, sqrt, tan, degrees, radians,
    zeros, nan
)

from .solution import solution

from ..analyse.plot import generate_plot
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
    values = soln.solution
    angles = soln.angles

    initial_conditions = soln.initial_conditions

    soln_input = soln.solution_input
    soln_input.use_taylor_jump = False
    soln_input.jump_before_sonic = False
    soln_input.v_θ_sonic_crit = None
    soln_input.after_sonic = None
    soln_input.interp_range = None
    soln_input.interp_slice = None
    soln_input.sonic_interp_size = None

    v_θ = values[:, ODEIndex.v_θ]
    jumpable_slice = (
        (v_θ > min_v_θ_pre) &
        (v_θ < max_v_θ_pre) &
        (angles > radians(min_angle))
    )
    jumpable_angles = angles[jumpable_slice]
    jumpable_values = values[jumpable_slice]
    search_values = list(zip(
        jumpable_angles, jumpable_values
    ))
    print("possible values to check:", len(search_values))
    for i, (jump_θ, pre_jump_values) in enumerate(search_values):
        v_θ_pre = pre_jump_values[ODEIndex.v_θ]
        print("v_θ", v_θ_pre, "for iteration", i, "angle", degrees(jump_θ))
        ρ_pre = pre_jump_values[ODEIndex.ρ]
        v_θ_post = get_v_θ_post(v_θ_pre)
        ρ_post = get_ρ_post(v_θ_pre=v_θ_pre, ρ_pre=ρ_pre)
        post_jump_values = pre_jump_values.copy()
        post_jump_values[ODEIndex.v_θ] = v_θ_post
        post_jump_values[ODEIndex.ρ] = ρ_post

        post_jump_angles = angles[angles >= jump_θ]
        modified_initial_conditions = initial_conditions.create_modified(
            init_con=post_jump_values, angles=post_jump_angles,
        )

        soln = solution(
            soln_input, initial_conditions, 
            root_func=root_func, root_func_args=root_func_args, 
            onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
            store_internal=store_internal, with_taylor=False, θ_scale=θ_scale,
            modified_initial_conditions=modified_initial_conditions,
        )
        if soln.angles.shape[0] < 5:
            # Not worth looking at
            continue

        fig = plt.figure(constrained_layout=True)
        generate_plot.__wrapped__(fig, soln, hide_roots=True)
        fig.savefig(f"shock_{i}.png")

        yield soln
