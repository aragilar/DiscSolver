from multiprocessing import current_process
from pathlib import Path

import arrow
import logbook
import matplotlib.pyplot as plt
from numpy import (
    degrees, radians, all as np_all,
)

from .solution import solution

from ..analyse.plot import generate_plot
from ..analyse.utils import analyse_main_wrapper, analysis_func_wrapper
from ..utils import ODEIndex, get_solutions, nicer_mp_pool, expanded_path

log = logbook.Logger(__name__)


def get_v_θ_post(v_θ_pre):
    return 1 / v_θ_pre


def get_ρ_post(*, v_θ_pre, ρ_pre):
    return ρ_pre * v_θ_pre ** 2


def get_shock_modified_initial_conditions(
    *, initial_conditions, angles, jump_θ, pre_jump_values
):
    v_θ_pre = pre_jump_values[ODEIndex.v_θ]
    ρ_pre = pre_jump_values[ODEIndex.ρ]
    v_θ_post = get_v_θ_post(v_θ_pre)
    ρ_post = get_ρ_post(v_θ_pre=v_θ_pre, ρ_pre=ρ_pre)
    post_jump_values = pre_jump_values.copy()
    post_jump_values[ODEIndex.v_θ] = v_θ_post
    post_jump_values[ODEIndex.ρ] = ρ_post

    print("v_θ", v_θ_pre, "angle", degrees(jump_θ))

    post_jump_angles = angles[angles >= jump_θ]
    return initial_conditions.create_modified(
        init_con=post_jump_values, angles=post_jump_angles,
    )


def find_shock_test_values(
    soln, *, min_v_θ_pre=1, max_v_θ_pre=10, min_angle=40,
):
    values = soln.solution
    angles = soln.angles
    v_θ = values[:, ODEIndex.v_θ]
    jumpable_slice = (
        (v_θ > min_v_θ_pre) &
        (v_θ < max_v_θ_pre) &
        (angles > radians(min_angle))
    )
    jumpable_angles = angles[jumpable_slice]
    jumpable_values = values[jumpable_slice]
    return jumpable_angles, jumpable_values


class ShockFinder:
    def __init__(
        self, *, new_solution_input, initial_conditions, angles,
        store_internal, plot_path="shock_plots",
    ):
        self.new_solution_input = new_solution_input
        self.initial_conditions = initial_conditions
        self.angles = angles
        self.plot_path = plot_path
        self.store_internal = store_internal

    def __call__(self, search_value):
        jump_θ, pre_jump_values = search_value
        try:
            shock_initial_conditions = get_shock_modified_initial_conditions(
                initial_conditions=self.initial_conditions, angles=self.angles,
                jump_θ=jump_θ, pre_jump_values=pre_jump_values,
            )
        except Exception as e:
            print("initial conditions failed", e)
            return None

        try:
            shock_soln = solution(
                self.new_solution_input, self.initial_conditions,
                store_internal=self.store_internal,
                modified_initial_conditions=shock_initial_conditions,
                with_taylor=False,
            )
        except Exception as e:
            print("solve failed", e)
            return None
        if shock_soln.angles.shape[0] < 5:
            # Not worth looking at
            return None

        try:
            fig_filename = plot_shock_solution(
                shock_soln, plot_path=self.plot_path,
            )
        except Exception as e:
            print("plot failed", e)
            return None

        if np_all(shock_soln.solution[:, ODEIndex.v_θ] < 1):
            print("\n\n\nSEE GOOD FILE", fig_filename, "\n\n\n")

        return fig_filename


def compute_shock(
    soln, *, min_v_θ_pre=1.1, max_v_θ_pre=10, min_angle=40,
    store_internal=True,
):

    soln_input = soln.solution_input.new_without_sonic_taylor()

    jumpable_angles, jumpable_values = find_shock_test_values(
        soln, min_v_θ_pre=min_v_θ_pre, max_v_θ_pre=max_v_θ_pre,
        min_angle=min_angle,
    )

    search_values = list(zip(
        jumpable_angles, jumpable_values
    ))
    print("possible values to check:", len(search_values))
    with nicer_mp_pool() as pool:
        for fig_filename in pool.imap(ShockFinder(
            new_solution_input=soln_input,
            initial_conditions=soln.initial_conditions,
            angles=soln.angles,
            store_internal=store_internal
        ), search_values):
            if fig_filename is not None:
                print("New plot", fig_filename)


def plot_shock_solution(soln, *, plot_path):

    process_name = current_process().name
    filename = expanded_path(plot_path, Path(
        str(arrow.now()) + str(process_name) + ".png"
    ))
    fig = plt.figure(constrained_layout=True)
    generate_plot.__wrapped__(fig, soln, hide_roots=True)
    fig.savefig(filename)
    return filename


def shock_parser(parser):
    return parser


@analyse_main_wrapper(
    "Continue with shock from DiscSolver",
    shock_parser,
    cmd_parser_splitters={
        # "group": lambda args: args["group"]
    }
)
def shock_main(soln_file, *, soln_range):
    return shock_from_file(soln_file, soln_range=soln_range)


@analysis_func_wrapper
def shock_from_file(soln_file, *, soln_range, filename):
    soln = get_solutions(soln_file, soln_range)
    compute_shock(soln)
