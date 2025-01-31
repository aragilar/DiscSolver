"""
Support for calculating shocks
"""
from sys import stderr
from traceback import format_exc

import logbook
from numpy import (
    degrees, radians, all as np_all,
)

from h5preserve import open as h5open

from .solution import solution
from .utils import get_multiprocessing_filename, add_worker_arguments

from ..analyse.plot import plot as plot_solution
from ..analyse.utils import analyse_main_wrapper, analysis_func_wrapper
from ..file_format import registries
from ..utils import ODEIndex, get_solutions, nicer_mp_pool, expanded_path

log = logbook.Logger(__name__)


def get_v_θ_post(v_θ_pre):
    """
    Get v_θ post shock.
    """
    return 1 / v_θ_pre


def get_ρ_post(*, v_θ_pre, ρ_pre):
    """
    Get ρ post shock.
    """
    return ρ_pre * v_θ_pre ** 2


def get_shock_modified_initial_conditions(
    *, initial_conditions, angles, jump_θ, pre_jump_values
):
    """
    Get the modified copy of the initial conditions object for after the shock.
    """
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
    """
    Find possible angles at which to perform the shock.
    """
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


def print_err(desc, tb):
    """
    Print the traceback with a comment to stderr
    """
    print(desc, "START TB", file=stderr)
    print(tb, file=stderr)
    print(desc, "END TB", file=stderr)


# pylint: disable=too-few-public-methods
class ShockFinder:
    """
    wrapper class to handle aligning the shock computation with the needs of
    multiprocessing.
    """
    def __init__(
        self, *, new_solution_input, initial_conditions, angles,
        store_internal, output_dir="shock_tests", run,
    ):
        self.new_solution_input = new_solution_input
        self.initial_conditions = initial_conditions
        self.angles = angles
        self.output_dir = output_dir
        self.store_internal = store_internal
        self.run = run

    def __call__(self, search_value):
        # pylint: disable=broad-exception-caught
        jump_θ, pre_jump_values = search_value
        try:
            shock_initial_conditions = get_shock_modified_initial_conditions(
                initial_conditions=self.initial_conditions, angles=self.angles,
                jump_θ=jump_θ, pre_jump_values=pre_jump_values,
            )
        except Exception:
            print_err("initial conditions failed", format_exc())
            return None, None

        try:
            shock_soln = solution(
                self.new_solution_input, self.initial_conditions,
                store_internal=self.store_internal,
                modified_initial_conditions=shock_initial_conditions,
                with_taylor=False, is_post_shock_only=True,
            )
        except Exception:
            print_err("solve failed", format_exc())
            return None, None

        if shock_soln.angles.shape[0] < 5:
            # Not worth looking at
            return None, None

        try:
            output_hdf5_filename = expanded_path(
                self.output_dir, get_multiprocessing_filename(self.run)
            )
            with h5open(output_hdf5_filename, registries, mode='x') as f:
                f["run"] = self.run
                self.run.solutions.add_solution(shock_soln)
                self.run.finalise()
        except Exception:
            print_err("save failed", format_exc())
            return None, None

        plot_filename = output_hdf5_filename.with_suffix(".png")

        try:
            plot_shock_solution(
                self.run, plot_filename=plot_filename,
                output_hdf5_filename=output_hdf5_filename,
            )
        except Exception:
            print_err("plot failed", format_exc())
            return None, None

        if np_all(shock_soln.solution[:, ODEIndex.v_θ] < 1):
            print("\n\n\nSEE GOOD FILE", plot_filename, "\n\n\n")

        return output_hdf5_filename, plot_filename
# pylint: enable=too-few-public-methods


def compute_shock(
    soln, *, min_v_θ_pre=1.0, max_v_θ_pre=10, min_angle=40,
    store_internal=True, run, nworkers,
):
    """
    Compute the shock at all possible angles that would make sense.
    """

    soln_input = soln.solution_input.new_without_sonic_taylor()

    jumpable_angles, jumpable_values = find_shock_test_values(
        soln, min_v_θ_pre=min_v_θ_pre, max_v_θ_pre=max_v_θ_pre,
        min_angle=min_angle,
    )

    search_values = list(zip(
        jumpable_angles, jumpable_values
    ))
    print("possible values to check:", len(search_values))
    with nicer_mp_pool(nworkers) as pool:
        for soln_filename, fig_filename in pool.imap(ShockFinder(
            new_solution_input=soln_input,
            initial_conditions=soln.initial_conditions,
            angles=soln.angles,
            store_internal=store_internal, run=run,
        ), search_values):
            if fig_filename is not None:
                print("New plot", fig_filename, "for", soln_filename)


def plot_shock_solution(run, *, plot_filename, output_hdf5_filename):
    """
    Plot the solution post shock
    """
    return plot_solution.__wrapped__(
        run, soln_range="final", hide_roots=True, plot_filename=plot_filename,
        filename=output_hdf5_filename, figargs={"figsize": (20, 15)},
    )


def shock_parser(parser):
    """
    Parser for shock command
    """
    add_worker_arguments(parser)
    return parser


@analyse_main_wrapper(
    "Continue with shock from DiscSolver",
    shock_parser,
    cmd_parser_splitters={
        "nworkers": lambda args: args["nworkers"]
    }
)
def shock_main(soln_file, *, soln_range, nworkers):
    """
    ds-shock entry point
    """
    # pylint: disable=missing-kwoa
    return shock_from_file(soln_file, soln_range=soln_range, nworkers=nworkers)


@analysis_func_wrapper
def shock_from_file(run, *, soln_range, filename, nworkers):
    """
    Primary shock function
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    print("Looking at", filename, "solution", soln_range)
    soln = get_solutions(run, soln_range)
    shock_run = run.create_shock_run(
        filename=filename, solution_name=soln_range,
    )
    compute_shock(soln, run=shock_run, nworkers=nworkers)
