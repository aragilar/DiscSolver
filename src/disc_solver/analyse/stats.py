# -*- coding: utf-8 -*-
"""
stats command
"""
from csv import DictWriter
from sys import stdin

from logbook.compat import redirected_warnings, redirected_logging

from numpy import degrees, argmax, max as np_max, sqrt

from h5preserve import open as h5open

from ..file_format import SOLUTION_INPUT_FIELDS, registries
from ..logging import log_handler
from ..utils import open_or_stream, main_entry_point_wrapper, ODEIndex
from .utils import get_mach_numbers, get_all_sonic_points
from .phys_ratio_plot import compute_M_dot_out_on_M_dot_in, compute_Σ


# STATS
# validate_plot has things

FIELD_NAMES = list(SOLUTION_INPUT_FIELDS) + [
    "max_slow_mach",
    "max_slow_mach_angle_degrees",
    "max_sonic_mach",
    "max_sonic_mach_angle_degrees",
    "max_alfven_mach",
    "max_alfven_mach_angle_degrees",
    "max_fast_mach",
    "max_fast_mach_angle_degrees",
    "slow_sonic_point",
    "sonic_sonic_point",
    "alfven_sonic_point",
    "fast_sonic_point",
    "M_dot_out_on_M_dot_in",
    "Σ",
    "max_B_p_on_B_t",
    "filename",
    "solution_name",
]


def get_max_mach_numbers(solution):
    """
    Find the maximum value and angle for the mach numbers associated with the
    all the sound speeds.
    """
    def get_max_mach_degrees(mach):
        """
        Find the angle for maximum value for the mach numbers associated with
        the all the sound speeds.
        """
        return degrees(solution.angles[argmax(mach)])
    slow_mach, sonic_mach, alfven_mach, fast_mach = get_mach_numbers(solution)

    return {
        "max_slow_mach": np_max(slow_mach),
        "max_slow_mach_angle_degrees": get_max_mach_degrees(slow_mach),
        "max_sonic_mach": np_max(sonic_mach),
        "max_sonic_mach_angle_degrees": get_max_mach_degrees(sonic_mach),
        "max_alfven_mach": np_max(alfven_mach),
        "max_alfven_mach_angle_degrees": get_max_mach_degrees(alfven_mach),
        "max_fast_mach": np_max(fast_mach),
        "max_fast_mach_angle_degrees": get_max_mach_degrees(fast_mach),
    }


def compute_B_poloidal_vs_B_toroidal(solution, indexes=slice(None)):
    """
    Compute B_p/B_t (i.e. B_φ / sqrt(B_r^2 + B_θ^2))
    """
    soln = solution.solution
    return soln[indexes, ODEIndex.B_φ] / sqrt(
        soln[indexes, ODEIndex.B_r] ** 2 + soln[indexes, ODEIndex.B_θ] ** 2
    )


def labelled_get_all_sonic_points(solution):
    """
    Labels the output from get_all_sonic_points for writing to csv files.
    """
    slow_θ, sonic_θ, alfven_θ, fast_θ = get_all_sonic_points(solution)
    return {
        "slow_sonic_point": slow_θ,
        "sonic_sonic_point": sonic_θ,
        "alfven_sonic_point": alfven_θ,
        "fast_sonic_point": fast_θ,
    }


def singluar_stats(solution):
    """
    Labels all the singular valued properties of a solution.
    """
    M_dot_out_on_M_dot_in = compute_M_dot_out_on_M_dot_in(solution)
    Σ = compute_Σ(solution)
    max_B_p_on_B_t = max(compute_B_poloidal_vs_B_toroidal(solution))
    return {
        "M_dot_out_on_M_dot_in": M_dot_out_on_M_dot_in,
        "Σ": Σ,
        "max_B_p_on_B_t": max_B_p_on_B_t,
    }


STATS_FUNCS = [
    get_max_mach_numbers, labelled_get_all_sonic_points, singluar_stats
]


def compute_solution_stats(solution, metadata=None):
    """
    Compute all the statistics of a single solution.
    """
    stats = {}
    if metadata is not None:
        stats.update(**metadata)

    for func in STATS_FUNCS:
        stats.update(func(solution))
    return stats


def write_stats(solutions, *, output_file):
    """
    Write to csv file `output_file` the statistics of all the solutions
    """
    with open_or_stream(output_file, mode='a') as out:
        csvwriter = DictWriter(
            out, fieldnames=FIELD_NAMES, dialect="unix",
        )
        csvwriter.writeheader()
        for solution in solutions:
            filename, soln_name, soln = solution
            soln_info = {
                'filename': filename,
                'solution_name': soln_name,
            }
            if soln is not None:
                soln_info.update(soln.solution_input.asdict())
                csvwriter.writerow(compute_solution_stats(
                    soln, metadata=soln_info
                ))
                out.flush()


def get_all_solutions(files):
    """
    Load all the solutions from a file, including the final one, for every file
    in `files`.
    """
    for file in files:
        with h5open(file, registries, mode='r') as soln_file:
            for soln_name, soln in soln_file["run"].solutions.items():
                yield file, soln_name, soln
            yield file, "final", soln_file["run"].final_solution


def get_all_files(args):
    """
    Get files from cli arguments and/or stdin
    """
    if args.solution_files:
        for file in args.solution_files:
            yield file
    if args.with_stdin:
        yield stdin.readline().strip()


@main_entry_point_wrapper(description="compute statistics of the solutions")
def stats_main(argv, parser):
    """
    Main entry point for ds-stats
    """
    parser.add_argument("--file", default='-')
    parser.add_argument("--with-stdin", action='store_true', default=False)
    parser.add_argument("solution_files", nargs='*')
    args = parser.parse_args(argv)
    with log_handler(args), redirected_warnings(), redirected_logging():
        return write_stats(
            get_all_solutions(get_all_files(args)), output_file=args.file
        )
