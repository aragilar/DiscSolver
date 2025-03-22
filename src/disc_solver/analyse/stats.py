# -*- coding: utf-8 -*-
"""
stats command
"""
from csv import DictWriter
from functools import wraps
from sys import stdin, exc_info

import logbook
from logbook.compat import redirected_warnings, redirected_logging

from numpy import degrees, nanargmax, nanmax, sqrt, nan

from h5preserve import open as h5open

from ..critical_points import get_mach_numbers, get_all_sonic_points
from ..file_format import SOLUTION_INPUT_FIELDS, registries
from ..logging import log_handler
from ..utils import (
    open_or_stream, main_entry_point_wrapper, ODEIndex, CylindricalODEIndex,
    convert_spherical_to_cylindrical, is_monotonically_increasing,
)
from .phys_ratio_plot import compute_M_dot_out_on_M_dot_in, compute_Σ
from .validate_plot import (
    validate_continuity, validate_solenoid, validate_radial_momentum,
    validate_polar_momentum, validate_azimuthal_mometum,
    validate_polar_induction, validate_E_φ, get_values,
)

log = logbook.Logger(__name__)


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
    "max_jet_v_z",
    "max_jet_v_z_height",
    "max_diff_continuity",
    "max_diff_solenoid",
    "max_diff_radial_momentum",
    "max_diff_azimuthal_momentum",
    "max_diff_polar_momentum",
    "max_diff_polar_induction",
    "max_diff_E_φ",
    "filename",
    "solution_name",
    "max_angle_reached_degrees",
    "v_θ_starts_monotonic",
    "v_θ_ends_low",
]


def compute_max_difference_in_equations(solution):
    """
    Use validity calculations to determine worst fit values.
    """
    values = get_values(solution)
    return {
        "max_diff_continuity": max(validate_continuity(
            values.initial_conditions, values
        )),
        "max_diff_solenoid": max(validate_solenoid(
            values.initial_conditions, values
        )),
        "max_diff_radial_momentum": max(validate_radial_momentum(
            values.initial_conditions, values
        )),
        "max_diff_azimuthal_momentum": max(validate_azimuthal_mometum(
            values.initial_conditions, values
        )),
        "max_diff_polar_momentum": max(validate_polar_momentum(
            values.initial_conditions, values
        )),
        "max_diff_polar_induction": max(validate_polar_induction(
            values.initial_conditions, values
        )),
        "max_diff_E_φ": max(validate_E_φ(
            values.initial_conditions, values
        )),
    }


def compute_max_vert_jet_velocity(soln):
    """
    Find the maximum v_z velocity of the jet, and its height
    """
    solution = soln.solution
    angles = soln.angles
    inp = soln.solution_input

    heights, vert_soln = convert_spherical_to_cylindrical(
        angles, solution, γ=inp.γ, c_s_on_v_k=inp.c_s_on_v_k, use_E_r=False,
    )
    max_v_z_idx = nanargmax(vert_soln[:, CylindricalODEIndex.v_z])
    max_v_z = vert_soln[max_v_z_idx, CylindricalODEIndex.v_z]
    max_v_z_height = heights[max_v_z_idx]
    return {
        "max_jet_v_z": max_v_z,
        "max_jet_v_z_height": max_v_z_height,
    }


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
        try:
            idx = nanargmax(mach)
        except ValueError:
            return nan
        return degrees(solution.angles[idx])

    slow_mach, sonic_mach, alfven_mach, fast_mach = get_mach_numbers(solution)

    return {
        "max_slow_mach": nanmax(slow_mach),
        "max_slow_mach_angle_degrees": get_max_mach_degrees(slow_mach),
        "max_sonic_mach": nanmax(sonic_mach),
        "max_sonic_mach_angle_degrees": get_max_mach_degrees(sonic_mach),
        "max_alfven_mach": nanmax(alfven_mach),
        "max_alfven_mach_angle_degrees": get_max_mach_degrees(alfven_mach),
        "max_fast_mach": nanmax(fast_mach),
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


def is_v_θ_monotonic(solution, *, upto_degrees=20):
    """
    Check if v_θ is monotonic up to `upto_degrees`.
    """
    to_check_slice = degrees(solution.angles) <= upto_degrees
    return is_monotonically_increasing(
        solution.solution[to_check_slice, ODEIndex.v_θ]
    )


def does_v_θ_end_low(solution, *, cutoff=10):
    """
    Check if v_θ ends below the value of cutoff.
    """
    return solution.solution[-1, ODEIndex.v_θ] <= cutoff


def singluar_stats(solution):
    """
    Labels all the singular valued properties of a solution.
    """
    M_dot_out_on_M_dot_in = compute_M_dot_out_on_M_dot_in(solution)
    Σ = compute_Σ(solution)
    max_B_p_on_B_t = max(compute_B_poloidal_vs_B_toroidal(solution))
    max_angle_reached = degrees(nanmax(solution.angles))
    v_θ_monotonic = is_v_θ_monotonic(solution)
    v_θ_ends_low = does_v_θ_end_low(solution)
    return {
        "M_dot_out_on_M_dot_in": M_dot_out_on_M_dot_in,
        "Σ": Σ,
        "max_B_p_on_B_t": max_B_p_on_B_t,
        "max_angle_reached_degrees": max_angle_reached,
        "v_θ_starts_monotonic": v_θ_monotonic,
        "v_θ_ends_low": v_θ_ends_low,
    }


STATS_FUNCS = [
    get_max_mach_numbers, labelled_get_all_sonic_points, singluar_stats,
    compute_max_vert_jet_velocity, compute_max_difference_in_equations,
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
                try:
                    csvwriter.writerow(compute_solution_stats(
                        soln, metadata=soln_info
                    ))
                except ValueError:
                    log.exception(
                        "Failed to collect stats for {} {}",
                        filename, soln_name,
                    )
                out.flush()
                del soln


def write_inputs(solutions, *, output_file):
    """
    Write to csv file `output_file` the statistics of all the solutions
    """
    with open_or_stream(output_file, mode='a') as out:
        csvwriter = DictWriter(
            out, fieldnames=SOLUTION_INPUT_FIELDS, dialect="unix",
        )
        csvwriter.writeheader()
        for solution in solutions:
            filename, soln_name, soln = solution
            log.info("Writing {} solution {}", filename, soln_name)
            if soln is not None:
                try:
                    csvwriter.writerow(soln.solution_input.asdict())
                except ValueError:
                    log.exception(
                        "Failed to write input for {} {}",
                        filename, soln_name,
                    )
                out.flush()
                del soln


def get_all_solutions(files):
    """
    Load all the solutions from a file, including the final one, for every file
    in `files`.
    """
    for file in files:
        with log.catch_exceptions(
            "Failed to read stats from file {}", file,
        ):
            with h5open(file, registries, mode='r') as soln_file:
                try:
                    run = soln_file["run"]
                    for soln_name, soln in run.solutions.items():
                        yield file, soln_name, soln
                    yield file, "final", run.final_solution
                except KeyError:
                    log.notice(
                        "Failed to read stats from file {}", file,
                        exc_info=exc_info(),
                    )


def get_all_files(args):
    """
    Get files from cli arguments and/or stdin
    """
    if args.solution_files:
        yield from args.solution_files
    if args.with_stdin:
        for line in stdin:
            yield line.strip()


def many_file_wrapper(*, description, **kwargs):
    """
    Wrapper to handle passing in many files, both from stdin and via arguments
    to the command.
    """
    def decorator(main_func):
        """
        Function decorator for many_file_wrapper.
        """
        @main_entry_point_wrapper(description=description, **kwargs)
        @wraps(main_func)
        def many_file_main(argv, parser):
            """
            Logic for many_file_wrapper.
            """
            parser.add_argument("--file", default='-')
            parser.add_argument(
                "--with-stdin", action='store_true', default=False
            )
            parser.add_argument("solution_files", nargs='*')
            args = parser.parse_args(argv)
            with (
                log_handler(args), redirected_warnings(), redirected_logging(),
            ):
                return main_func(args)
        return many_file_main
    return decorator


@many_file_wrapper(description="compute statistics of the solutions")
def stats_main(args):
    """
    Main entry point for ds-stats
    """
    return write_stats(
        get_all_solutions(get_all_files(args)), output_file=args.file
    )


@many_file_wrapper(description="Dump out all the solution inputs of files")
def dump_csv_inputs_main(args):
    """
    Main entry point for ds-dumpcsv
    """
    return write_inputs(
        get_all_solutions(get_all_files(args)), output_file=args.file
    )
