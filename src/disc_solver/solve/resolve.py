# -*- coding: utf-8 -*-
"""
reSolver component of Disc Solver
"""
import argparse
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from . import SONIC_METHOD_MAP
from .utils import add_solver_arguments, SolverError

from .. import __version__ as ds_version
from ..file_format import registries, Run
from ..float_handling import float_type
from ..logging import log_handler
from ..utils import expanded_path, get_solutions

log = logbook.Logger(__name__)


def resolve(
    *, output_file, sonic_method, soln_filename, soln_range, output_dir,
    store_internal, use_E_r=False
):
    """
    Main function to generate solution
    """
    with h5open(soln_filename, registries) as soln_file:
        old_run = soln_file["run"]
        old_solution = get_solutions(old_run, soln_range)

    run = Run(
        config_input=old_run.config_input,
        config_filename=old_run.config_filename,
        disc_solver_version=ds_version,
        float_type=str(float_type),
        sonic_method=sonic_method,
        use_E_r=use_E_r
    )

    if output_file is None:
        output_file = Path(
            old_run.config_input.label + str(arrow.now()) + ".hdf5"
        )
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries) as f:
        f["run"] = run
        sonic_solver = SONIC_METHOD_MAP.get(sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")
        sonic_solver(
            old_solution.solution_input, run,
            store_internal=store_internal,
        )

    return output_file


def main():
    """
    Entry point for ds-resoln
    """
    parser = argparse.ArgumentParser(description='reSolver for DiscSolver')
    add_solver_arguments(parser)
    parser.add_argument("soln_filename")
    parser.add_argument("soln_range")

    args = vars(parser.parse_args())

    soln_filename = expanded_path(args["soln_filename"])
    soln_range = args["soln_range"]
    output_dir = expanded_path(args["output_dir"])
    sonic_method = args["sonic_method"]
    output_file = args.get("output_file", None)
    store_internal = args.get("store_internal", True)
    use_E_r = args.get("use_E_r", False)

    with log_handler(args), redirected_warnings(), redirected_logging():
        print(resolve(
            soln_filename=soln_filename, soln_range=soln_range,
            sonic_method=sonic_method, output_dir=output_dir,
            store_internal=store_internal, output_file=output_file,
            use_E_r=use_E_r,
        ))
