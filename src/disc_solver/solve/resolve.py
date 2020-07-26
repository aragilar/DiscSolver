# -*- coding: utf-8 -*-
"""
reSolver component of Disc Solver
"""
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from . import SONIC_METHOD_MAP
from .config import new_inputs_with_overrides
from .utils import add_solver_arguments, SolverError, validate_overrides

from .. import __version__ as ds_version
from ..file_format import registries, Run
from ..float_handling import float_type
from ..logging import log_handler
from ..utils import expanded_path, get_solutions, main_entry_point_wrapper

log = logbook.Logger(__name__)


def resolve(
    *, output_file, sonic_method, soln_filename, soln_range, output_dir,
    store_internal, overrides=None, **kwargs
):
    """
    Main function to generate solution
    """
    with h5open(soln_filename, registries, mode='r') as soln_file:
        old_run = soln_file["run"]
        old_solution = get_solutions(old_run, soln_range)

    new_config_input, new_soln_input = new_inputs_with_overrides(
        config_input=old_run.config_input, overrides=overrides,
        solution_input=old_solution.solution_input,
    )

    run = Run(
        config_input=new_config_input,
        config_filename=old_run.config_filename,
        disc_solver_version=ds_version,
        float_type=str(float_type),
        sonic_method=sonic_method,
        use_E_r=new_soln_input.use_E_r
    )

    if output_file is None:
        output_file = Path(
            old_run.config_input.label + str(arrow.now()) + ".hdf5"
        )
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries, mode='x') as f:
        f["run"] = run
        sonic_solver = SONIC_METHOD_MAP.get(sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")
        succeeded = sonic_solver(
            new_soln_input, run, store_internal=store_internal, **kwargs
        )
        run.finalise()

    return output_file, succeeded


@main_entry_point_wrapper(description='reSolver for DiscSolver')
def main(argv, parser):
    """
    Entry point for ds-resoln
    """
    add_solver_arguments(parser)
    parser.add_argument("soln_filename", type=expanded_path)
    parser.add_argument("soln_range")

    args = parser.parse_args(argv)

    overrides = validate_overrides(args.override)

    with log_handler(args), redirected_warnings(), redirected_logging():
        filename, succeeded = resolve(
            soln_filename=args.soln_filename, soln_range=args.soln_range,
            output_file=args.output_file, sonic_method=args.sonic_method,
            config_file=args.config_file, output_dir=args.output_dir,
            store_internal=args.store_internal, overrides=overrides,
        )
        print(filename)
        return int(not succeeded)
