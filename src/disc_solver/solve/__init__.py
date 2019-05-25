# -*- coding: utf-8 -*-
"""
Solver component of Disc Solver
"""
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from .config import get_input_from_conffile
from .stepper import solver as stepper_solver
from .single import solver as single_solver
from .mcmc import solver as mcmc_solver
from .sonic_root import solver as sonic_root_solver
from .hydrostatic import solver as hydrostatic_solver
from .mod_hydro import solver as mod_hydro_solver
from .utils import add_solver_arguments, SolverError, validate_overrides

from .. import __version__ as ds_version
from ..file_format import registries, Run
from ..float_handling import float_type
from ..logging import log_handler
from ..utils import expanded_path, main_entry_point_wrapper

log = logbook.Logger(__name__)

SONIC_METHOD_MAP = {
    "step": stepper_solver,
    "single": single_solver,
    "mcmc": mcmc_solver,
    "sonic_root": sonic_root_solver,
    "hydrostatic": hydrostatic_solver,
    "mod_hydro": mod_hydro_solver,
}


def solve(
    *, output_file, sonic_method, config_file, output_dir, store_internal,
    overrides=None, use_E_r=False, **kwargs
):
    """
    Main function to generate solution
    """
    config_input = get_input_from_conffile(
        config_file=config_file, overrides=overrides
    )
    run = Run(
        config_input=config_input,
        config_filename=str(config_file),
        disc_solver_version=ds_version,
        float_type=str(float_type),
        sonic_method=sonic_method,
        use_E_r=use_E_r
    )

    if output_file is None:
        output_file = Path(config_input.label + str(arrow.now()) + ".hdf5")
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries, mode='x') as f:
        f["run"] = run
        sonic_solver = SONIC_METHOD_MAP.get(sonic_method)
        if sonic_solver is None:
            raise SolverError("No method chosen to cross sonic point")
        succeeded = sonic_solver(
            config_input.to_soln_input(), run,
            store_internal=store_internal, **kwargs
        )

    return output_file, succeeded


@main_entry_point_wrapper(description='Solver for DiscSolver')
def main(argv, parser):
    """
    Entry point for ds-soln
    """
    add_solver_arguments(parser)
    parser.add_argument("config_file", type=expanded_path)

    args = parser.parse_args(argv)

    overrides = validate_overrides(args.override)

    with log_handler(args), redirected_warnings(), redirected_logging():
        filename, succeeded = solve(
            output_file=args.output_file, sonic_method=args.sonic_method,
            config_file=args.config_file, output_dir=args.output_dir,
            store_internal=args.store_internal, overrides=overrides,
            use_E_r=args.use_E_r,
        )
        print(filename)
        return int(not succeeded)
